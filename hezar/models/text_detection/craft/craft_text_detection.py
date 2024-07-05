from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ....constants import Backends
from ....registry import register_model
from ...model import Model
from ...model_outputs import TextDetectionOutput
from .craft_text_detection_config import CraftTextDetectionConfig
from .craft_utils import adjust_result_coordinates, get_detection_boxes, polys2boxes


_required_backends = [
    Backends.OPENCV,
    Backends.PILLOW,
    Backends.TORCHVISION,
]


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


@register_model("craft_text_detection", config_class=CraftTextDetectionConfig)
class CraftTextDetection(Model):
    """
    CRAFT for text detection. Copied from the original implementation at https://github.com/clovaai/CRAFT-pytorch
    """
    required_backends = _required_backends

    def __init__(self, config: CraftTextDetectionConfig, **kwargs):
        super(CraftTextDetection, self).__init__(config=config, **kwargs)

        """ Base network """
        self.basenet = VGG16BN()

        """ U network """
        self.upconv1 = DoubleConv(1024, 512, 256)
        self.upconv2 = DoubleConv(512, 256, 128)
        self.upconv3 = DoubleConv(256, 128, 64)
        self.upconv4 = DoubleConv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, pixel_values, ratio_values=None):
        """ Base network """
        sources = self.basenet(pixel_values)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature).permute(0, 2, 3, 1)

        return {"logits": y, "ratio_values": ratio_values}

    def preprocess(self, inputs, **kwargs):
        processed_outputs = self.preprocessor.image_processor(inputs, **kwargs)
        return processed_outputs

    def post_process(
        self,
        model_outputs: dict,
        text_threshold: float = None,
        link_threshold: float = None,
        low_text: float = None,
        poly: bool = False,
    ):
        text_threshold = text_threshold or self.config.text_threshold
        link_threshold = link_threshold or self.config.link_threshold
        low_text = low_text or self.config.low_text

        logits = model_outputs["logits"]
        ratio_values = model_outputs["ratio_values"]

        results = []

        for output, ratio in zip(logits, ratio_values):
            # make score and link map
            score_text = output[:, :, 0].cpu().data.numpy()
            score_link = output[:, :, 1].cpu().data.numpy()

            # Post-processing
            boxes, polys, mapper = get_detection_boxes(
                score_text,
                score_link,
                text_threshold,
                link_threshold,
                low_text,
                poly,
            )

            # coordinate adjustment
            boxes = adjust_result_coordinates(boxes, ratio, ratio)
            for k in range(len(polys)):
                if polys[k] is None:
                    polys[k] = boxes[k]
            boxes = polys2boxes(polys)
            results.append(TextDetectionOutput(boxes=boxes))
        return results


class VGG16BN(nn.Module):
    def __init__(self):
        super(VGG16BN, self).__init__()
        vgg_pretrained_features = models.vgg16_bn().features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(12):  # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):  # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):  # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):  # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # fc6, fc7 without atrous conv
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )

        init_weights(self.slice1.modules())
        init_weights(self.slice2.modules())
        init_weights(self.slice3.modules())
        init_weights(self.slice4.modules())
        init_weights(self.slice5.modules())  # no pretrained model for fc6 and fc7

    def forward(self, x):
        h = self.slice1(x)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
