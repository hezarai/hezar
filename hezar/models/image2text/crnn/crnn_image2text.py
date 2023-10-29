import torch
from torch import nn

from ....registry import register_model
from ...model import Model
from ...model_outputs import Image2TextOutput
from .crnn_decode_utils import ctc_decode
from .crnn_image2text_config import CRNNImage2TextConfig


@register_model("crnn_image2text", config_class=CRNNImage2TextConfig)
class CRNNImage2Text(Model):
    """
    A robust CRNN model for character level OCR based on the original paper.
    """

    is_generative = True
    image_processor = "image_processor"
    loss_fn_name = "ctc"

    def __init__(self, config: CRNNImage2TextConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.cnn = nn.Sequential(
            ConvBlock(self.config.n_channels, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 256, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 1)),
            ConvBlock(256, 512, 3, 1, 1, batch_norm=True),
            ConvBlock(512, 512, 3, 1, 1, batch_norm=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            ConvBlock(512, 512, 3, 1, 1),
        )
        # map CNN to sequence
        self.map2seq = nn.Linear(self.config.map2seq_in_dim, self.config.map2seq_out_dim)
        # RNN
        self.rnn1 = nn.LSTM(self.config.map2seq_out_dim, self.config.rnn_dim, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * self.config.rnn_dim, self.config.rnn_dim, bidirectional=True)
        # classifier
        self.classifier = nn.Linear(2 * self.config.rnn_dim, len(self.config.id2label))

    def forward(self, pixel_values, **kwargs):
        # CNN block
        x = self.cnn(pixel_values)
        # reformat array
        batch, channel, height, width = x.size()
        x = x.view(batch, channel * height, width)
        x = x.permute(2, 0, 1)
        x = self.map2seq(x)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, 2)
        outputs = {"logits": x}
        return outputs

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        batch_size = logits.size(1)
        labels_lengths = torch.LongTensor([len(t) for t in labels])
        labels = labels.flatten()
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

        loss = self.criterion(logits, labels, input_lengths, labels_lengths) / batch_size

        return loss

    def generate(self, pixel_values, **kwargs):
        logits = self(pixel_values)["logits"]
        output_ids = ctc_decode(logits, blank=self.config.blank_id)
        return output_ids

    def preprocess(self, inputs, **kwargs):
        image_processor = self.preprocessor[self.image_processor]
        processed_outputs = image_processor(inputs, **kwargs)
        return processed_outputs

    def post_process(self, model_outputs: torch.Tensor, **kwargs):
        texts = []
        generated_ids = model_outputs.cpu().numpy().tolist()
        for decoded_ids in generated_ids:
            chars = [self.config.id2label[id_] for id_ in decoded_ids]
            text = "".join(chars)
            if self.config.reverse_prediction_text:
                text = text[::-1]
            texts.append(text)
        return Image2TextOutput(texts=texts)


class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_sizes, strides, paddings, batch_norm: bool = False):
        super(ConvBlock, self).__init__()
        self.do_batch_norm = batch_norm
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_sizes, strides, paddings)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.do_batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        return x
