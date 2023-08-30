from torch import nn

from ...model import Model
from .crnn_image2text_config import CRNNImage2TextConfig
from ....registry import register_model


@register_model("crnn_image2text", config_class=CRNNImage2TextConfig)
class CRNNImage2Text(Model):
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
            ConvBlock(512, 512, 2, 1, 0)
        )
        # map CNN to sequence
        self.map2seq = nn.Linear(
            512 * (self.config.img_height // 16 - 1), self.config.map2seq_dim)
        # RNN
        self.rnn1 = nn.LSTM(self.config.map2seq_dim, self.config.rnn_dim, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * self.config.rnn_dim, self.config.rnn_dim, bidirectional=True)
        # classifier
        self.classifier = nn.Linear(2 * self.config.rnn_dim, len(self.config.id2label))

    def forward(self, inputs, **kwargs):
        # CNN block
        x = self.cnn(inputs)
        # reformat array
        batch, channel, height, width = x.size()
        x = x.view(batch, channel * height, width)
        x = x.permute(2, 0, 1)
        x = self.map2seq(x)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, 2)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        input_channel,
        output_channel,
        kernel_sizes,
        strides,
        paddings,
        batch_norm: bool = False
    ):
        super(ConvBlock, self).__init__()
        self.do_batch_norm = batch_norm
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_sizes, strides, paddings)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.do_batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        return x
