import torch
from torch import nn

from ....registry import register_model
from ....utils import reverse_string_digits
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
    loss_func_name = "ctc"
    loss_func_kwargs = {"zero_infinity": True}

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
        labels_lengths = torch.count_nonzero(labels, dim=1).flatten()
        labels = labels[labels != self.config.blank_id]
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

        loss = self.loss_func(logits, labels, input_lengths, labels_lengths)

        return loss

    def generate(self, pixel_values):
        logits = self(pixel_values)["logits"]
        output_ids = ctc_decode(logits, blank=self.config.blank_id)
        probs, values = logits.permute(1, 0, 2).softmax(2).max(2)
        mean_probs = probs.mean(1)
        return {"generated_ids": output_ids, "scores": mean_probs}

    def preprocess(self, inputs, **kwargs):
        image_processor = self.preprocessor[self.image_processor]
        processed_outputs = image_processor(inputs, **kwargs)
        return processed_outputs

    def post_process(self, generation_outputs, return_scores=False):
        if isinstance(generation_outputs, torch.Tensor):
            generated_ids = generation_outputs.clone().detach()
            scores = torch.zeros(generated_ids.shape)
        else:
            generated_ids, scores = generation_outputs.values()

        outputs = []
        generated_ids = generated_ids.cpu().numpy().tolist()
        scores = scores.cpu().numpy().tolist()
        for decoded_ids, score in zip(generated_ids, scores):
            chars = [self.config.id2label[id_] for id_ in decoded_ids]
            text = "".join(chars)
            if self.config.reverse_output_digits:
                text = reverse_string_digits(text)
            if return_scores:
                outputs.append(Image2TextOutput(text=text, score=score))
            else:
                outputs.append(Image2TextOutput(text=text))
        return outputs


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
