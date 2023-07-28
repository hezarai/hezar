"""
A RoBERTa Language Model (HuggingFace Transformers) wrapped by a Hezar Model class
"""
from typing import List, Union

from torch import nn, tanh
from transformers import RobertaConfig, RobertaModel

from ....models import Model
from ....registry import register_model
from .roberta_text_classification_config import RobertaTextClassificationConfig


@register_model("roberta_text_classification", config_class=RobertaTextClassificationConfig)
class RobertaTextClassification(Model):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.roberta = RobertaModel(self._build_inner_config(), add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(self.config)

    def _build_inner_config(self):
        if self.config.num_labels is None and self.config.id2label is None:
            raise ValueError("Both `num_labels` and `id2label` are None. Please provide at least one of them!")
        if self.config.id2label and self.config.num_labels is None:
            self.config.num_labels = len(self.config.id2label)
        bert_config = RobertaConfig(**self.config)
        return bert_config

    def forward(self, inputs, **kwargs):
        input_ids = inputs.get("token_ids")
        attention_mask = inputs.get("attention_mask", None)
        head_mask = inputs.get("head_mask", None)
        inputs_embeds = inputs.get("inputs_embeds", None)
        output_attentions = inputs.get("output_attentions", None)
        output_hidden_states = inputs.get("output_hidden_states", None)

        lm_outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        sequence_output = lm_outputs[0]
        logits = self.classifier(sequence_output)

        outputs = {
            "logits": logits,
            "hidden_states": lm_outputs.hidden_states,
            "attentions": lm_outputs.attentions,
        }
        return outputs

    def preprocess(self, inputs: Union[str, List[str]], **kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]
        if "normalizer" in self.preprocessor:
            normalizer = self.preprocessor["normalizer"]
            inputs = normalizer(inputs)
        tokenizer = self.preprocessor["bpe_tokenizer"]
        inputs = tokenizer(inputs, return_tensors="pt", device=self.device)
        return inputs

    def post_process(self, inputs, **kwargs):
        logits = inputs["logits"]
        predictions = logits.argmax(1)
        predictions_probs = logits.softmax(1).max(1)
        outputs = {"labels": [], "probs": []}
        for prediction, prob in zip(predictions, predictions_probs):
            label = self.config.id2label[prediction.item()]
            outputs["labels"].append(label)
            outputs["probs"].append(prob.item())
        return outputs


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs, **kwargs):
        x = inputs[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
