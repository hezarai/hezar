"""
A DistilBERT model for text classification built using HuggingFace Transformers
"""
from typing import Dict

from torch import nn
from transformers import DistilBertConfig, DistilBertModel

from .distilbert_text_classification_config import DistilBertTextClassificationConfig
from ..text_classification import TextClassificationModel
from ....registry import register_model


@register_model(model_name="distilbert_text_classification", config_class=DistilBertTextClassificationConfig)
class DistilBertTextClassification(TextClassificationModel):
    """
    A standard 🤗Transformers DistilBert model for text classification

    Args:
        config: The whole model config including arguments needed for the inner 🤗Transformers model.
    """
    tokenizer_name = "wordpiece_tokenizer"

    def __init__(self, config: DistilBertTextClassificationConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.distilbert = DistilBertModel(self._build_inner_config())
        self.pre_classifier = nn.Linear(self.config.dim, self.config.dim)
        self.classifier = nn.Linear(self.config.dim, self.config.num_labels)
        self.dropout = nn.Dropout(self.config.seq_classif_dropout)

    def _build_inner_config(self):
        if self.config.num_labels is None and self.config.id2label is None:
            raise ValueError("Both `num_labels` and `id2label` are None. Please provide at least one of them!")
        if self.config.id2label and self.config.num_labels is None:
            self.config.num_labels = len(self.config.id2label)
        bert_config = DistilBertConfig(**self.config)
        return bert_config

    def forward(self, inputs, **kwargs) -> Dict:
        input_ids = inputs.get("token_ids")
        attention_mask = inputs.get("attention_mask", None)
        head_mask = inputs.get("head_mask", None)
        inputs_embeds = inputs.get("inputs_embeds", None)
        output_attentions = inputs.get("output_attentions", None)
        output_hidden_states = inputs.get("output_hidden_states", None)

        lm_outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_state = lm_outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = {
            "logits": logits,
            "hidden_states": lm_outputs.hidden_states,
            "attentions": lm_outputs.attentions,
        }
        return outputs
