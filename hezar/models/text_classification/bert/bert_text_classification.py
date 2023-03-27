"""
A BERT model for text classification built using HuggingFace Transformers
"""
from typing import Dict

from torch import nn
from transformers import BertConfig, BertModel

from ....models import Model
from ....registry import register_model
from .bert_text_classification_config import BertTextClassificationConfig


@register_model(model_name="bert_text_classification", config_class=BertTextClassificationConfig)
class BertTextClassification(Model):
    """
    A standard 🤗Transformers Bert model for text classification

    Args:
        config: The whole model config including arguments needed for the inner 🤗Transformers model.
    """

    def __init__(self, config: BertTextClassificationConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.bert = BertModel(self._build_inner_config())
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def _build_inner_config(self):
        if self.config.num_labels is None and self.config.id2label is None:
            raise ValueError("Both `num_labels` and `id2label` are None. Please provide at least one of them!")
        if self.config.id2label and self.config.num_labels is None:
            self.config.num_labels = len(self.config.id2label)
        bert_config = BertConfig(**self.config)
        return bert_config

    def forward(self, inputs, **kwargs) -> Dict:
        input_ids = inputs.get("token_ids")
        labels = inputs.get("labels")

        lm_outputs = self.bert(input_ids=input_ids, **kwargs)
        pooled_output = lm_outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        outputs = {
            "loss": loss,
            "logits": logits,
            "hidden_states": lm_outputs.hidden_states,
            "attentions": lm_outputs.attentions,
        }
        return outputs

    def post_process(self, inputs, **kwargs) -> Dict:
        logits = inputs["logits"]
        predictions = logits.argmax(1)
        predictions_probs = logits.softmax(1).max(1)
        outputs = {"labels": [], "probs": []}
        for prediction, prob in zip(predictions, predictions_probs):
            label = self.config.id2label[prediction.item()]
            outputs["labels"].append(label)
            outputs["probs"].append(prob.item())
        return outputs
