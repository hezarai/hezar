"""
A BERT model for text classification built using HuggingFace Transformers
"""
from typing import Dict

from transformers import BertConfig, BertForSequenceClassification

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
        self.model = self._build()

    def _build(self):
        if self.config.num_labels is None and self.config.id2label is None:
            raise ValueError("Both `num_labels` and `id2label` are None. Please provide at least one of them!")
        if self.config.id2label and self.config.num_labels is None:
            self.config.num_labels = len(self.config.id2label)
        config = BertConfig(**self.config)
        model = BertForSequenceClassification(config)
        return model

    def forward(self, inputs, **kwargs) -> Dict:
        input_ids = inputs.get("token_ids")
        attention_mask = inputs.get("attention_mask", None)
        labels = inputs.get("labels", None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
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
