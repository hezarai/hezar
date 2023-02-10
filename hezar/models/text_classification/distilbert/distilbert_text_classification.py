"""
A DistilBERT model for text classification built using HuggingFace Transformers
"""
from typing import Dict

import torch
from transformers import DistilBertForSequenceClassification, DistilBertConfig

from hezar.data import Text
from hezar.models import register_model, Model
from .distilbert_text_classification_config import DistilBertTextClassificationConfig


@register_model(model_name='distilbert_text_classification', model_config=DistilBertTextClassificationConfig)
class DistilBertTextClassification(Model):
    """
    A standard 🤗Transformers DistilBert model for text classification

    Args:
        config: The whole model config including arguments needed for the inner 🤗Transformers model.
    """

    def __init__(self, config: DistilBertTextClassificationConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = self._build()
        self.tokenizer = None

    def _build(self):
        config = DistilBertConfig(**self.config.dict())
        model = DistilBertForSequenceClassification(config)
        if self.config.id2label is None:
            self.config.id2label = model.config.id2label
        return model

    def forward(self, inputs, **kwargs) -> Dict:
        outputs = self.model(**inputs, **kwargs)
        return outputs

    def preprocess(self, inputs: str, **kwargs):
        invalid_chars = self.config.get('invalid_chars', [])
        inputs = Text(inputs, tokenizer=self.tokenizer)
        inputs = inputs.normalize().filter_out(invalid_chars).tokenize(return_tensors='pt')
        return inputs

    def postprocess(self, inputs, **kwargs) -> Dict:
        logits = inputs['logits']
        predictions = logits.argmax(1)
        predictions_probs = logits.max(1)
        outputs = {'labels': [], 'probs': []}
        for prediction, prob in zip(predictions, predictions_probs):
            label = self.config.id2label[prediction.item()]
            outputs['labels'].append(label)
            outputs['probs'].append(prob.item())
        return outputs
