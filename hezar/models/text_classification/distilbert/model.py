"""
A DistilBERT model for text classification built using HuggingFace Transformers
"""
from typing import Dict

from omegaconf import OmegaConf, DictConfig
from transformers import DistilBertForSequenceClassification, DistilBertConfig, AutoTokenizer, BatchEncoding

from hezar.data import Text
from hezar.models import register_model, BaseModel
from .config import DistilBertTextClassificationConfig


@register_model(model_name='distilbert_text_classification', model_config=DistilBertTextClassificationConfig)
class DistilBertTextClassification(BaseModel):
    """
    A standard 🤗Transformers DistilBert model for text classification

    Args:
        config: The whole model config including arguments needed for the inner 🤗Transformers model.
    """

    def __init__(self, config: DistilBertTextClassificationConfig, **kwargs):
        super(DistilBertTextClassification, self).__init__(config, **kwargs)
        self.model = self.build_model()
        self.tokenizer = None

    def build_model(self):
        config = DistilBertConfig(**self.config.dict())
        model = DistilBertForSequenceClassification(config)
        return model

    def forward(self, inputs, **kwargs) -> Dict:
        outputs = self.model(**inputs, **kwargs)
        return outputs

    def preprocess(self, inputs: str):
        invalid_chars = self.config.get('invalid_chars', [])
        inputs = Text(inputs, tokenizer=self.tokenizer)
        inputs = inputs.normalize().filter_out(invalid_chars).tokenize(return_tensors='pt')
        return inputs

    def predict(self, inputs: str, **kwargs) -> Dict:
        inputs = self.preprocess(inputs)
        outputs = self.forward(inputs, **kwargs)
        processed_outputs = self.postprocess(outputs)
        return processed_outputs

    def postprocess(self, inputs, **kwargs) -> Dict:
        # TODO
        return inputs

