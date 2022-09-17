"""
A DistilBERT model for text classification built using HuggingFace Transformers
"""

from typing import *

from omegaconf import OmegaConf, DictConfig
import transformers

from hezar.data import Text
from hezar.models import register_model
from hezar.models.base_model import BaseModel, ModelMode
from .config import DistilBertTextClassificationConfig


@register_model(model_name='distilbert_text_classification', model_config=DistilBertTextClassificationConfig)
class DistilBertTextClassification(BaseModel):
    def __init__(self, config: DistilBertTextClassificationConfig, mode: ModelMode, **kwargs):
        super(DistilBertTextClassification, self).__init__(config, mode, **kwargs)
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(**self.config.inner_model_config)

    def build_model(self, mode: ModelMode):
        if mode.value == 'training':
            model = transformers.DistilBertForSequenceClassification.from_pretrained(**self.config.inner_model_config)
            model_config = OmegaConf.structured(model.config.__dict__)
            self.config.inner_model_config = OmegaConf.merge(self.config.inner_model_config, model_config)
        elif mode.value == 'inference':
            inner_model_config = transformers.DistilBertConfig(**self.config.inner_model_config)
            model = transformers.DistilBertForSequenceClassification(inner_model_config)
        else:
            raise ValueError(f'Unknown mode for model: `{mode}`, expected: `Literal{ModelMode.list()}`')
        return model

    def push_to_hub(self, path, **kwargs):
        ...

    def forward(self, inputs: transformers.BatchEncoding, **kwargs) -> Dict:
        outputs = self.model(**inputs, **kwargs)
        return outputs

    def preprocess(self, inputs: str):
        inputs = Text(inputs, tokenizer=self.tokenizer)
        inputs = inputs.normalize().filter_out(['#', '@']).tokenize(return_tensors='pt')
        return inputs

    def predict(self, inputs: str, **kwargs) -> Dict:
        inputs = self.preprocess(inputs)
        outputs = self.forward(inputs, **kwargs)
        processed_outputs = self.postprocess(outputs)
        return processed_outputs

    def postprocess(self, inputs, **kwargs) -> Dict:
        # TODO
        return inputs

