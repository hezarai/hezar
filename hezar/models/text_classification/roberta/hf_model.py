"""
A Roberta model for text classification using transformers.RobertaForSequenceClassification
"""
from typing import Dict, Union

import torch
import torch.nn as nn
import transformers

from hezar.models.base_model import BaseModel
from hezar.data import Sentence
from hezar.utils.hub_utils import load_state_dict_from_hub

from .config import RobertaTextClassificationConfig


class RobertaTextClassification(BaseModel):
    def __init__(self, config: RobertaTextClassificationConfig, **kwargs):
        super(RobertaTextClassification, self).__init__(config, **kwargs)
        self.vocab_size = self.config.vocab_size
        self.pretrained_path = self.config.pretrained_path
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(self.pretrained_path)
        self.__dict__.update(kwargs)

    def build_model(self):
        model_config = transformers.RobertaConfig(vocab_size=self.vocab_size)
        model = transformers.RobertaForSequenceClassification(model_config)
        return model

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        config = RobertaTextClassificationConfig.from_pretrained(path)
        model = cls(config, **kwargs)
        state_dict = load_state_dict_from_hub(path)
        model.model.load_state_dict(state_dict)
        return model

    def forward(self, inputs: torch.Tensor, **kwargs) -> Dict:
        outputs = self.model(input_ids=inputs, **kwargs)
        return outputs

    def preprocess(self, inputs: str):
        inputs = Sentence(inputs, tokenizer=self.tokenizer)
        inputs = inputs.normalize().filter_out(['#', '@']).tokenize(return_tensors='pt')
        return inputs

    def predict(self, inputs: str, **kwargs) -> Dict:
        inputs = self.preprocess(inputs)
        outputs = self.forward(**inputs, **kwargs)
        processed_outputs = self.postprocess(outputs)
        return processed_outputs

    def postprocess(self, inputs, **kwargs) -> Dict:
        # TODO
        return inputs
