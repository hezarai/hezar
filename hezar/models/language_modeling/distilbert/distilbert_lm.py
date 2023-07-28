"""
A DistilBERT Language Model (HuggingFace Transformers) wrapped by a Hezar Model class
"""
from typing import List, Union

from transformers import DistilBertConfig, DistilBertModel

from ....models import Model
from ....registry import register_model
from .distilbert_lm_config import DistilBertLMConfig


@register_model("distilbert_lm", config_class=DistilBertLMConfig)
class DistilBertLM(Model):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.distilbert = DistilBertModel(DistilBertConfig(**self.config))

    def forward(self, inputs, **kwargs):
        input_ids = inputs.get("token_ids")
        attention_mask = inputs.get("attention_mask", None)
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return outputs

    def preprocess(self, inputs: Union[str, List[str]], **kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]
        if "normalizer" in self.preprocessor:
            normalizer = self.preprocessor["normalizer"]
            inputs = normalizer(inputs)
        tokenizer = self.preprocessor["wordpiece_tokenizer"]
        inputs = tokenizer(inputs, return_tensors="pt", device=self.device)
        return inputs

    def post_process(self, inputs, **kwargs):
        hidden_states = inputs.get("hidden_states", None)
        attentions = inputs.get("attentions", None)
        outputs = {
            "last_hidden_state": inputs.get("last_hidden_state"),
            "hidden_states": hidden_states,
            "attentions": attentions,
        }
        return outputs
