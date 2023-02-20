"""
A DistilBERT Language Model (HuggingFace Transformers) wrapped by a Hezar Model class
"""
from transformers import DistilBertModel, DistilBertConfig

from hezar.models import register_model, Model
from hezar.data import Text
from .distilbert_lm_config import DistilBertLMConfig


@register_model("distilbert_lm", config_class=DistilBertLMConfig)
class DistilBertLM(Model):
    def __init__(self, config):
        super().__init__(config=config)
        self.model = self._build()

    def _build(self):
        config = DistilBertConfig(**self.config)
        model = DistilBertModel(config)
        return model

    def forward(self, inputs, **kwargs):
        outputs = self.model(**inputs, **kwargs)
        return outputs

    def post_process(self, inputs, **kwargs):
        hidden_states = inputs.get("hidden_states", None)
        attentions = inputs.get("attentions", None)
        outputs = {
            "last_hidden_state": inputs.get("last_hidden_state"),
            "hidden_states": hidden_states,
            "attentions": attentions,
        }
        return outputs
