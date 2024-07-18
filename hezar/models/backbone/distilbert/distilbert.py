"""
A DistilBERT base language model (HuggingFace Transformers) wrapped by a Hezar Model class
"""
from __future__ import annotations

from typing import List

from ....constants import Backends
from ....models import Model
from ....registry import register_model
from ....utils import is_backend_available
from .distilbert_config import DistilBERTConfig


if is_backend_available(Backends.TRANSFORMERS):
    from transformers import DistilBertConfig, DistilBertModel

_required_backends = [
    Backends.TRANSFORMERS,
    Backends.TOKENIZERS,
]


@register_model("distilbert", config_class=DistilBERTConfig)
class DistilBERT(Model):
    required_backends = _required_backends

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.distilbert = DistilBertModel(DistilBertConfig(**self.config))

    def forward(
        self,
        token_ids,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ):
        outputs = self.distilbert(
            input_ids=token_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        return outputs

    def preprocess(self, inputs: str | List[str], **kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]
        if self.preprocessor.text_normalizer is not None:
            inputs = self.preprocessor.text_normalizer(inputs)
        tokenizer = self.preprocessor.tokenizer
        inputs = tokenizer(inputs, return_tensors="torch", device=self.device)
        return inputs

    def post_process(self, model_outputs, **kwargs):
        hidden_states = model_outputs.get("hidden_states", None)
        attentions = model_outputs.get("attentions", None)
        outputs = {
            "last_hidden_state": model_outputs.get("last_hidden_state"),
            "hidden_states": hidden_states,
            "attentions": attentions,
        }
        return outputs
