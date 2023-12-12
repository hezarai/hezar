"""
RoBERTa base language model (HuggingFace Transformers) wrapped by a Hezar Model class
"""
from __future__ import annotations

from typing import List

from ....constants import Backends
from ....models import Model
from ....registry import register_model
from ....utils import is_backend_available
from .roberta_config import RoBERTaConfig


if is_backend_available(Backends.TRANSFORMERS):
    from transformers import RobertaConfig, RobertaModel

_required_backends = [
    Backends.TRANSFORMERS,
    Backends.TOKENIZERS,
]


@register_model("roberta", config_class=RoBERTaConfig)
class RoBERTa(Model):
    required_backends = _required_backends
    tokenizer_name = "bpe_tokenizer"
    skip_keys_on_load = ["model.embeddings.position_ids", "roberta.embeddings.position_ids"]  # For older versions

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.roberta = RobertaModel(RobertaConfig(**self.config))

    def forward(
        self,
        token_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ):
        outputs = self.roberta(
            input_ids=token_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        return outputs

    def preprocess(self, inputs: str | List[str], **kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]
        if "text_normalizer" in self.preprocessor:
            normalizer = self.preprocessor["text_normalizer"]
            inputs = normalizer(inputs)
        tokenizer = self.preprocessor[self.tokenizer_name]
        inputs = tokenizer(inputs, return_tensors="pt", device=self.device)
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
