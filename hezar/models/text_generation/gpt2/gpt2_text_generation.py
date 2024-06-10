from __future__ import annotations

from typing import List

import torch

from ....constants import Backends
from ....registry import register_model
from ....utils import is_backend_available
from ...model import Model
from ...model_outputs import TextGenerationOutput
from .gpt2_text_generation_config import GPT2TextGenerationConfig


if is_backend_available(Backends.TRANSFORMERS):
    from transformers import (
        GenerationConfig,
        GPT2Config,
        GPT2LMHeadModel,
    )

_required_backends = [Backends.TRANSFORMERS, Backends.TOKENIZERS]


@register_model("gpt2_text_generation", config_class=GPT2TextGenerationConfig)
class GPT2TextGeneration(Model):
    is_generative = True
    tokenizer_name = "bpe_tokenizer"
    required_backends = _required_backends
    loss_func_name = "cross_entropy"

    def __init__(self, config: GPT2TextGenerationConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.gpt2 = GPT2LMHeadModel(config=GPT2Config(**self.config))

    def forward(
        self,
        token_ids,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        outputs = self.gpt2(
            input_ids=token_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return dict(outputs)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(logits.device)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Compute loss
        loss = self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def generate(self, token_ids, **kwargs):
        self.config.generation.update(kwargs or {})
        generation_config = GenerationConfig(**self.config.generation)
        generated_ids = self.gpt2.generate(token_ids, generation_config=generation_config)
        return generated_ids

    def preprocess(self, texts: str | List[str], **kwargs):
        tokenizer = self.preprocessor[self.tokenizer_name]
        inputs = tokenizer(texts, return_tensors="pt", device=self.device)
        return inputs

    def post_process(self, generated_ids: torch.Tensor):
        tokenizer = self.preprocessor[self.tokenizer_name]
        decoded_outputs = tokenizer.decode(generated_ids.cpu().numpy().tolist())
        outputs = [TextGenerationOutput(text=text) for text in decoded_outputs]
        return outputs
