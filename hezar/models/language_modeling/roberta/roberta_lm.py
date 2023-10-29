"""
A RoBERTa Language Model (HuggingFace Transformers) wrapped by a Hezar Model class
"""
from typing import List, Union

import torch

from ....constants import Backends
from ....models import Model
from ....registry import register_model
from ....utils import is_backend_available
from ...model_outputs import LanguageModelingOutput
from .roberta_lm_config import RobertaLMConfig


if is_backend_available(Backends.TRANSFORMERS):
    from transformers import RobertaConfig, RobertaForMaskedLM

_required_backends = [
    Backends.TRANSFORMERS,
    Backends.TOKENIZERS,
]


@register_model("roberta_lm", config_class=RobertaLMConfig)
class RobertaLM(Model):
    required_backends = _required_backends
    tokenizer_name = "bpe_tokenizer"
    skip_keys_on_load = ["model.embeddings.position_ids", "roberta.embeddings.position_ids"]  # For older versions
    loss_fn_name = "cross_entropy"

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.roberta_mlm = RobertaForMaskedLM(RobertaConfig(**self.config))

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
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ):
        outputs = self.roberta_mlm(
            input_ids=token_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        outputs["token_ids"] = token_ids

        return outputs

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return loss

    def preprocess(self, inputs: Union[str, List[str]], **kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]

        tokenizer = self.preprocessor[self.tokenizer_name]

        for text in inputs:
            if tokenizer.mask_token not in text:
                raise ValueError(f"The input must have a `{tokenizer.mask_token}` token!")

        inputs = tokenizer(inputs, return_tensors="pt", device=self.device)
        return inputs

    def post_process(self, model_outputs, top_k=5):
        output_logits = model_outputs.get("logits")
        token_ids = model_outputs.get("token_ids")

        tokenizer = self.preprocessor[self.tokenizer_name]

        filled_token_ids = token_ids.cpu().numpy().copy()
        fill_tokens = []
        for batch_i, logits in enumerate(output_logits):
            masked_index = torch.nonzero(
                token_ids[batch_i] == tokenizer.mask_token_id, as_tuple=False
            ).flatten()  # noqa
            if len(masked_index) > 1:
                raise ValueError(
                    f"Can't handle multiple `{tokenizer.mask_token}` tokens in the input for {self.__class__.__name__}!"
                )
            fill_token_id = logits[masked_index].argmax().item()
            filled_token_ids[batch_i, masked_index.item()] = fill_token_id
            fill_tokens.append(tokenizer.decode([fill_token_id])[0])

        outputs = LanguageModelingOutput(
            filled_texts=tokenizer.decode(filled_token_ids),
            filled_tokens=fill_tokens,
        )

        return outputs
