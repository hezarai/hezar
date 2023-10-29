"""
A DistilBERT Language Model (HuggingFace Transformers) wrapped by a Hezar Model class
"""
from typing import List, Union

import torch

from ....constants import Backends
from ....models import Model
from ....registry import register_model
from ....utils import is_backend_available
from ...model_outputs import LanguageModelingOutput
from .distilbert_lm_config import DistilBertLMConfig


if is_backend_available(Backends.TRANSFORMERS):
    from transformers import DistilBertConfig, DistilBertForMaskedLM

_required_backends = [
    Backends.TRANSFORMERS,
    Backends.TOKENIZERS,
]


@register_model("distilbert_lm", config_class=DistilBertLMConfig)
class DistilBertLM(Model):
    required_backends = _required_backends
    tokenizer_name = "wordpiece_tokenizer"
    loss_fn_name = "cross_entropy"

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.distilbert_mlm = DistilBertForMaskedLM(DistilBertConfig(**self.config))

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
        outputs = self.distilbert_mlm(
            input_ids=token_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
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

    def post_process(self, model_outputs, **kwargs):
        output_logits = model_outputs.get("logits", None)
        token_ids = model_outputs.get("token_ids", None)

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
