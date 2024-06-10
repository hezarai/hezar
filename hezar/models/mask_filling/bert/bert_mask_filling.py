from __future__ import annotations

from typing import List

import torch

from ....constants import Backends
from ....models import Model
from ....registry import register_model
from ....utils import is_backend_available
from ...model_outputs import MaskFillingOutput
from .bert_mask_filling_config import BertMaskFillingConfig


if is_backend_available(Backends.TRANSFORMERS):
    from transformers import BertConfig, BertForMaskedLM

_required_backends = [
    Backends.TRANSFORMERS,
    Backends.TOKENIZERS,
]


@register_model("bert_mask_filling", config_class=BertMaskFillingConfig)
class BertMaskFilling(Model):
    required_backends = _required_backends
    tokenizer_name = "wordpiece_tokenizer"
    skip_keys_on_load = ["model.embeddings.position_ids", "bert.embeddings.position_ids"]  # For older versions
    loss_func_name = "cross_entropy"

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.bert_mask_filling = BertForMaskedLM(BertConfig(**self.config))

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
        outputs = self.bert_mask_filling(
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
            return_dict=True,
        )
        outputs["token_ids"] = token_ids  # needed for post-process

        return outputs

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.loss_func(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return loss

    def preprocess(self, inputs: str | List[str], **kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]

        tokenizer = self.preprocessor[self.tokenizer_name]

        for text in inputs:
            if tokenizer.mask_token not in text:
                raise ValueError(f"The input must have a `{tokenizer.mask_token}` token!")

        inputs = tokenizer(inputs, return_tensors="pt", device=self.device)
        return inputs

    def post_process(self, model_outputs: dict, top_k=1):
        output_logits = model_outputs.get("logits")
        token_ids = model_outputs.get("token_ids")

        tokenizer = self.preprocessor[self.tokenizer_name]
        mask_token_id = tokenizer.mask_token_id

        unfilled_token_ids = token_ids.cpu().numpy().copy()
        outputs = []
        for batch_i, logits in enumerate(output_logits):
            masked_index = torch.nonzero(token_ids[batch_i] == mask_token_id, as_tuple=False).flatten()  # noqa
            if len(masked_index) > 1:
                raise ValueError(
                    f"Can't handle multiple `{tokenizer.mask_token}` tokens in the input for {self.__class__.__name__}!"
                )
            logits = logits[masked_index]
            probs = logits.softmax(dim=-1)
            probs, top_fill_token_ids = probs.topk(top_k)
            if top_k != 1:
                probs = probs.squeeze()
                top_fill_token_ids = top_fill_token_ids.squeeze()

            row = []
            for i, (prob, token_id) in enumerate(zip(probs, top_fill_token_ids)):
                candidate = unfilled_token_ids[batch_i].copy()
                candidate[masked_index.item()] = token_id
                row.append(
                    MaskFillingOutput(
                        token=tokenizer.decode([token_id.item()])[0].strip(),
                        sequence=tokenizer.decode(candidate.tolist())[0],
                        token_id=token_id.item(),
                        score=prob.item(),
                    )
                )
            outputs.append(row)
        return outputs
