"""
A RoBERTa Language Model (HuggingFace Transformers) wrapped by a Hezar Model class
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn, tanh

from ....constants import Backends
from ....registry import register_model
from ....utils import is_backend_available
from ...model import Model
from ...model_outputs import SequenceLabelingOutput
from .roberta_sequence_labeling_config import RobertaSequenceLabelingConfig


if is_backend_available(Backends.TRANSFORMERS):
    from transformers import RobertaConfig, RobertaModel

_required_backends = [
    Backends.TRANSFORMERS,
    Backends.TOKENIZERS,
]


@register_model("roberta_sequence_labeling", config_class=RobertaSequenceLabelingConfig)
class RobertaSequenceLabeling(Model):
    """
    A standard 🤗Transformers RoBERTa model for sequence labeling

    Args:
        config: The whole model config including arguments needed for the inner 🤗Transformers model.
    """

    required_backends = _required_backends
    tokenizer_name = "bpe_tokenizer"
    skip_keys_on_load = ["roberta.embeddings.position_ids", "model.embeddings.position_ids"]

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.roberta = RobertaModel(self._build_inner_config(), add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(self.config)

    def _build_inner_config(self):
        if self.config.num_labels is None and self.config.id2label is None:
            raise ValueError("Both `num_labels` and `id2label` are None. Please provide at least one of them!")
        if self.config.id2label and self.config.num_labels is None:
            self.config.num_labels = len(self.config.id2label)
        config = RobertaConfig(**self.config)
        return config

    def forward(
        self,
        token_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        lm_outputs = self.roberta(
            token_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = lm_outputs[0]
        logits = self.classifier(sequence_output)

        outputs = {
            "logits": logits,
            "hidden_states": lm_outputs.hidden_states,
            "attentions": lm_outputs.attentions,
            "tokens": kwargs.get("tokens", None),
            "offsets": kwargs.get("offsets_mapping", None)
        }
        return outputs

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits.view(-1, self.config.num_labels), labels.view(-1))
        return loss

    def preprocess(self, inputs: str | List[str], **kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]
        if "text_normalizer" in self.preprocessor:
            normalizer = self.preprocessor["text_normalizer"]
            inputs = normalizer(inputs)
        tokenizer = self.preprocessor[self.tokenizer_name]
        inputs = tokenizer(
            inputs,
            return_word_ids=True,
            return_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            device=self.device,
        )
        return inputs

    def post_process(
        self,
        model_outputs: Dict[str, torch.Tensor],
        return_offsets: bool = False,
        return_scores: bool = False,
    ):
        logits = model_outputs["logits"].softmax(2)
        tokens = model_outputs["tokens"]
        offsets = model_outputs["offsets"]
        probs, predictions = logits.max(2)
        predictions = [[self.config.id2label[p.item()] for p in prediction] for prediction in predictions]
        outputs = []
        for tokens_list, prediction, probs_, offsets_mapping in zip(tokens, predictions, probs, offsets):
            results = []
            for token, label, prob, offset in zip(tokens_list, prediction, probs_, offsets_mapping):
                if token not in self.config.prediction_skip_tokens:
                    token_results = {"token": token, "label": label}
                    if return_scores:
                        token_results["score"] = prob.item()
                    if return_offsets:
                        start, end = offset
                        token_results["start"] = start
                        token_results["end"] = end
                    results.append(SequenceLabelingOutput(**token_results))
            outputs.append(results)
        return outputs


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs, **kwargs):
        x = inputs  # Apply to all tokens
        x = self.dropout(x)
        x = self.dense(x)
        x = tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
