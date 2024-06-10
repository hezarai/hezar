"""
A BERT model for sequence labeling built using HuggingFace Transformers
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from ....constants import Backends
from ....registry import register_model
from ....utils import is_backend_available
from ...model import Model
from ...model_outputs import SequenceLabelingOutput
from .bert_sequence_labeling_config import BertSequenceLabelingConfig


if is_backend_available(Backends.TRANSFORMERS):
    from transformers import BertConfig, BertModel

_required_backends = [
    Backends.TRANSFORMERS,
    Backends.TOKENIZERS,
]


@register_model("bert_sequence_labeling", BertSequenceLabelingConfig)
class BertSequenceLabeling(Model):
    """
    BERT model for sequence labeling
    """

    required_backends = _required_backends
    tokenizer_name = "wordpiece_tokenizer"
    skip_keys_on_load = ["model.embeddings.position_ids", "bert.embeddings.position_ids"]  # For older versions

    def __init__(self, config: BertSequenceLabelingConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.bert = BertModel(self._build_inner_config())
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def _build_inner_config(self):
        if self.config.num_labels is None and self.config.id2label is None:
            raise ValueError("Both `num_labels` and `id2label` are None. Please provide at least one of them!")
        if self.config.id2label and self.config.num_labels is None:
            self.config.num_labels = len(self.config.id2label)
        bert_config = BertConfig(**self.config)
        return bert_config

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
        **kwargs,
    ) -> Dict:
        lm_outputs = self.bert(
            token_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = lm_outputs[0]

        sequence_output = self.dropout(sequence_output)
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
