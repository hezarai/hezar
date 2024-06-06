"""
A RoBERTa Language Model (HuggingFace Transformers) wrapped by a Hezar Model class
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn, tanh

from ....constants import Backends
from ....registry import register_model
from ....utils import is_backend_available
from ...model import Model
from ...model_outputs import TextClassificationOutput
from .roberta_text_classification_config import RobertaTextClassificationConfig


if is_backend_available(Backends.TRANSFORMERS):
    from transformers import RobertaConfig, RobertaModel

_required_backends = [
    Backends.TRANSFORMERS,
    Backends.TOKENIZERS,
]


@register_model("roberta_text_classification", config_class=RobertaTextClassificationConfig)
class RobertaTextClassification(Model):
    """
    A standard 🤗Transformers RoBERTa model for text classification

    Args:
        config: The whole model config including arguments needed for the inner 🤗Transformers model.
    """

    required_backends = _required_backends
    tokenizer_name = "bpe_tokenizer"
    skip_keys_on_load = ["model.embeddings.position_ids", "roberta.embeddings.position_ids"]  # For older versions

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.roberta = RobertaModel(self._build_inner_config(), add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(self.config)

    def _build_inner_config(self):
        if self.config.num_labels is None and self.config.id2label is None:
            raise ValueError("Both `num_labels` and `id2label` are None. Please provide at least one of them!")
        if self.config.id2label and self.config.num_labels is None:
            self.config.num_labels = len(self.config.id2label)
        bert_config = RobertaConfig(**self.config)
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
        inputs = tokenizer(inputs, return_tensors="pt", device=self.device)
        return inputs

    def post_process(self, model_outputs: dict, top_k=1):
        output_logits = model_outputs["logits"]
        outputs = []
        for logits in output_logits:
            probs = logits.softmax(dim=-1)
            scores, label_ids = probs.sort(descending=True)
            row = []
            for i, (score, label_id) in enumerate(zip(scores, label_ids)):
                if i == top_k:
                    break
                label_str = self.config.id2label[label_id.item()]
                score = score.item()
                row.append(TextClassificationOutput(label=label_str, score=score))
            outputs.append(row)
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
        x = inputs[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
