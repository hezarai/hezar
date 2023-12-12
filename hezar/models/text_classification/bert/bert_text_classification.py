"""
A BERT model for text classification built using HuggingFace Transformers
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from ....constants import Backends
from ....registry import register_model
from ....utils import is_backend_available
from ...model import Model
from ...model_outputs import TextClassificationOutput
from .bert_text_classification_config import BertTextClassificationConfig


if is_backend_available(Backends.TRANSFORMERS):
    from transformers import BertConfig, BertModel

_required_backends = [
    Backends.TRANSFORMERS,
    Backends.TOKENIZERS,
]


@register_model(model_name="bert_text_classification", config_class=BertTextClassificationConfig)
class BertTextClassification(Model):
    """
    A standard 🤗Transformers Bert model for text classification

    Args:
        config: The whole model config including arguments needed for the inner 🤗Transformers model.
    """

    required_backends = _required_backends
    tokenizer_name = "wordpiece_tokenizer"
    skip_keys_on_load = [
        "model.embeddings.position_ids",  # For older versions
        "bert.embeddings.position_ids",
        "model.bert.embeddings.position_ids",
    ]

    def __init__(self, config: BertTextClassificationConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.bert = BertModel(self._build_inner_config())
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def _build_inner_config(self):
        if self.config.num_labels is None and self.config.id2label is None:
            raise ValueError("Both `num_labels` and `id2label` are None. Please provide at least one of them!")
        if self.config.id2label and self.config.num_labels is None:
            self.config.num_labels = len(self.config.id2label)
        bert_config = BertConfig(**self.config)
        return bert_config

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits.view(-1, self.config.num_labels), labels.view(-1))
        return loss

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
    ) -> Dict:
        lm_outputs = self.bert(
            token_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        pooled_output = lm_outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = {
            "logits": logits,
            "hidden_states": lm_outputs.hidden_states,
            "attentions": lm_outputs.attentions,
        }
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
