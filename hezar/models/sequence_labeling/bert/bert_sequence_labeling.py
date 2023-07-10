"""
A BERT model for sequence labeling built using HuggingFace Transformers
"""
from typing import Dict

from torch import nn
from transformers import BertConfig, BertModel

from ....models import Model
from ....registry import register_model
from .bert_sequence_labeling_config import BertSequenceLabelingConfig


@register_model("bert_sequence_labeling", BertSequenceLabelingConfig)
class BertSequenceLabeling(Model):
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

    def forward(self, inputs, **kwargs) -> Dict:
        input_ids = inputs.get("token_ids")
        labels = inputs.get("labels")

        lm_outputs = self.bert(input_ids=input_ids, **kwargs)
        sequence_output = lm_outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits.view(-1, self.config.num_labels), labels.view(-1))

        outputs = {
            "loss": loss,
            "logits": logits,
            "hidden_states": lm_outputs.hidden_states,
            "attentions": lm_outputs.attentions,
            **inputs
        }
        return outputs

    def post_process(self, inputs, **kwargs):
        # TODO sequence labeling outputs should consider rejoining split words into single words with proper tag
        logits = inputs["logits"]
        tokens = inputs["tokens"]
        word_ids = inputs["word_ids"]  # noqa
        predictions = logits.argmax(2).cpu()
        predictions = [[self.config.id2label[p.item()] for p in prediction] for prediction in predictions]
        outputs = []
        for tokens_list, prediction in zip(tokens, predictions):
            results = []
            for token, tag in zip(tokens_list, prediction):
                if token not in self.config.prediction_skip_tokens:
                    results.append({"token": token, "tag": tag})
            outputs.append(results)
        return outputs
