"""
A BERT model for text classification built using HuggingFace Transformers
"""
from typing import Dict, List, Union

from torch import nn
from transformers import BertConfig, BertModel

from ..text_classification import TextClassificationModel
from ....registry import register_model
from .bert_text_classification_config import BertTextClassificationConfig


@register_model(model_name="bert_text_classification", config_class=BertTextClassificationConfig)
class BertTextClassification(TextClassificationModel):
    """
    A standard 🤗Transformers Bert model for text classification

    Args:
        config: The whole model config including arguments needed for the inner 🤗Transformers model.
    """
    tokenizer_name = "wordpiece_tokenizer"

    def __init__(self, config: BertTextClassificationConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.bert = BertModel(self._build_inner_config())
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None
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

    def forward(self, inputs, **kwargs) -> Dict:
        input_ids = inputs.get("token_ids")
        attention_mask = inputs.get("attention_mask", None)
        token_types_ids = inputs.get("token_type_ids", None)
        position_ids = inputs.get("position_ids", None)
        head_mask = inputs.get("head_mask", None)
        inputs_embeds = inputs.get("inputs_embeds", None)
        encoder_hidden_states = inputs.get("encoder_hidden_states", None)
        encoder_attention_mask = inputs.get("encoder_attention_mask", None)
        past_key_values = inputs.get("past_key_values", None)
        use_cache = inputs.get("use_cache", None)
        output_attentions = inputs.get("output_attentions", None)
        output_hidden_states = inputs.get("output_hidden_states", None)

        lm_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_types_ids,
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
