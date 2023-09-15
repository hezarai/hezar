"""
A DistilBERT model for text classification built using HuggingFace Transformers
"""
from typing import Dict, List, Union

from torch import nn

from ....constants import Backends
from ....registry import register_model
from ...model import Model
from ....utils import is_backend_available
from .distilbert_text_classification_config import DistilBertTextClassificationConfig

if is_backend_available(Backends.TRANSFORMERS):
    from transformers import DistilBertConfig, DistilBertModel

_required_backends = [
    Backends.TRANSFORMERS,
    Backends.TOKENIZERS,
]


@register_model(model_name="distilbert_text_classification", config_class=DistilBertTextClassificationConfig)
class DistilBertTextClassification(Model):
    """
    A standard 🤗Transformers DistilBert model for text classification

    Args:
        config: The whole model config including arguments needed for the inner 🤗Transformers model.
    """
    required_backends = _required_backends
    tokenizer_name = "wordpiece_tokenizer"

    def __init__(self, config: DistilBertTextClassificationConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.distilbert = DistilBertModel(self._build_inner_config())
        self.pre_classifier = nn.Linear(self.config.dim, self.config.dim)
        self.classifier = nn.Linear(self.config.dim, self.config.num_labels)
        self.dropout = nn.Dropout(self.config.seq_classif_dropout)

    def _build_inner_config(self):
        if self.config.num_labels is None and self.config.id2label is None:
            raise ValueError("Both `num_labels` and `id2label` are None. Please provide at least one of them!")
        if self.config.id2label and self.config.num_labels is None:
            self.config.num_labels = len(self.config.id2label)
        bert_config = DistilBertConfig(**self.config)
        return bert_config

    def forward(self, inputs, **kwargs) -> Dict:
        input_ids = inputs.get("token_ids")
        attention_mask = inputs.get("attention_mask", None)
        head_mask = inputs.get("head_mask", None)
        inputs_embeds = inputs.get("inputs_embeds", None)
        output_attentions = inputs.get("output_attentions", None)
        output_hidden_states = inputs.get("output_hidden_states", None)

        lm_outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_state = lm_outputs[0]
        pooled_output = hidden_state[:, 0]  # classification output
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = {
            "logits": logits,
            "hidden_states": lm_outputs.hidden_states,
            "attentions": lm_outputs.attentions,
        }
        return outputs

    def preprocess(self, inputs: Union[str, List[str]], **kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]
        if "text_normalizer" in self.preprocessor:
            normalizer = self.preprocessor["text_normalizer"]
            inputs = normalizer(inputs)
        tokenizer = self.preprocessor[self.tokenizer_name]
        inputs = tokenizer(inputs, return_tensors="pt", device=self.device)
        return inputs

    def post_process(self, inputs, **kwargs) -> Dict:
        return_all_scores = kwargs.get("return_all_scores", False)
        logits = inputs["logits"]
        if return_all_scores:
            predictions = logits
            predictions_probs = logits.softmax(1)
            outputs = []
            for sample_index in range(predictions.shape[0]):
                sample_outputs = []
                for label_index, score in enumerate(predictions_probs[sample_index]):
                    label = self.config.id2label[label_index]
                    sample_outputs.append({"label": label, "score": score.item()})
                outputs.append(sample_outputs)
        else:
            predictions = logits.argmax(1)
            predictions_probs = logits.softmax(1).max(1)
            outputs = {"labels": [], "probs": []}
            for prediction, prob in zip(predictions, predictions_probs):
                label = self.config.id2label[prediction.item()]
                outputs["labels"].append(label)
                outputs["probs"].append(prob.item())
        return outputs
