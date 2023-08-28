"""
A BERT Language Model (HuggingFace Transformers) wrapped by a Hezar Model class
"""
from typing import List, Union

from transformers import BertConfig, BertModel

from ....models import Model
from ....registry import register_model
from .bert_lm_config import BertLMConfig


@register_model("bert_lm", config_class=BertLMConfig)
class BertLM(Model):
    tokenizer_name = "wordpiece_tokenizer"
    skip_keys_on_load = ["bert.embeddings.position_ids"]

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.bert = BertModel(BertConfig(**self.config))

    def forward(self, inputs, **kwargs):
        input_ids = inputs.get("token_ids")
        attention_mask = inputs.get("attention_mask", None)
        token_type_ids = inputs.get("token_type_ids", None)
        position_ids = inputs.get("position_ids", None)
        head_mask = inputs.get("head_mask", None)
        inputs_embeds = inputs.get("inputs_embeds", None)
        encoder_hidden_states = inputs.get("encoder_hidden_states", None)
        encoder_attention_mask = inputs.get("encoder_attention_mask", None)
        past_key_values = inputs.get("past_key_values", None)
        use_cache = inputs.get("use_cache", None)
        output_attentions = inputs.get("output_attentions", None)
        output_hidden_states = inputs.get("output_hidden_states", None)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
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

    def post_process(self, inputs, **kwargs):
        hidden_states = inputs.get("hidden_states", None)
        attentions = inputs.get("attentions", None)
        outputs = {
            "last_hidden_state": inputs.get("last_hidden_state"),
            "hidden_states": hidden_states,
            "attentions": attentions,
        }
        return outputs
