from typing import Dict, List, Union

import torch

from ....constants import Backends
from ....registry import register_model
from ...model import GenerativeModel
from ....utils import is_backend_available
from .t5_text_generation_config import T5TextGenerationConfig

if is_backend_available(Backends.TRANSFORMERS):
    from transformers import T5Config, T5ForConditionalGeneration

_required_backends = [
    Backends.TRANSFORMERS,
    Backends.TOKENIZERS,
]


@register_model("t5_text_generation", config_class=T5TextGenerationConfig)
class T5TextGeneration(GenerativeModel):
    """
    T5 for text to text generation
    """
    required_backends = _required_backends
    tokenizer_name = "sentencepiece_unigram_tokenizer"

    def __init__(self, config: T5TextGenerationConfig, **kwargs):
        super().__init__(config=config, **kwargs)

        self.t5 = T5ForConditionalGeneration(T5Config(**self.config))

    def forward(self, inputs, **kwargs) -> Dict:
        input_ids = inputs.get("token_ids")
        attention_mask = inputs.get("attention_mask", None)
        decoder_input_ids = inputs.get("decoder_input_ids", None)
        decoder_attention_mask = inputs.get("decoder_attention_mask", None)
        head_mask = inputs.get("head_mask", None)
        decoder_head_mask = inputs.get("decoder_head_mask", None)
        cross_attn_head_mask = inputs.get("cross_attn_head_mask", None)
        encoder_outputs = inputs.get("encoder_outputs", None)
        past_key_values = inputs.get("past_key_values", None)
        inputs_embeds = inputs.get("inputs_embeds", None)
        decoder_inputs_embeds = inputs.get("decoder_inputs_embeds", None)
        labels = inputs.get("labels", None)
        use_cache = inputs.get("use_cache", None)
        output_attentions = inputs.get("output_attentions", None)
        output_hidden_states = inputs.get("output_hidden_states", None)

        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return outputs

    def generate(self, inputs, **kwargs):
        input_ids = inputs.get("token_ids")
        attention_mask = inputs.get("attention_mask", None)
        input_bs, input_length = input_ids.shape
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        generation_kwargs = {"min_length": self.config.min_length, "max_length": self.config.max_length}
        output_ids = self.t5.generate(**model_inputs, **generation_kwargs, **kwargs)
        output_bs = output_ids.shape[0]
        output_ids = output_ids.reshape(input_bs, output_bs // input_bs, *output_ids.shape[1:])
        outputs = {"output_ids": output_ids}
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
        records = []
        tokenizer = self.preprocessor["sentencepiece_unigram_tokenizer"]
        for output_ids in inputs["output_ids"][0]:
            if isinstance(output_ids, torch.Tensor):
                output_ids = output_ids.numpy().tolist()
            record = {
                "output_text": tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                )
            }
            records.append(record)
        return records
