import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration

from ...model import Model
from ....registry import register_model
from .whisper_speech_recognition_config import WhisperSpeechRecognitionConfig


@register_model("whisper_speech_recognition", config_class=WhisperSpeechRecognitionConfig)
class WhisperSpeechRecognition(Model):
    def __init__(self, config: WhisperSpeechRecognitionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.whisper = WhisperForConditionalGeneration(WhisperConfig(**self.config))

    def forward(self, inputs, **kwargs):
        input_features = inputs.get("input_features")
        attention_mask = inputs.get("attention_mask", None)
        decoder_input_ids = inputs.get("decoder_input_ids", None)
        decoder_attention_mask = inputs.get("decoder_attention_mask", None)
        head_mask = inputs.get("head_mask", None)
        decoder_head_mask = inputs.get("decoder_head_mask", None)
        cross_attn_head_mask = inputs.get("cross_attn_head_mask", None)
        encoder_outputs = inputs.get("encoder_outputs", None)
        past_key_values = inputs.get("past_key_values", None)
        decoder_inputs_embeds = inputs.get("decoder_inputs_embeds", None)
        labels = inputs.get("labels", None)
        use_cache = inputs.get("use_cache", None)
        output_attentions = inputs.get("output_attentions", None)
        output_hidden_states = inputs.get("output_hidden_states", None)

        outputs = self.whisper(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return outputs

    def generate(
        self,
        inputs: torch.Tensor = None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        return_timestamps=None,
        task=None,
        language=None,
        is_multilingual=None,
        prompt_ids: torch.Tensor = None,
        **kwargs,
    ):
        generation_outputs = self.whisper.generate(
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            return_timestamps,
            task,
            language,
            is_multilingual,
            prompt_ids,
            **kwargs,
        )
        return generation_outputs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        **kwargs,
    ):
        return self.whisper.prepare_inputs_for_generation(
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            **kwargs,
        )

    def get_encoder(self):
        return self.whisper.get_encoder()

    def get_decoder(self):
        return self.whisper.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> torch.nn.Embedding:
        new_embeddings = self.whisper.resize_token_embeddings(new_num_tokens)
        return new_embeddings

    def get_output_embeddings(self):
        return self.whisper.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.whisper.set_output_embeddings(new_embeddings)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.whisper.get_input_embeddings()

    def freeze_encoder(self):
        self.whisper.freeze_encoder()

    def preprocess(self, inputs, **kwargs):
        # TODO
        return inputs

    def post_process(self, inputs, **kwargs):
        # TODO
        return inputs
