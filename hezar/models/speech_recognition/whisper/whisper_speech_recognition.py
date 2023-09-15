from typing import List, Union

import numpy as np
import torch

from ....constants import Backends
from ....registry import register_model
from ....utils import is_backend_available
from ...model import GenerativeModel
from ...model_outputs import SpeechRecognitionOutput
from .whisper_speech_recognition_config import WhisperSpeechRecognitionConfig

if is_backend_available(Backends.TRANSFORMERS):
    from transformers import WhisperConfig, WhisperForConditionalGeneration

if is_backend_available(Backends.LIBROSA):
    import librosa

_required_backends = [
    Backends.TRANSFORMERS,
    Backends.TOKENIZERS,
    Backends.LIBROSA,
]


@register_model("whisper_speech_recognition", config_class=WhisperSpeechRecognitionConfig)
class WhisperSpeechRecognition(GenerativeModel):
    """
    Whisper model for automatic speech recognition
    """
    required_backends = _required_backends
    is_generation_model = True
    feature_extractor_name = "whisper_feature_extractor"
    tokenizer_name = "whisper_bpe_tokenizer"

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

    def generate(self, inputs, **kwargs):
        inputs_tensor = inputs.pop("input_features")
        forced_decoder_ids = inputs.pop("forced_decoder_ids", None)
        generation_config = inputs.pop("generation_config", None)
        logits_processor = inputs.pop("logits_processor", None)
        stopping_criteria = inputs.pop("stopping_criteria", None)
        prefix_allowed_tokens_fn = inputs.pop("prefix_allowed_tokens_fn", None)
        synced_gpus = inputs.pop("synced_gpus", None)
        return_timestamps = inputs.pop("return_timestamps", None)
        task = inputs.pop("task", None)
        language = inputs.pop("language", None)
        is_multilingual = inputs.pop("is_multilingual", None)
        prompt_ids = inputs.pop("prompt_ids", None)

        generation_outputs = self.whisper.generate(
            inputs=inputs_tensor,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            return_timestamps=return_timestamps,
            task=task,
            language=language,
            is_multilingual=is_multilingual,
            prompt_ids=prompt_ids,
            forced_decoder_ids=forced_decoder_ids,
            **inputs,
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

    def preprocess(self, inputs: Union[str, np.ndarray, List[np.ndarray], List[str]], **kwargs):
        if isinstance(inputs, np.ndarray) and len(inputs.shape) > 1:
            inputs = [inputs]
        if isinstance(inputs, str) or (isinstance(inputs, List), isinstance(inputs[0], str)):
            if isinstance(inputs, str):
                inputs = [inputs]
            inputs = [librosa.load(x, sr=self.config.sampling_rate)[0] for x in inputs]

        tokenizer = self.preprocessor[self.tokenizer_name]
        feature_extractor = self.preprocessor[self.feature_extractor_name]

        forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="persian", task="transcribe")
        inputs = feature_extractor(inputs, sampling_rate=self.config.sampling_rate, return_tensors="pt")
        inputs["forced_decoder_ids"] = forced_decoder_ids
        return inputs

    def post_process(self, inputs, **kwargs):
        tokenizer = self.preprocessor[self.tokenizer_name]
        transcripts = tokenizer.decode(inputs, decode_with_timestamps=True, skip_special_tokens=True)
        return SpeechRecognitionOutput(transcripts=transcripts).dict()
