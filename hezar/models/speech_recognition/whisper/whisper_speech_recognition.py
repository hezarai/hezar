from __future__ import annotations

import copy
from typing import List

import librosa
import numpy as np
import torch

from ....constants import Backends
from ....registry import register_model
from ....utils import is_backend_available, load_audio_files, shift_tokens_right
from ...model import Model
from ...model_outputs import SpeechRecognitionOutput
from .whisper_speech_recognition_config import WhisperSpeechRecognitionConfig


if is_backend_available(Backends.TRANSFORMERS):
    from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration

_required_backends = [
    Backends.TRANSFORMERS,
    Backends.TOKENIZERS,
    Backends.LIBROSA,
]


@register_model("whisper_speech_recognition", config_class=WhisperSpeechRecognitionConfig)
class WhisperSpeechRecognition(Model):
    """
    Whisper model for automatic speech recognition
    """

    is_generative = True
    required_backends = _required_backends
    feature_extractor_name = "whisper_feature_extractor"
    tokenizer_name = "whisper_bpe_tokenizer"
    loss_func_name = "cross_entropy"

    def __init__(self, config: WhisperSpeechRecognitionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.whisper = WhisperForConditionalGeneration(WhisperConfig(**self.config))

    def forward(
        self,
        input_features,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ):
        if decoder_input_ids is None or decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
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
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return dict(outputs)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor = None):
        labels = copy.deepcopy(labels)
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask.ne(1), -100)
        loss = self.loss_func(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return loss

    def generate(
        self,
        input_features,
        forced_decoder_ids=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        return_timestamps=None,
        task=None,
        language=None,
        is_multilingual=None,
        prompt_ids=None,
        **kwargs,
    ):
        if generation_config is not None:
            self.config.generation_config.update(**generation_config)

        generation_config = GenerationConfig(**self.config.generation_config.dict())

        generation_outputs = self.whisper.generate(
            input_features=input_features,
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

    def preprocess(self, inputs: str | np.ndarray | List[np.ndarray] | List[str], **kwargs):
        if isinstance(inputs, str) or (isinstance(inputs, List) and isinstance(inputs[0], str)):
            inputs = load_audio_files(inputs)
        elif isinstance(inputs, List) and isinstance(inputs[0], np.ndarray):
            inputs = [librosa.to_mono(x.transpose()) for x in inputs if isinstance(x, np.ndarray) and len(x.shape) > 1]

        tokenizer = self.preprocessor[self.tokenizer_name]
        feature_extractor = self.preprocessor[self.feature_extractor_name]

        forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="persian", task="transcribe")
        inputs = feature_extractor(inputs, sampling_rate=self.config.sampling_rate, return_tensors="pt")
        inputs["forced_decoder_ids"] = forced_decoder_ids
        return inputs

    def post_process(
        self,
        model_outputs,
        skip_special_tokens=True,
        decode_with_timestamps=True,
        output_offsets=False,
        **kwargs,
    ):
        tokenizer = self.preprocessor[self.tokenizer_name]
        if isinstance(model_outputs, torch.Tensor):
            model_outputs = model_outputs.cpu().numpy().tolist()
        transcripts = tokenizer.decode(
            model_outputs,
            decode_with_timestamps=decode_with_timestamps,
            skip_special_tokens=skip_special_tokens,
            output_offsets=output_offsets,
        )
        outputs = [SpeechRecognitionOutput(text=transcript) for transcript in transcripts]
        return outputs
