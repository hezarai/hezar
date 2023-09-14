from dataclasses import dataclass
from typing import List

import numpy as np

from ...constants import Backends
from ...registry import register_preprocessor
from ...utils import Logger, is_backend_available
from .bpe import BPEConfig, BPETokenizer


if is_backend_available(Backends.TOKENIZERS):
    from tokenizers import processors

_required_backends = [
    Backends.TOKENIZERS,
]


logger = Logger(__name__)

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}

TASK_IDS = ["translate", "transcribe"]


@dataclass
class WhisperBPEConfig(BPEConfig):
    name = "whisper_bpe_tokenizer"
    unk_token: str = "<|endoftext|>"
    unk_token_id: int = 50257
    bos_token: str = "<|startoftranscript|>"
    bos_token_id: int = 50257
    eos_token: str = "<|endoftext|>"
    eos_token_id: int = 50257
    padding_direction: str = "right"
    add_prefix_space: bool = False
    add_bos_token: bool = False
    model_max_length: int = 1024
    language: str = None
    task: str = None
    predict_timestamps: str = False


@register_preprocessor("whisper_bpe_tokenizer", config_class=WhisperBPEConfig)
class WhisperBPETokenizer(BPETokenizer):
    required_backends = _required_backends

    def __init__(self, config, tokenizer_file=None, **kwargs):
        super().__init__(config, tokenizer_file=tokenizer_file, **kwargs)
        self.language = self.config.language
        self.task = self.config.task
        self.predict_timestamps = self.config.predict_timestamps

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        output_offsets: bool = False,
        time_precision=0.02,
        decode_with_timestamps: bool = False,
        **kwargs,
    ):
        """
        Override decode method to enable timestamps and offsets.
        """
        text = super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            **kwargs
        )
        if decode_with_timestamps:
            text = [
                self._decode_with_timestamps(
                    token_id,
                    time_precision=time_precision,
                    skip_special_tokens=skip_special_tokens,
                )
                for token_id in token_ids
            ]
        # retrieve offsets
        if output_offsets:
            offsets = self._compute_offsets(token_ids, time_precision=time_precision)
            return {"text": text, "offsets": offsets}
        return text

    def _decode_with_timestamps(self, token_ids, skip_special_tokens=False, time_precision=0.02) -> str:
        """
        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`. This method decodes
        given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        timestamp_begin = self.special_ids[-1] + 1
        outputs = [[]]
        for token in token_ids:
            if token >= timestamp_begin:
                timestamp = f"<|{(token - timestamp_begin) * time_precision:.2f}|>"
                outputs.append(timestamp)
                outputs.append([])
            else:
                outputs[-1].append(token)
        outputs = self.decode(outputs, skip_special_tokens=skip_special_tokens)
        return "".join(outputs)

    def _compute_offsets(self, token_ids, time_precision=0.02):
        """
        Compute offsets for a given tokenized input

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            time_precision (`float`, `optional`, defaults to 0.02):
                The time ratio to convert from token to time.
        """
        offsets = []
        token_ids = np.array(token_ids)
        if token_ids.shape[0] > 1 and len(token_ids.shape) > 1:
            raise ValueError("Can only process a single input at a time")
        timestamp_begin = self.special_ids[-1] + 1
        timestamp_tokens = token_ids >= timestamp_begin

        consecutive = np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1
        if consecutive.shape[0] == 0 and timestamp_tokens.sum() <= 1:
            # either there are no timestamps or there are no consecutive ones
            return []
        elif np.where(timestamp_tokens)[0][-1] + 1 not in consecutive:
            # we add the final timestamp if it is not already in the list
            consecutive = np.append(consecutive, np.where(timestamp_tokens)[0][-1] + 1)

        last_slice = np.where(timestamp_tokens)[0][0]
        for current_slice in consecutive:
            sliced_tokens = token_ids[last_slice:current_slice]
            if len(sliced_tokens) > 1:
                start_timestamp_position = sliced_tokens[0].item() - timestamp_begin
                end_timestamp_position = sliced_tokens[-1].item() - timestamp_begin
                offsets.append(
                    {
                        "text": self.decode(sliced_tokens),
                        "timestamp": (
                            start_timestamp_position * time_precision,
                            end_timestamp_position * time_precision,
                        ),
                    }
                )
            last_slice = current_slice

        return offsets

    def get_prompt_ids(self, text: str, return_tensors="np"):
        """Converts prompt text to IDs that can be passed to [`~WhisperForConditionalGeneration.generate`]."""
        batch_encoding = self([("<|startofprev|>", " " + text.strip())], add_special_tokens=False)

        # Check for special tokens
        prompt_text_ids = batch_encoding["input_ids"][1:]
        special_token_id = next((x for x in prompt_text_ids if x >= self.all_special_ids[0]), None)
        if special_token_id is not None:
            token = self.convert_ids_to_tokens(special_token_id)
            raise ValueError(f"Encountered text in the prompt corresponding to disallowed special token: {token}.")

        batch_encoding.convert_to_tensors(tensor_type=return_tensors)
        return batch_encoding["input_ids"]

    @staticmethod
    def _strip_prompt(token_ids, prompt_token_id: int, decoder_start_token_id: int):
        has_prompt = isinstance(token_ids, list) and token_ids and token_ids[0] == prompt_token_id
        if has_prompt:
            if decoder_start_token_id in token_ids:
                return token_ids[token_ids.index(decoder_start_token_id):]
            else:
                return []

        return token_ids

    def set_prefix_tokens(self, language: str = None, task: str = None, predict_timestamps: bool = None):
        self.language = language if language is not None else self.language
        self.task = task if task is not None else self.task
        self.predict_timestamps = predict_timestamps if predict_timestamps is not None else self.predict_timestamps

        prefix_token_ids = self.prefix_tokens
        prefixes = self.convert_ids_to_tokens(prefix_token_ids)
        eos = self.config.eos_token
        eos_token_id = self.config.eos_token_id
        prefix_template = " ".join([f"{token}:0" for token in prefixes])
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{prefix_template} $A:0 {eos}:0",
            pair=f"{prefix_template} $A:0 $B:1 {eos}:1",
            special_tokens=[
                (eos, eos_token_id),
                *zip(prefixes, prefix_token_ids),
            ],
        )

    @property
    def prefix_tokens(self) -> List[int]:
        all_special_ids = self.special_ids
        bos_token_id = all_special_ids[-106]
        translate_token_id = all_special_ids[-6]
        transcribe_token_id = all_special_ids[-5]
        notimestamps_token_id = all_special_ids[-1]
        langs = tuple(LANGUAGES.keys())

        if self.language is not None:
            self.language = self.language.lower()
            if self.language in TO_LANGUAGE_CODE:
                language_id = TO_LANGUAGE_CODE[self.language]
            elif self.language in TO_LANGUAGE_CODE.values():
                language_id = self.language
            else:
                is_language_code = len(self.language) == 2
                raise ValueError(
                    f"Unsupported language: {self.language}. Language should be one of:"
                    f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                )

        if self.task is not None:
            if self.task not in TASK_IDS:
                raise ValueError(f"Unsupported task: {self.task}. Task should be in: {TASK_IDS}")

        bos_sequence = [bos_token_id]
        if self.language is not None:
            bos_sequence.append(bos_token_id + 1 + langs.index(language_id))
        if self.task is not None:
            bos_sequence.append(transcribe_token_id if self.task == "transcribe" else translate_token_id)
        if not self.predict_timestamps:
            bos_sequence.append(notimestamps_token_id)
        return bos_sequence

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        self.set_prefix_tokens(task=task, language=language, predict_timestamps=not no_timestamps)
        # prefix tokens are of the form: <|startoftranscript|> <|lang_id|> <|task|> <|notimestamps|>
        # we don't want to force the bos token at position 1, as this is the starting token
        # when we generate, so we slice the prefix tokens to: <|lang_id|> <|task|> <|notimestamps|>
        # to get the forced tokens
        forced_tokens = self.prefix_tokens[1:]
        forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_tokens)]
        return forced_decoder_ids

    def _decode_asr(self, model_outputs, *, return_timestamps, return_language, time_precision):
        """
        Internal method meant to only be used by asr pipeline. Handles all the little quirks specific to whisper
        to handle the various options not allowed in other seq2seq models
        """

        # =========== Overview ============
        # - iterate over all outputs
        # - all tokens within output
        # - Each token can be
        #   - language token
        #   - special token
        #   - timestamp token
        #   - text token
        # - We accumulate the text tokens.
        # - We split on end timestamps
        # - Lots of complexity comes from stride and timestamps

        last_language = None

        def new_chunk():
            return {"language": last_language, "timestamp": [None, None], "text": ""}

        # Welcome to the state machine !
        chunks = []
        chunk = new_chunk()
        time_offset = 0.0
        timestamp_begin = self.convert_tokens_to_ids("<|notimestamps|>") + 1
        previous_tokens = []
        skip = False
        right_stride_start = None

        all_special_ids = set(self.special_ids)
        # - iterate over all outputs
        for chunk_id, output in enumerate(model_outputs):
            # We can drop everything to Python list, it's going to make
            # our lives easier
            token_ids = output["tokens"][0].tolist()

            # Those keep track of timestamps within strides
            # Which need to be skipped and resolve all tokens in a single
            # chunk.
            last_timestamp = None
            first_timestamp = timestamp_begin

            if "stride" in output:
                chunk_len, stride_left, stride_right = output["stride"]
                # Offset the timings to account for the other `model_outputs`.
                time_offset -= stride_left
                right_stride_start = chunk_len - stride_right

                # Keeping track of timestamps within strides
                # We're going to NOT split on those, and delay until we're
                # out of BOTH stride. Otherwise, lots of issues occur and
                # corner cases
                if stride_left:
                    first_timestamp = stride_left / time_precision + timestamp_begin
                if stride_right:
                    for token in reversed(token_ids):
                        if token >= timestamp_begin:
                            # There can be several token in the right stride
                            # But the last one is ALWAYS going to be skipped
                            if (
                                last_timestamp is not None
                                and (token - timestamp_begin) * time_precision < right_stride_start
                            ):
                                break
                            last_timestamp = token

            current_tokens = []

            # - all tokens within output
            for i, token in enumerate(token_ids):
                # 4 possible states for each token
                # - 1/ Language code
                # - 2/ all other special tokens (which we ignore)
                # - 3/ Timestamp
                # - 4/ Regular text
                if token in all_special_ids:
                    # Either language code or other
                    text = self.decode([token])
                    # Removing outer shell <|XX|>
                    text = text[2:-2]
                    language = LANGUAGES.get(text, None)
                    if language is not None:
                        # 1/ Indeed some language
                        # TODO Handle when language is different from the previous
                        # one, and we cannot use timestamped tokens to create chunks
                        if last_language and language != last_language and not return_timestamps:
                            previous_tokens.append(current_tokens)
                            resolved_tokens = self._find_longest_common_sequence(previous_tokens)
                            resolved_text = self.decode(resolved_tokens)
                            chunk["text"] = resolved_text
                            chunks.append(chunk)

                            # Flush all our temporary context
                            previous_tokens = []
                            current_tokens = []
                            chunk = new_chunk()
                        chunk["language"] = language
                        last_language = language
                    else:
                        # 2/ This is a regular special token, ignoring it
                        pass
                elif token >= timestamp_begin:
                    # 3/ Timestamp token
                    time = (token - timestamp_begin) * time_precision + time_offset
                    time = round(time, 2)
                    if last_timestamp and token >= last_timestamp:
                        # Whisper outputted a timestamp token, but it falls within
                        # our stride, so we're going to skip it for the time being
                        # and resolve this later
                        # Skip is necessary because timestamp tokens always come
                        # by pair, so we need to skip the next one too (which would mark the start of another chunk).
                        skip = True
                    elif skip or (previous_tokens and token < first_timestamp):
                        skip = False
                    elif chunk["timestamp"][0] is None:
                        chunk["timestamp"][0] = time
                    else:
                        # This is the end of the timestamp chunk
                        if time == chunk["timestamp"][0]:
                            # This is a bug in timestamp token output
                            # where we're taking the duplicate token
                            # as a stop where it should be a start.
                            # This is an issue in the underlying model output
                            pass
                        else:
                            chunk["timestamp"][1] = time
                            # Handling merges.
                            previous_tokens.append(current_tokens)
                            resolved_tokens = self._find_longest_common_sequence(previous_tokens)
                            resolved_text = self.decode(resolved_tokens)
                            chunk["text"] = resolved_text
                            chunks.append(chunk)

                            # Flush all our temporary context
                            previous_tokens = []
                            current_tokens = []
                            chunk = new_chunk()
                else:
                    # 4/ Regular token
                    # We just append to the list of all tokens so we can handle
                    # merges later and decode into text.
                    current_tokens.append(token)

            if "stride" in output:
                time_offset += chunk_len - stride_right

            # Leftover tokens
            if current_tokens:
                previous_tokens.append(current_tokens)
            elif not (any(p for p in previous_tokens)):
                # print("Flushing previous tokens (END)")
                chunk = new_chunk()
                previous_tokens = []
                current_tokens = []

        if previous_tokens:
            if return_timestamps:
                logger.warning(
                    "There was an error while processing timestamps, we haven't found a timestamp as last token. Was"
                    " WhisperTimeStampLogitsProcessor used?"
                )
            # Happens when we don't use timestamps
            resolved_tokens = self._find_longest_common_sequence(previous_tokens)
            # print("Flushing previous tokens (FINAL)")
            resolved_text = self.decode(resolved_tokens)
            chunk["text"] = resolved_text
            chunks.append(chunk)

        # Preparing and cleaning up the pipeline output
        full_text = "".join(chunk["text"] for chunk in chunks)
        if return_timestamps or return_language:
            for chunk in chunks:
                if not return_timestamps:
                    chunk.pop("timestamp")
                else:
                    chunk["timestamp"] = tuple(chunk["timestamp"])
                if not return_language:
                    chunk.pop("language")
            optional = {"chunks": chunks}
        else:
            optional = {}
        return full_text, optional

    @staticmethod
    def _find_longest_common_sequence(sequences):
        # It would be much harder to do O(n) because of fault tolerance.
        # We actually have a good property which is that the total sequence
        # MUST be those subsequences in order.
        left_sequence = sequences[0]
        left_length = len(left_sequence)
        total_sequence = []
        for right_sequence in sequences[1:]:
            # index = 0
            max_ = 0.0
            max_indices = (left_length, left_length, 0, 0)
            # Here we're sliding matches
            # [a, b, c, d]
            #          [c, d, f]
            # =        [c] == [d]
            #
            # [a, b, c, d]
            #       [c, d, f]
            # =     [c, d] == [c, d]
            #
            #
            # [a, b, c, d]
            #    [c, d, f]
            #
            # =  [b, c, d] == [c, d, f]
            #
            # [a, b, c, d]
            # [c, d, f]
            #
            # [a, b, c] == [c, d, f]
            #
            # [a, b, c, d]
            # [d, f]
            #
            # [a, b] == [d, f]
            #
            # [a, b, c, d]
            # [f]
            #
            # [a] == [f]
            right_length = len(right_sequence)
            for i in range(1, left_length + right_length):
                # epsilon to favor long perfect matches
                eps = i / 10000.0

                # Slightly convoluted because we don't want out of bound indices
                # This will be necessary for a small conflict resolution optimization
                # later
                left_start = max(0, left_length - i)
                left_stop = min(left_length, left_length + right_length - i)
                left = np.array(left_sequence[left_start:left_stop])

                right_start = max(0, i - left_length)
                right_stop = min(right_length, i)
                right = np.array(right_sequence[right_start:right_stop])

                # We can only match subsequences of the same size.
                if len(left) != len(right):
                    raise RuntimeError(
                        "There is a bug within whisper `decode_asr` function, please report it. Dropping to prevent bad inference."
                    )

                matches = np.sum(left == right)
                matching = matches / i + eps
                if matches > 1 and matching > max_:
                    max_ = matching
                    max_indices = (left_start, left_stop, right_start, right_stop)

            (left_start, left_stop, right_start, right_stop) = max_indices

            # This is a small conflict optimization since those sequences overlap
            # in audio.
            # We're going to give more confidence to the left sequence
            # for the left of the overlap,
            # and to the right of the sequence, for the right of the overlap
            left_mid = (left_stop + left_start) // 2
            right_mid = (right_stop + right_start) // 2
            total_sequence.extend(left_sequence[:left_mid])
            left_sequence = right_sequence[right_mid:]
            left_length = len(left_sequence)

        total_sequence.extend(left_sequence)

        return total_sequence
