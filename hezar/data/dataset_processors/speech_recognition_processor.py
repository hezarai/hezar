from .dataset_processor import DatasetProcessor


class SpeechRecognitionDatasetProcessor(DatasetProcessor):
    def __init__(
        self,
        feature_extractor,
        tokenizer,
        sampling_rate=16000,
        audio_array_padding="longest",
        max_audio_array_length=None,
        labels_padding="longest",
        labels_max_length=None,
        audio_field="audio",
        transcript_field="transcript",
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sampling_rate = sampling_rate
        self.audio_array_padding = audio_array_padding
        self.max_audio_array_length = max_audio_array_length
        self.labels_padding = labels_padding
        self.labels_max_length = labels_max_length
        self.audio_field = audio_field
        self.transcript_field = transcript_field

    def process_single(self, data):
        """
        Process a single speech recognition example.

        Args:
            data: A data example containing audio and its transcript.

        Returns:
            dict: Processed input features and labels.
        """
        audio_array = data[self.audio_field]["array"]
        transcript = data[self.transcript_field]

        # Extract input features from audio
        input_features = self.feature_extractor(
            audio_array,
            sampling_rate=self.sampling_rate,
            padding=self.audio_array_padding,
            max_length=self.max_audio_array_length,
            return_tensors="torch",
        )["input_features"]

        # Tokenize the transcript
        labels = self.tokenizer(
            transcript,
            padding=self.labels_padding,
            max_length=self.labels_max_length,
            return_tensors="torch",
        )

        data["input_features"] = input_features
        data["labels"] = labels["token_ids"]
        data["attention_mask"] = labels["attention_mask"]

        return data

    def process_batch(self, data):
        """
        Process a batch of speech recognition examples.

        Args:
            data: A batch of data examples containing audio arrays and their corresponding transcripts.

        Returns:
            dict: Batch of processed input features and labels.
        """
        audio_arrays = [x["array"] for x in data[self.audio_field]]
        transcripts = data[self.transcript_field]

        # Extract input features in batch
        input_features = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sampling_rate,
            padding=self.audio_array_padding,
            max_length=self.max_audio_array_length,
            return_tensors="torch",
        )["input_features"]

        # Tokenize transcripts in batch
        labels = self.tokenizer(
            transcripts,
            padding=self.labels_padding,
            max_length=self.labels_max_length,
            return_tensors="torch",
        )

        data["input_features"] = input_features
        data["labels"] = labels["token_ids"]
        data["attention_mask"] = labels["attention_mask"]

        return data
