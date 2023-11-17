from dataclasses import dataclass, field

from datasets import load_dataset

from ...configs import DatasetConfig
from ...constants import TaskType
from ...preprocessors import Tokenizer
from ...registry import register_dataset
from ...utils import Logger
from ..data_collators import TextGenerationDataCollator
from .dataset import Dataset


logger = Logger(__name__)


@dataclass
class TextSummarizationDatasetConfig(DatasetConfig):
    """
    Configuration class for text summarization datasets.

    Args:
        path (str): Path to the dataset.
        tokenizer_path (str): Path to the tokenizer file.
        prefix (str): Prefix for conditional generation.
        text_field (str): Field name for text in the dataset.
        summary_field (str): Field name for summary in the dataset.
        title_field (str): Field name for title in the dataset.
        max_length (int): Maximum length of text.
        max_target_length (int): Maximum length of the target summary.
    """

    name = "text_summarization"
    task: TaskType = TaskType.TEXT_GENERATION
    path: str = None
    tokenizer_path: str = None
    prefix: str = None
    text_field: str = None
    summary_field: str = None
    title_field: str = None
    max_length: int = None
    max_target_length: int = None


@register_dataset("text_summarization", config_class=TextSummarizationDatasetConfig)
class TextSummarizationDataset(Dataset):
    """
    A text summarization dataset class.
    As of now this class is intended for datasets existing on the Hub!

    Args:
        config (TextSummarizationDatasetConfig): Dataset config object.
        split: Which split to use.
        **kwargs: Extra config parameters to assign to the original config.
    """

    def __init__(self, config: TextSummarizationDatasetConfig, split=None, **kwargs):
        """
        Initializes a new TextSummarizationDataset instance.

        Args:
            config (TextSummarizationDatasetConfig): The configuration object for the dataset.
            split: Dataset split, defaults to None.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(config, **kwargs)
        self.dataset = self._load(split)
        self.tokenizer = self._build_tokenizer()
        self.data_collator = TextGenerationDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            max_target_length=self.config.max_target_length,
            padding_type="max_length" if self.config.max_length else "longest",
        )

    def _load(self, split):
        """
        Load the dataset.

        Args:
            split: Dataset split.

        Returns:
            The whole dataset.

        """
        # TODO: In case we want to make this class work on other types like csv, json, etc. we have to do it here.
        dataset = load_dataset(self.config.path, split=split, cache_dir=self.cache_dir)
        return dataset

    def _build_tokenizer(self):
        """
        Build the tokenizer.

        Returns:
            Tokenizer: The tokenizer.

        """
        if self.config.tokenizer_path:
            tokenizer = Tokenizer.load(self.config.tokenizer_path)
        else:
            logger.warning(
                "This dataset requires a tokenizer to work. Provide it in config as `tokenizer_path` "
                "or set it manually as `dataset.tokenizer = your_tokenizer` after building the dataset."
            )
            tokenizer = None
        return tokenizer

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Tokenize inputs and return a dict containing ids, masks, labels, etc.

        Args:
            index: Sample index.

        Returns:
            dict: The input data.

        """
        text = self.dataset[index][self.config.text_field]
        if self.config.prefix is not None:
            text = self.config.prefix + text  # for conditional generation we might need a static prefix
        summary = self.dataset[index][self.config.summary_field]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_length,
            padding="max_length" if self.config.max_length else "longest",
            return_attention_mask=True,
        )
        labels = self.tokenizer(
            summary,
            return_tensors="pt",
            max_length=self.config.max_length,
            padding="max_length" if self.config.max_target_length else "longest",
            return_attention_mask=True,
        )

        inputs["labels"] = labels["token_ids"]

        return inputs
