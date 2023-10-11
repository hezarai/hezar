# -*- coding: utf-8 -*-
import csv

import datasets
from datasets.tasks import Summarization

logger = datasets.logging.get_logger(__name__)


_CITATION = """Citation"""

_DESCRIPTION = """Description"""

_DOWNLOAD_URLS = {
    "train": "https://huggingface.co/datasets/hezarai/xlsum-fa/resolve/main/xlsum-fa_train.csv",
    "test": "https://huggingface.co/datasets/hezarai/xlsum-fa/resolve/main/xlsum-fa_test.csv",
}


class DatasetNameConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(DatasetNameConfig, self).__init__(**kwargs)


class DatasetName(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        DatasetNameConfig(
            name="NAME FOR YOUR DATASET",
            version=datasets.Version("DATASET VERSION"),
            description=_DESCRIPTION,
        ),
    ]

    def _info(self):
        text_column = "text"
        summary_column = "summary"
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {text_column: datasets.Value("string"), summary_column: datasets.features.Value("string")}
            ),
            homepage="PUT PATH TO THE ORIGINAL DATASET HOME PAGE HERE (OPTIONAL BUT RECOMMENDED)",
            citation=_CITATION,
            task_templates=[Summarization(text_column=text_column, summary_column=summary_column)],
        )

    def _split_generators(self, dl_manager):
        """
        Returns SplitGenerators.
        """
        train_path = dl_manager.download_and_extract(_DOWNLOAD_URLS["train"])
        test_path = dl_manager.download_and_extract(_DOWNLOAD_URLS["test"])

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """
        Per each file_path read the csv file and iterate it.
        For each row yield a tuple of (id, {"text": ..., "summary": ..., ...})
        Each call to this method yields an output like below:
        ```
        (123, {"text": "...", "summary": "..."})
        ```
        """
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, quotechar='"', skipinitialspace=True)

            # Uncomment below line to skip the first row if your csv file has a header row
            # next(csv_reader, None)

            for id_, row in enumerate(csv_reader):
                text, label = row
                # Optional preprocessing here
                yield id_, {"text": text, "summary": label}
