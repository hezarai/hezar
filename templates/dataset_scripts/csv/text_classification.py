# -*- coding: utf-8 -*-
import csv

import datasets
from datasets.tasks import TextClassification

logger = datasets.logging.get_logger(__name__)


_CITATION = """Citation"""
_DESCRIPTION = """Description"""

_DOWNLOAD_URLS = {
    "train": "PUT PATH TO TRAIN CSV FILE HERE (REQUIRED)",
    "dev": "PUT PATH TO DEV CSV FILE HERE (OPTIONAL)",
    "test": "PUT PATH TO TEST CSV FILE HERE (OPTIONAL BUT RECOMMENDED)",
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
        label_column = "label"
        # TODO PROVIDE THE LABELS HERE
        label_names = ["negative", "positive", "neutral"]
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {text_column: datasets.Value("string"), label_column: datasets.features.ClassLabel(names=label_names)}
            ),
            homepage="HOMEPAGE",
            citation=_CITATION,
            task_templates=[TextClassification(text_column=text_column, label_column=label_column)],
        )

    def _split_generators(self, dl_manager):
        """
        Return SplitGenerators.
        """
        train_path = dl_manager.download_and_extract(_DOWNLOAD_URLS["train"])
        test_path = dl_manager.download_and_extract(_DOWNLOAD_URLS["test"])

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    # TODO
    def _generate_examples(self, filepath):
        """
        Per each file_path read the csv file and iterate it.
        For each row yield a tuple of (id, {"text": ..., "label": ..., ...})
        Each call to this method yields an output like below:
        ```
        (123, {"text": "I liked it", "label": "positive"})
        ```
        """
        label2id = self.info.features[self.info.task_templates[0].label_column].str2int
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, quotechar='"', skipinitialspace=True)

            # Uncomment below line to skip the first row if your csv file has a header row
            # next(csv_reader, None)

            for id_, row in enumerate(csv_reader):
                text, label = row
                label = label2id(label)
                # Optional preprocessing here
                yield id_, {"text": text, "label": label}
