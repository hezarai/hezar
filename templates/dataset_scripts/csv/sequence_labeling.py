import csv

import datasets

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
            name="",
            version=datasets.Version(""),
            description=_DESCRIPTION,
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    # TODO YOU SHOULD PUT THE EXTRACTED UNIQUE TAGS IN YOUR DATASET HERE. THIS LIST IS JUST AN EXAMPLE
                    """
                    To extract unique tags from a pandas dataframe use this code and paste the output list below.

                    ```python
                    unique_tags = df["TAGS_COLUMN_NAME"].explode().unique()
                    print(unique_tags)
                    ```
                    """
                    "pos_tags": datasets.Sequence(  # USE `pos_tags`, `ner_tags`, `chunk_tags`, etc.
                        datasets.features.ClassLabel(names=["YOU SHOULD PUT THE UNIQUE TAGS LIST HERE"])  # TODO
                    ),
                }
            ),
            homepage="PUT PATH TO THE ORIGINAL DATASET HOME PAGE HERE (OPTIONAL BUT RECOMMENDED)",
            citation=_CITATION,
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
        For each row yield a tuple of (id, {"tokens": ..., "tags": ..., ...})
        Each call to this method yields an output like below:
        ```
        (124, {"tokens": ["hello", "world"], "pos_tags": ["NOUN", "NOUN"]})
        ```
        """
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, quotechar='"', skipinitialspace=True)

            # Uncomment below line to skip the first row if your csv file has a header row
            # next(csv_reader, None)

            for id_, row in enumerate(csv_reader):
                tokens, pos_tags = row
                # Optional preprocessing here
                yield id_, {"tokens": tokens, "pos_tags": pos_tags}
