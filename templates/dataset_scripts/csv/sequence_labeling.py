# TODO
"""
PROVIDE GENERAL DESCRIPTION OF ANYTHING REGARDING THIS LOADING SCRIPT OF THE DATASET. (OPTIONAL)
"""

import csv
import datasets

logger = datasets.logging.get_logger(__name__)

# TODO
_CITATION = """\
ORIGINAL CITATION HERE. (OPTIONAL BUT RECOMMENDED)
"""

# TODO
_DESCRIPTION = """\
THIS IS THE MAIN DESCRIPTION. IF YOUR DATASET IS FROM A PAPER, IT'S RECOMMENDED TO PROVIDE THE ABSTRACT OF THE PAPER. 
THIS VALUE WILL BE USED AS THE DATASET'S MAIN DESCRIPTION. TRY TO BE SHORT BUT EXPLICIT HERE. (OPTIONAL BUT RECOMMENDED) 
"""

# TODO
"""
PUT THE URL TO THE CSV FILES HERE. USUALLY WE UPLOAD THE FILES TO THE HUB ON THE SAME REPOSITORY THAT THIS SCRIPT LIVES.
FOR EXAMPLE:

```python
_DOWNLOAD_URLS = {
    "train": "https://huggingface.co/datasets/hezarai/lscp-500k/resolve/main/lscp-500k_train.csv",
    "test": "https://huggingface.co/datasets/hezarai/lscp-500k/resolve/main/lscp-500k_test.csv",
}
```
"""
_DOWNLOAD_URLS = {
    "train": "PUT PATH TO TRAIN CSV FILE HERE (REQUIRED)",
    "dev": "PUT PATH TO DEV CSV FILE HERE (OPTIONAL)",
    "test": "PUT PATH TO TEST CSV FILE HERE (OPTIONAL BUT RECOMMENDED)",
}

# TODO
"""
YOU HAVE TO PROVIDE A BUILDER CONFIG CLASS LIKE BELOW FOR UNDER YOUR DATASET'S NAME 
"""


# TODO
class DatasetNameConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(DatasetNameConfig, self).__init__(**kwargs)


# TODO
class DatasetName(datasets.GeneratorBasedBuilder):
    """OPTIONAL DOCSTRING FOR THIS CLASS"""

    BUILDER_CONFIGS = [
        DatasetNameConfig(
            name="NAME FOR YOUR DATASET",
            version=datasets.Version("DATASET VERSION"),
            description=_DESCRIPTION,  # PUT YOUR OWN DESCRIPTION IF THE ABOVE `_DESCRIPTION` IS NOT PROVIDED
        ),
    ]

    # TODO
    def _info(self):
        """
        FOR SEQUENCE LABELING FOLLOW THE BELOW STRUCTURE.
        """
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            # TODO
            # THESE FEATURES ARE THE COLUMN NAMES FOR YOUR DATASET. FOR SEQUENCE LABELING FOLLOW THIS STRUCTURE
            features=datasets.Features(
                {
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    # TODO YOU SHOULD PUT THE EXTRACTED UNIQUE TAGS IN YOUR DATASET HERE. THIS LIST IS JUST AN EXAMPLE
                    """
                    TO EXTRACT UNIQUE TAGS FROM A PANDAS DATAFRAME USE THIS CODE AND PASTE THE OUTPUT LIST HERE.

                    ```python
                    unique_tags = df["TAGS_COLUMN_NAME"].explode().unique()
                    print(unique_tags)
                    ```
                    """
                    "pos_tags": datasets.Sequence(  # USE `pos_tags`, `ner_tags`, `chunk_tags`, etc.
                        datasets.features.ClassLabel(
                            names=[
                                'P',
                                'Ne',
                                'PRO',
                                'CONJ',
                                'N',
                                'PUNC',
                                'Pe',
                                'ADV',
                                'V',
                                'AJ',
                                'AJe',
                                'DET',
                                'POSTP',
                                'NUM',
                                'DETe',
                                'NUMe',
                                'PROe',
                                'ADVe',
                                'RES',
                                'CL',
                                'INT',
                                'CONJe',
                                'RESe',
                            ]
                        )
                    ),
                }
            ),
            homepage="PUT PATH TO THE ORIGINAL DATASET HOME PAGE HERE (OPTIONAL BUT RECOMMENDED)",
            citation=_CITATION,
        )

    # TODO USUALLY YOU WON'T NEED TO CHANGE THIS METHOD. THIS JUST RETURNS A GENERATOR FOR THE EXTRACTED FILES
    def _split_generators(self, dl_manager):
        """
        Returns SplitGenerators.
        """

        # THIS DOWNLOADS AND EXTRACT THE CSV FILES TO CACHE PATH
        train_path = dl_manager.download_and_extract(_DOWNLOAD_URLS["train"])
        test_path = dl_manager.download_and_extract(_DOWNLOAD_URLS["test"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]

    # TODO
    def _generate_examples(self, filepath):
        """
        PER EACH file_path READ THE CSV FILE AND ITERATE IT.
        OR EACH ROW YIELD A TUPLE OF (ID, {"tokens": ..., "tags": ..., ...})
        EACH CALL TO THIS METHOD YIELDS AN OUTPUT LIKE BELOW:
        ```
        (124, {"tokens": ["hello", "world"], "pos_tags": ["NOUN", "NOUN"]})
        ```
        """
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', skipinitialspace=True
            )

            # UNCOMMENT BELOW LINE TO SKIP THE FIRST ROW IF YOUR CSV FILE HAS A HEADER ROW
            # next(csv_reader, None)

            for id_, row in enumerate(csv_reader):
                tokens, pos_tags = row
                # YOUR OWN PREPROCESSING HERE (OPTIONAL)
                yield id_, {"tokens": tokens, "pos_tags": pos_tags}
