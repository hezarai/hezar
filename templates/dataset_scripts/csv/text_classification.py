# TODO
"""
PROVIDE GENERAL DESCRIPTION OF ANYTHING REGARDING THIS LOADING SCRIPT OF THE DATASET. (OPTIONAL)
"""

import csv

import datasets
from datasets.tasks import TextClassification


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
    "train": "https://huggingface.co/datasets/hezar-ai/sentiment_digikala_snappfood/raw/main/sentiment_digikala_snappfood_train.csv",
    "test": "https://huggingface.co/datasets/hezar-ai/sentiment_digikala_snappfood/raw/main/sentiment_digikala_snappfood_test.csv"
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
        # PROVIDE TEXT_COLUMN NAME AND LABEL_COLUMN NAME IF IT'S DIFFERENT FROM BELOW VALUES
        text_column = "text"
        label_column = "label"
        # TODO PROVIDE THE LABELS HERE
        label_names = ["negative", "positive", "neutral"]
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {text_column: datasets.Value("string"),
                 label_column: datasets.features.ClassLabel(names=label_names)}
            ),
            homepage="PUT PATH TO THE ORIGINAL DATASET HOME PAGE HERE (OPTIONAL BUT RECOMMENDED)",
            citation=_CITATION,
            task_templates=[TextClassification(text_column=text_column, label_column=label_column)],
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
        OR EACH ROW YIELD A TUPLE OF (ID, {"text": ..., "label": ..., ...})
        EACH CALL TO THIS METHOD YIELDS AN OUTPUT LIKE BELOW:
        ```
        (123, {"text": "I liked it", "label": "positive"})
        ```
        """
        label2id = self.info.features[self.info.task_templates[0].label_column].str2int
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', skipinitialspace=True
            )

            # UNCOMMENT BELOW LINE TO SKIP THE FIRST ROW IF YOUR CSV FILE HAS A HEADER ROW
            # next(csv_reader, None)

            for id_, row in enumerate(csv_reader):
                text, label = row
                label = label2id(label)
                # YOUR OWN PREPROCESSING HERE (OPTIONAL)
                yield id_, {"text": text, "label": label}
