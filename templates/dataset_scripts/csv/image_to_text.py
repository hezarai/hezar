import csv
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """Citation"""

_DESCRIPTION = """Dataset description"""

_DOWNLOAD_URLS = {
    "train": "https://huggingface.co/datasets/hezarai/persian-license-plate-v1/resolve/main/persian_license_plate_train.csv",
    "test": "https://huggingface.co/datasets/hezarai/persian-license-plate-v1/resolve/main/persian_license_plate_test.csv",
    'train_dataset': "https://huggingface.co/datasets/hezarai/persian-license-plate-v1/resolve/main/persian_license_plate_train.zip",
    'test_dataset': "https://huggingface.co/datasets/hezarai/persian-license-plate-v1/resolve/main/persian_license_plate_test.zip",
}


class ImageToTextDatasetConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(ImageToTextDatasetConfig, self).__init__(**kwargs)


class PersianLicensePlate(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ImageToTextDatasetConfig(
            name="image to text dataset name",
            version=datasets.Version("1.0.0"),
            description=_DESCRIPTION,
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "label": datasets.Value("string"),
                    "image_path": datasets.Value("string"),
                }
            ),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """
        Return SplitGenerators.
        """

        train_path = dl_manager.download_and_extract(_DOWNLOAD_URLS["train"])
        test_path = dl_manager.download_and_extract(_DOWNLOAD_URLS["test"])

        archive_path = dl_manager.download(_DOWNLOAD_URLS['train_dataset'])
        train_extracted_path = dl_manager.extract(archive_path) if not dl_manager.is_streaming else ""

        archive_path = dl_manager.download(_DOWNLOAD_URLS['test_dataset'])
        test_extracted_path = dl_manager.extract(archive_path) if not dl_manager.is_streaming else ""

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path, "dataset_dir": train_extracted_path}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": test_path, "dataset_dir": test_extracted_path}
            ),
        ]

    def _generate_examples(self, filepath, dataset_dir):
        logger.info("⏳ Generating examples from = %s", filepath)

        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, quotechar='"', skipinitialspace=True)

            # Skip header
            next(csv_reader, None)

            for id_, row in enumerate(csv_reader):
                label, filename = row
                image_path = os.path.join(dataset_dir, filename)
                yield id_, {"image_path": image_path, "label": label}
