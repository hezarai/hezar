import csv
import os

import datasets
from tqdm import tqdm


_DESCRIPTION = """\
Persian portion of the common voice 13 dataset, gathered and maintained by Hezar AI.
"""

_CITATION = """\
@inproceedings{commonvoice:2020,
  author = {Ardila, R. and Branson, M. and Davis, K. and Henretty, M. and Kohler, M. and Meyer, J. and Morais, R. and Saunders, L. and Tyers, F. M. and Weber, G.},
  title = {Common Voice: A Massively-Multilingual Speech Corpus},
  booktitle = {Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)},
  pages = {4211--4215},
  year = 2020
}
"""

_HOMEPAGE = "https://commonvoice.mozilla.org/en/datasets"

_LICENSE = "https://creativecommons.org/publicdomain/zero/1.0/"

_BASE_URL = "https://huggingface.co/datasets/hezarai/common-voice-13-fa/resolve/main/"

_AUDIO_URL = _BASE_URL + "audio/{split}.zip"

_TRANSCRIPT_URL = _BASE_URL + "transcripts/{split}.tsv"


class CommonVoiceFaConfig(datasets.BuilderConfig):
    """BuilderConfig for CommonVoice."""

    def __init__(self, **kwargs):
        super(CommonVoiceFaConfig, self).__init__(**kwargs)


class CommonVoice(datasets.GeneratorBasedBuilder):
    DEFAULT_WRITER_BATCH_SIZE = 1000

    BUILDER_CONFIGS = [
        CommonVoiceFaConfig(
            name="commonvoice-13-fa",
            version="1.0.0",
            description=_DESCRIPTION,
        )
    ]

    def _info(self):
        features = datasets.Features(
            {
                "client_id": datasets.Value("string"),
                "path": datasets.Value("string"),
                "audio": datasets.features.Audio(sampling_rate=48_000),
                "sentence": datasets.Value("string"),
                "up_votes": datasets.Value("int64"),
                "down_votes": datasets.Value("int64"),
                "age": datasets.Value("string"),
                "gender": datasets.Value("string"),
                "accent": datasets.Value("string"),
                "locale": datasets.Value("string"),
                "segment": datasets.Value("string"),
                "variant": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            version=self.config.version,
        )

    def _split_generators(self, dl_manager):
        splits = ("train", "dev", "test")
        audio_urls = {split: _AUDIO_URL.format(split=split) for split in splits}

        archive_paths = dl_manager.download(audio_urls)
        local_extracted_archive_paths = dl_manager.extract(archive_paths) if not dl_manager.is_streaming else {}

        transcript_urls = {split: _TRANSCRIPT_URL.format(split=split) for split in splits}
        transcript_paths = dl_manager.download_and_extract(transcript_urls)

        split_generators = []
        split_names = {
            "train": datasets.Split.TRAIN,
            "dev": datasets.Split.VALIDATION,
            "test": datasets.Split.TEST,
        }
        for split in splits:
            split_generators.append(
                datasets.SplitGenerator(
                    name=split_names.get(split, split),
                    gen_kwargs={
                        "local_extracted_archive_paths": local_extracted_archive_paths.get(split),
                        "archives": [dl_manager.iter_archive(archive_paths.get(split))],
                        "transcript_path": transcript_paths[split],
                    },
                ),
            )

        return split_generators

    def _generate_examples(self, local_extracted_archive_paths, archives, transcript_path):
        data_fields = list(self._info().features.keys())
        metadata = {}
        with open(transcript_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in tqdm(reader, desc="Reading metadata..."):
                if not row["path"].endswith(".mp3"):
                    row["path"] += ".mp3"
                # accent -> accents in CV 8.0
                if "accents" in row:
                    row["accent"] = row["accents"]
                    del row["accents"]
                # if data is incomplete, fill with empty values
                for field in data_fields:
                    if field not in row:
                        row[field] = ""
                metadata[row["path"]] = row

        for i, audio_archive in enumerate(archives):
            for path, file in audio_archive:
                _, filename = os.path.split(path)
                if filename in metadata:
                    result = dict(metadata[filename])
                    # set the audio feature and the path to the extracted file
                    path = os.path.join(local_extracted_archive_paths[i], path) if local_extracted_archive_paths else path
                    result["audio"] = {"path": path, "bytes": file.read()}
                    result["path"] = path
                    yield path, result
