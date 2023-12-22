"""
In this test suite, we don't try to actually train or finetune a model so that it's safe to just load a trained model
for the same task and test trainer's functionalities. This way we don't need to provide inter-related config parameters.
For example, a model might need to get its `id2label` config parameter from the dataset's `id2label` which can make this
test suite complex and require redundant code.
"""

import shutil

import pytest

from hezar.builders import build_model
from hezar.data import Dataset
from hezar.models import ModelConfig
from hezar.preprocessors import Preprocessor
from hezar.trainer import Trainer, TrainerConfig


tasks_setups = {
    "text_classification": {
        "dataset": {
            "path": "hezarai/sentiment-dksf",
            "config": {
                "tokenizer_path": "hezarai/bert-base-fa",
            }
        },
        "model": {
            "path": "hezarai/bert-fa-sentiment-dksf",
        },
        "config": {
            "task": "text_classification",
            "metrics": ["accuracy", "f1", "precision", "recall"]
        }
    },
    "sequence_labeling": {
        "dataset": {
            "path": "hezarai/lscp-pos-500k",
            "config": {
                "tokenizer_path": "hezarai/bert-base-fa",
            }
        },
        "model": {
            "path": "hezarai/distilbert-fa-pos-lscp-500k",
        },
        "config": {
            "task": "sequence_labeling",
            "metrics": ["seqeval"]
        }
    },
    "text_summarization": {
        "dataset": {
            "path": "hezarai/xlsum-fa",
            "config": {
                "tokenizer_path": "hezarai/t5-base-fa",
            }
        },
        "model": {
            "path": "hezarai/t5-base-fa",
        },
        "config": {
            "task": "text_generation",
            "metrics": ["rouge"]
        }
    },
    "ocr": {
        "dataset": {
            "path": "hezarai/persian-license-plate-v1",
            "config": {
                "max_length": 8,
                "reverse_digits": True,
            }
        },
        "model": {
            "path": "hezarai/crnn-fa-64x256-license-plate-recognition",
        },
        "config": {
            "task": "image2text",
            "metrics": ["cer"]
        }
    },
    "image-captioning": {
        "dataset": {
            "path": "hezarai/flickr30k-fa",
            "config": {
                "max_length": 32,
                "tokenizer_path": "hezarai/vit-roberta-fa-base"
            }
        },
        "model": {
            "path": "hezarai/vit-roberta-fa-base",
        },
        "config": {
            "task": "image2text",
            "metrics": ["wer"]
        }
    },
}

common_train_config = {
    "output_dir": "tests-tmp-train-dir",
    "batch_size": 2,
    "num_epochs": 1,
    "use_cpu": True,
}


@pytest.mark.parametrize("task", tasks_setups.keys())
def test_trainer(task):
    setup = tasks_setups[task]

    # Datasets
    train_dataset = Dataset.load(setup["dataset"]["path"], split="train[:4]", **setup["dataset"]["config"])
    eval_dataset = Dataset.load(setup["dataset"]["path"], split="test[:4]", **setup["dataset"]["config"])

    # Model & Preprocessor
    model_config = ModelConfig.load(setup["model"]["path"])
    model = build_model(model_config.name, config=model_config)
    preprocessor = Preprocessor.load(setup["model"]["path"])

    # Trainer config
    config = TrainerConfig(**common_train_config, **setup["config"])

    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preprocessor=preprocessor
    )
    trainer.train()

    shutil.rmtree(config.output_dir)

