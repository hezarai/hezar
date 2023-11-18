"""
In this test suite, we don't try to actually train or finetune a model so that it's safe to just load a trained model
for the same task and test trainer's functionalities. This way we don't need to provide inter-related config parameters.
For example, a model might need to get its `id2label` config parameter from the dataset's `id2label` which can make this
test suite complex and require redundant code.
"""

import pytest

from hezar.data import Dataset
from hezar.models import ModelConfig
from hezar.builders import build_model
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
            "device": "cpu",
            "batch_size": 8,
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
            "device": "cpu",
            "batch_size": 8,
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
            "device": "cpu",
            "batch_size": 8,
            "metrics": ["rouge"]
        }
    },
    "ocr": {
        "dataset": {
            "path": "hezarai/persian-license-plate-v1",
            "config": {
                "max_length": 8,
                "reverse_text": True,
            }
        },
        "model": {
            "path": "hezarai/crnn-fa-64x256-license-plate-recognition",
        },
        "config": {
            "task": "image2text",
            "device": "cpu",
            "batch_size": 8,
            "metrics": ["cer"]
        }
    },
}


@pytest.mark.parametrize("task", tasks_setups.keys())
def test_trainer(task):
    setup = tasks_setups[task]

    # Datasets
    train_dataset = Dataset.load(setup["dataset"]["path"], split="train", **setup["dataset"]["config"])
    eval_dataset = Dataset.load(setup["dataset"]["path"], split="test", **setup["dataset"]["config"])

    # Model & Preprocessor
    model_config = ModelConfig.load(setup["model"]["path"])
    model = build_model(model_config.name, config=model_config)
    preprocessor = Preprocessor.load(setup["model"]["path"])

    # Trainer config
    config = TrainerConfig(**setup["config"])

    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        preprocessor=preprocessor
    )

    batch_sample = next(iter(trainer.train_dataloader))
    train_batch_results = trainer.training_step(batch_sample)
    assert "loss" in train_batch_results, "Training step output must contain loss value!"

    eval_batch_sample = next(iter(trainer.eval_dataloader))
    eval_batch_results = trainer.evaluation_step(eval_batch_sample)
    assert isinstance(eval_batch_results, dict), "Evaluation step must return a dictionary of output metrics"
    # TODO add a `all_metrics_keys` property to MetricsHandler and check if `eval_batch_results` has all of them here
