from hezar.constants import TaskType
from hezar.models import Model
from hezar.utils import Logger


logger = Logger(__name__)

task_to_example_repo = {
    TaskType.TEXT_CLASSIFICATION: "hezarai/roberta-fa-sentiment-dksf",
    TaskType.SEQUENCE_LABELING: "hezarai/bert-fa-pos-lscp-500k",
    TaskType.LANGUAGE_MODELING: "hezarai/bert-base-fa",
    TaskType.SPEECH_RECOGNITION: "hezarai/whisper-small-fa",
    TaskType.IMAGE2TEXT: "hezarai/crnn-base-fa-64x256",
    TaskType.TEXT_GENERATION: "hezarai/t5-base-fa"
}


def test_model_loading():
    models = []
    for task, repo in task_to_example_repo.items():
        model = Model.load(repo)
        models.append(model)
        logger.info(f"Successfully loaded {repo} for `{task}`")
    return models


def test_model_saving():
    models = test_model_loading()
    for model in models:
        model.save(model.config.name)
        logger.info(f"Successfully saved {model.config.name}")
