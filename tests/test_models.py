from hezar import Model
from hezar.constants import TaskType


task_to_example_repo = {
    TaskType.TEXT_CLASSIFICATION: "hezarai/roberta-fa-sentiment-dksf",
    TaskType.SEQUENCE_LABELING: "hezarai/bert-fa-pos-lscp-500k",
    TaskType.LANGUAGE_MODELING: "hezarai/bert-base-fa",
    TaskType.SPEECH_RECOGNITION: "hezarai/whisper-small-fa",
}


def test_model_loading():
    models = []
    for task, repo in task_to_example_repo.items():
        model = Model.load(repo)
        models.append(model)
        print(f"Successfully loaded {repo} for `{task}`")
    return models


def test_model_saving():
    models = test_model_loading()
    for model in models:
        model.save(model.config.name)


