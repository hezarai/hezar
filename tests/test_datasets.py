from hezar.constants import TaskType
from hezar.data import Dataset
from hezar.utils import Logger


logger = Logger(__name__)

task_to_dataset_example_repo = {
    TaskType.TEXT_CLASSIFICATION: "hezarai/sentiment-dksf",
    TaskType.SEQUENCE_LABELING: "hezarai/parstwiner",
    TaskType.TEXT_GENERATION: "hezarai/xlsum-fa",
}


def test_dataset_loading():
    datasets = []
    for task, repo in task_to_dataset_example_repo.items():
        d = Dataset.load(repo)
        logger.info(f"Successfully loaded {repo} for {task}")
        datasets.append(d)
    return datasets
