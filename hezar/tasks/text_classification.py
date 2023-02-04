from hezar.configs import TrainConfig
from hezar.tasks import BaseTask


class TextClassificationTask(BaseTask):
    def __init__(self, config: TrainConfig):
        super(TextClassificationTask, self).__init__(config)
