from hezar.configs import TaskConfig
from hezar.tasks import BaseTask


class TextClassificationTask(BaseTask):
    def __init__(self, config: TaskConfig):
        super(TextClassificationTask, self).__init__(config)
