from enum import Enum
from typing import Iterator, Iterable, Dict, Any, Callable

from ..constants import MetricType
from .. import Recall


# class TextClassificationMetrics(str, Enum):
#     RECALL: Metric = Recall()
#     F1: Metric = F1()


class TextClassificationEvaluator:
    AVAILABLE_METRICS = ("f1", Recall, Recall(), MetricType.F1)

    def __init__(self, metrics_dict: Dict[str, Callable]):
        self.metrics_dict = metrics_dict

    def _compute(self, preds, labels, **kwargs) -> Dict[str, Any]:
        results = {}
        for metric_name, metric_fn in self.metrics_dict.items():
            if metric_fn is not None:
                results[metric_name] = metric_fn(preds, labels).item()

        return results
    # def _preprocessing(self, preds, labels):
    #     pass

#
#
#
# # def a(b: Iterable[str | int]):
# #     pass
# #
# #
# # a([1, 2, 3])
#


class TextClassificationMetrics(str, Enum):
    RECALL = "recall"


if __name__ == '__main__':
    print(TextClassificationMetrics.RECALL)