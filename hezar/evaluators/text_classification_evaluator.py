from typing import Any, Callable, Dict

from .. import Recall
from ..constants import MetricType


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

