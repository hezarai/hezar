from typing import Any, Dict, Iterable
from dataclasses import dataclass

from sklearn.metrics import recall_score

from ..registry import register_metric
from ..configs import MetricConfig
from ..constants import MetricType
from .metric import Metric


@dataclass
class RecallConfig(MetricConfig):
    name: str = MetricType.RECALL
    pos_label: int = None
    average: str = "binary"
    sample_weight: Iterable[float] = None


@register_metric("recall", config_class=RecallConfig)
class Recall(Metric):
    def __init__(self, config: RecallConfig, **kwargs):
        super().__init__(config, **kwargs)

    def compute(
        self,
        predictions=None,
        targets=None,
        labels=None,
        pos_label=1,
        average="binary",
        sample_weight=None,
        zero_division="warn",
    ) -> Dict[str, Any]:

        pos_label = pos_label or self.config.pos_label
        average = average or self.config.average
        sample_weight = sample_weight or self.config.sample_weight

        score = recall_score(
            targets,
            predictions,
            labels=labels,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

        return {"recall": float(score) if score.size == 1 else score}
