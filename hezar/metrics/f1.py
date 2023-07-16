from dataclasses import dataclass
from typing import Iterable

from sklearn.metrics import f1_score

from ..configs import MetricConfig
from ..constants import MetricType
from ..registry import register_metric
from .metric import Metric


@dataclass
class F1Config(MetricConfig):
    name: str = MetricType.F1
    pos_label: int = None
    average: str = None
    sample_weight: Iterable[float] = None


@register_metric("f1", config_class=F1Config)
class F1(Metric):
    def __init__(self, config: F1Config, **kwargs):
        super().__init__(config, **kwargs)

    def compute(
        self,
        predictions=None,
        targets=None,
        labels=None,
        pos_label=1,
        average=None,
        sample_weight=None
    ):
        pos_label = pos_label or self.config.pos_label
        average = average or self.config.average
        sample_weight = sample_weight or self.config.sample_weight

        score = f1_score(
            targets,
            predictions,
            labels=labels,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
        )

        return float(score) if score.size == 1 else score
