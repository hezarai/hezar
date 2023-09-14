from dataclasses import dataclass
from typing import Iterable

from sklearn.metrics import f1_score

from ..configs import MetricConfig
from ..constants import MetricType, Backends
from ..registry import register_metric
from .metric import Metric

_required_backends = [
    Backends.SCIKIT,
]


@dataclass
class F1Config(MetricConfig):
    name = MetricType.F1
    pos_label: int = 1
    average: str = "macro"
    sample_weight: Iterable[float] = None
    output_keys: tuple = ("f1",)


@register_metric("f1", config_class=F1Config)
class F1(Metric):
    required_backends = _required_backends

    def __init__(self, config: F1Config, **kwargs):
        super().__init__(config, **kwargs)

    def compute(
        self,
        predictions=None,
        targets=None,
        labels=None,
        pos_label=1,
        average=None,
        sample_weight=None,
        zero_division="warn",
        n_decimals=None,
        output_keys=None,
    ):
        pos_label = pos_label or self.config.pos_label
        average = average or self.config.average
        sample_weight = sample_weight or self.config.sample_weight
        n_decimals = n_decimals or self.config.n_decimals
        output_keys = output_keys or self.config.output_keys

        score = f1_score(
            targets,
            predictions,
            labels=labels,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

        score = float(score) if score.size == 1 else score

        results = {"f1": round(float(score), n_decimals)}

        if output_keys:
            results = {k: v for k, v in results.items() if k in output_keys}

        return results
