from dataclasses import dataclass
from typing import Iterable

from sklearn.metrics import accuracy_score

from ..configs import MetricConfig
from ..constants import Backends, MetricType
from ..registry import register_metric
from .metric import Metric


_required_backends = [
    Backends.SCIKIT,
]


@dataclass
class AccuracyConfig(MetricConfig):
    name = MetricType.ACCURACY
    normalize: bool = True
    sample_weight: Iterable[float] = None
    output_keys: tuple = ("accuracy",)


@register_metric("accuracy", config_class=AccuracyConfig)
class Accuracy(Metric):
    required_backends = _required_backends

    def __init__(self, config: AccuracyConfig, **kwargs):
        super().__init__(config, **kwargs)

    def compute(
        self,
        predictions=None,
        targets=None,
        normalize=None,
        sample_weight=None,
        n_decimals=None,
        output_keys=None,
    ):
        normalize = normalize or self.config.normalize
        sample_weight = sample_weight or self.config.sample_weight
        n_decimals = n_decimals or self.config.n_decimals
        output_keys = output_keys or self.config.output_keys

        score = accuracy_score(
            targets,
            predictions,
            normalize=normalize,
            sample_weight=sample_weight,
        )

        results = {"accuracy": round(float(score), n_decimals)}

        if output_keys:
            results = {k: v for k, v in results.items() if k in output_keys}

        return results
