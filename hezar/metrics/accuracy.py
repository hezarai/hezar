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
    objective: str = "maximize"
    normalize: bool = True
    sample_weight: Iterable[float] = None
    output_keys: tuple = ("accuracy",)


@register_metric("accuracy", config_class=AccuracyConfig)
class Accuracy(Metric):
    """
    Accuracy metric for numeric arrays backed by Scikit-learn's `accuracy_score`.

    Args:
        config (AccuracyConfig): Metric config object
        **kwargs: Extra config parameters passed as kwargs to update the `config`
    """
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
        """
        Compute the accuracy score for the given predictions against targets.

        Args:
            predictions: A list of prediction labels
            targets: A list of ground truth labels
            normalize: Whether to normalize the inputs or not
            sample_weight: Sample weight
            n_decimals: Floating point decimals for the final score
            output_keys: Filter the output keys

        Returns:
            A dictionary of the metric results
        """
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
