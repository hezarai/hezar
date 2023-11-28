from dataclasses import dataclass
from typing import Iterable

from sklearn.metrics import f1_score

from ..configs import MetricConfig
from ..constants import Backends, MetricType
from ..registry import register_metric
from .metric import Metric


_required_backends = [
    Backends.SCIKIT,
]


@dataclass
class F1Config(MetricConfig):
    """
    Configuration class for F1 metric.

    Args:
        name (MetricType): The type of metric, F1 in this case.
        pos_label (int): Label of the positive class.
        average (str): Type of averaging for the F1 score.
        sample_weight (Iterable[float]): Sample weights for the F1 score.
        output_keys (tuple): Keys to filter the metric results for output.
    """
    name = MetricType.F1
    objective: str = "maximize"
    pos_label: int = 1
    average: str = "macro"
    sample_weight: Iterable[float] = None
    output_keys: tuple = ("f1",)


@register_metric("f1", config_class=F1Config)
class F1(Metric):
    """
    F1 metric for evaluating classification performance using sklearn's `f1_score`.

    Args:
        config (F1Config): Metric configuration object.
        **kwargs: Extra configuration parameters passed as kwargs to update the `config`.
    """
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
        """
        Computes the F1 score for the given predictions against targets.

        Args:
            predictions: Predicted labels.
            targets: Ground truth labels.
            labels: List of labels to include in the calculation.
            pos_label (int): Label of the positive class.
            average (str): Type of averaging for the F1 score.
            sample_weight (Iterable[float]): Sample weights for the F1 score.
            zero_division (str): Strategy to use for zero-division, default is "warn".
            n_decimals (int): Number of decimals for the final score.
            output_keys (tuple): Filter the output keys.

        Returns:
            dict: A dictionary of the metric results, with keys specified by `output_keys`.
        """
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
