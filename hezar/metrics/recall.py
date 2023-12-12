from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sklearn.metrics import recall_score

from ..configs import MetricConfig
from ..constants import Backends, MetricType
from ..registry import register_metric
from .metric import Metric


_required_backends = [
    Backends.SCIKIT,
]


@dataclass
class RecallConfig(MetricConfig):
    """
    Configuration class for Recall metric.

    Args:
        name (MetricType): The type of metric, Recall in this case.
        pos_label (int): Label of the positive class.
        average (str): Type of averaging for the recall score.
        sample_weight (Iterable[float]): Sample weights for the recall score.
        zero_division (str | float): Strategy for zero-division, default is 0.0.
        output_keys (tuple): Keys to filter the metric results for output.
    """
    name = MetricType.RECALL
    objective: str = "maximize"
    pos_label: int = 1
    average: str = "macro"
    sample_weight: Iterable[float] = None
    zero_division: str | float = 0.0
    output_keys: tuple = ("recall",)


@register_metric("recall", config_class=RecallConfig)
class Recall(Metric):
    """
    Recall metric for evaluating classification performance using sklearn's `recall_score`.

    Args:
        config (RecallConfig): Metric configuration object.
        **kwargs: Extra configuration parameters passed as kwargs to update the `config`.
    """
    required_backends = _required_backends

    def __init__(self, config: RecallConfig, **kwargs):
        super().__init__(config, **kwargs)

    def compute(
        self,
        predictions=None,
        targets=None,
        labels=None,
        pos_label=None,
        average=None,
        sample_weight=None,
        zero_division=None,
        n_decimals=None,
        output_keys=None,
    ):
        """
        Computes the Recall score for the given predictions against targets.

        Args:
            predictions: Predicted labels.
            targets: Ground truth labels.
            labels: List of labels to include in the calculation.
            pos_label (int): Label of the positive class.
            average (str): Type of averaging for the recall score.
            sample_weight (Iterable[float]): Sample weights for the recall score.
            zero_division (str | float): Strategy for zero-division, default is 0.0.
            n_decimals (int): Number of decimals for the final score.
            output_keys (tuple): Filter the output keys.

        Returns:
            dict: A dictionary of the metric results, with keys specified by `output_keys`.
        """
        pos_label = pos_label or self.config.pos_label
        average = average or self.config.average
        sample_weight = sample_weight or self.config.sample_weight
        zero_division = zero_division or self.config.zero_division
        n_decimals = n_decimals or self.config.n_decimals
        output_keys = output_keys or self.config.output_keys

        score = recall_score(
            targets,
            predictions,
            labels=labels,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )

        score = float(score) if score.size == 1 else score

        results = {"recall": round(float(score), n_decimals)}

        if output_keys:
            results = {k: v for k, v in results.items() if k in output_keys}

        return results
