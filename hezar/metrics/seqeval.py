from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..configs import MetricConfig
from ..constants import Backends, MetricType
from ..registry import register_metric
from ..utils import Logger, is_backend_available
from .metric import Metric


if is_backend_available(Backends.SEQEVAL):
    from seqeval.metrics import accuracy_score, classification_report

_required_backends = [
    Backends.SEQEVAL,
]

logger = Logger(__name__)


@dataclass
class SeqevalConfig(MetricConfig):
    """
    Configuration class for Seqeval metric.

    Args:
        name (MetricType): The type of metric, Seqeval in this case.
        output_keys (tuple): Keys to filter the metric results for output.
        suffix (bool): Flag to indicate whether the labels have suffixes.
        mode (Optional[str]): Evaluation mode for seqeval.
        sample_weight (Optional[List[int]]): Sample weights for the seqeval metrics.
        zero_division (str | int): Strategy for zero-division, default is 0.
    """
    name = MetricType.SEQEVAL
    objective: str = "maximize"
    output_keys: tuple = ("accuracy", "recall", "precision", "f1")
    suffix: bool = False
    mode: Optional[str] = None
    sample_weight: Optional[List[int]] = None
    zero_division: str | int = 0


@register_metric("seqeval", config_class=SeqevalConfig)
class Seqeval(Metric):
    """
    Seqeval metric for sequence labeling tasks using `seqeval`.

    Args:
        config (SeqevalConfig): Metric configuration object.
        **kwargs: Extra configuration parameters passed as kwargs to update the `config`.
    """
    required_backends = _required_backends

    def __init__(self, config: SeqevalConfig, **kwargs):
        super().__init__(config, **kwargs)

    def compute(
        self,
        predictions=None,
        targets=None,
        suffix: bool = None,
        mode: Optional[str] = None,
        sample_weight: Optional[List[int]] = None,
        zero_division: str | int = None,
        n_decimals: int = None,
        output_keys=None,
        **kwargs,
    ):
        """
        Computes the Seqeval scores for the given predictions against targets.

        Args:
            predictions: Predicted labels.
            targets: Ground truth labels.
            suffix (bool): Flag to indicate whether the labels have suffixes.
            mode (Optional[str]): Evaluation mode for seqeval.
            sample_weight (Optional[List[int]]): Sample weights for the seqeval metrics.
            zero_division (str | int): Strategy for zero-division, default is 0.
            n_decimals (int): Number of decimals for the final score.
            output_keys (tuple): Filter the output keys.

        Returns:
            dict: A dictionary of the metric results, with keys specified by `output_keys`.
        """
        suffix = suffix or self.config.suffix
        mode = mode or self.config.mode
        sample_weight = sample_weight or self.config.sample_weight
        zero_division = zero_division or self.config.zero_division
        n_decimals = n_decimals or self.config.n_decimals
        output_keys = output_keys or self.config.output_keys

        report = classification_report(
            y_true=targets,
            y_pred=predictions,
            suffix=suffix,
            output_dict=True,
            mode=mode,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        report.pop("macro avg")
        report.pop("weighted avg")
        overall_score = report.pop("micro avg")

        results = {
            "accuracy": format(accuracy_score(predictions, targets)),
            "f1": overall_score["f1-score"],
            "recall": overall_score["recall"],
            "precision": overall_score["precision"],
        }

        results = {k: round(float(v), n_decimals) for k, v in results.items()}

        if output_keys:
            results = {k: v for k, v in results.items() if k in output_keys}

        return results
