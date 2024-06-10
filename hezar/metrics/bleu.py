from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..configs import MetricConfig
from ..constants import Backends, MetricType
from ..registry import register_metric
from ..utils import is_backend_available
from .metric import Metric


if is_backend_available(Backends.NLTK):
    from nltk.translate.bleu_score import corpus_bleu

_required_backends = [
    Backends.NLTK,
]


@dataclass
class BLEUConfig(MetricConfig):
    """
    Configuration class for BLEU metric.

    args:
        name (MetricType): The type of metric, BLEU in this case.
        output_keys (tuple): Keys to filter the metric results for output.
    """
    name = MetricType.BLEU
    objective: str = "maximize"
    output_keys: tuple = ("bleu",)


@register_metric("bleu", config_class=BLEUConfig)
class BLEU(Metric):
    """
    BLEU metric for evaluating text generation models like translation, summarization, etc.
    """
    required_backends = _required_backends

    def __init__(self, config: BLEUConfig, **kwargs):
        super().__init__(config=config, **kwargs)

    def compute(
        self,
        predictions: Iterable[str] | str = None,
        targets: Iterable[str] | str = None,
        weights=(0.25, 0.25, 0.25, 0.25),
        n_decimals=None,
        output_keys=None,
        **kwargs,
    ):
        """
        Computes the BLEU score for the given predictions against targets.

        Args:
            predictions (Iterable[str] | str): Predicted sentences or tokens.
            targets (Iterable[str] | str): Ground truth sentences or tokens.
            weights (tuple): Weights for n-gram precision, default is (0.25, 0.25, 0.25, 0.25).
            n_decimals (int): Number of decimals for the final score.
            output_keys (tuple): Filter the output keys.

        Returns:
            dict: A dictionary of the metric results, with keys specified by `output_keys`.
        """
        n_decimals = n_decimals or self.config.n_decimals
        output_keys = output_keys or self.config.output_keys

        predictions = [x.split() if isinstance(x, str) else x for x in predictions]
        targets = [x.split() if isinstance(x, str) else x for x in targets]

        score = corpus_bleu(targets, predictions, weights=weights)

        results = {"bleu": round(float(score), n_decimals)}

        if output_keys:
            results = {k: v for k, v in results.items() if k in output_keys}

        return results
