from dataclasses import dataclass
from typing import Iterable, Union

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
    name = MetricType.BLEU
    output_keys: tuple = ("bleu",)


@register_metric("bleu", config_class=BLEUConfig)
class BLEU(Metric):
    required_backends = _required_backends

    def __init__(self, config: BLEUConfig, **kwargs):
        super().__init__(config=config, **kwargs)

    def compute(
        self,
        predictions: Union[Iterable[str], str] = None,
        targets: Union[Iterable[str], str] = None,
        weights=(.25, .25, .25, .25),
        n_decimals=None,
        output_keys=None,
        **kwargs,
    ):
        n_decimals = n_decimals or self.config.n_decimals
        output_keys = output_keys or self.config.output_keys

        predictions = [x.split() if isinstance(x, str) else x for x in predictions]
        targets = [x.split() if isinstance(x, str) else x for x in targets]

        score = corpus_bleu(targets, predictions, weights=weights)

        results = {"bleu": round(float(score), n_decimals)}

        if output_keys:
            results = {k: v for k, v in results.items() if k in output_keys}

        return results
