from dataclasses import dataclass

from ..configs import MetricConfig
from ..constants import Backends, MetricType
from ..registry import register_metric
from ..utils import is_backend_available
from .metric import Metric


if is_backend_available(Backends.JIWER):
    import jiwer

_DESCRIPTION = "Word Error Rate (WER) using `jiwer`. Commonly used for Speech Recognition systems"

_required_backends = [
    Backends.JIWER,
]


@dataclass
class WERConfig(MetricConfig):
    name = MetricType.WER
    concatenate_texts: bool = False
    output_keys: tuple = ("wer",)


@register_metric("wer", config_class=WERConfig, description=_DESCRIPTION)
class WER(Metric):
    required_backends = _required_backends

    def __init__(self, config: WERConfig, **kwargs):
        super().__init__(config=config, **kwargs)

    def compute(
        self,
        predictions=None,
        targets=None,
        concatenate_texts=None,
        n_decimals=None,
        output_keys=None,
        **kwargs,
    ):
        concatenate_texts = concatenate_texts or self.config.concatenate_texts
        n_decimals = n_decimals or self.config.n_decimals

        if concatenate_texts:
            score = jiwer.compute_measures(targets, predictions)["wer"]
        else:
            incorrect = 0
            total = 0
            for prediction, reference in zip(predictions, targets):
                measures = jiwer.compute_measures(reference, prediction)
                incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
                total += measures["substitutions"] + measures["deletions"] + measures["hits"]

            score = incorrect / total

        results = {"wer": round(float(score), n_decimals)}

        if output_keys:
            results = {k: v for k, v in results.items() if k in output_keys}

        return results
