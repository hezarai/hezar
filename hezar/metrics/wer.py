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
    """
    Configuration class for WER metric.

    Args:
        name (MetricType): The type of metric, WER in this case.
        concatenate_texts (bool): Flag to indicate whether to concatenate texts before WER calculation.
        output_keys (tuple): Keys to filter the metric results for output.
    """
    name = MetricType.WER
    objective: str = "minimize"
    concatenate_texts: bool = False
    output_keys: tuple = ("wer",)


@register_metric("wer", config_class=WERConfig, description=_DESCRIPTION)
class WER(Metric):
    """
    WER metric for evaluating Word Error Rate using `jiwer`.

    Args:
        config (WERConfig): Metric configuration object.
        **kwargs: Extra configuration parameters passed as kwargs to update the `config`.
    """
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
        """
        Computes the WER for the given predictions against targets.

        Args:
            predictions: Predicted texts.
            targets: Ground truth texts.
            concatenate_texts (bool): Flag to indicate whether to concatenate texts before WER calculation.
            n_decimals (int): Number of decimals for the final score.
            output_keys (tuple): Filter the output keys.

        Returns:
            dict: A dictionary of the metric results, with keys specified by `output_keys`.
        """
        concatenate_texts = concatenate_texts or self.config.concatenate_texts
        n_decimals = n_decimals or self.config.n_decimals

        if concatenate_texts:
            score = jiwer.process_words(targets, predictions).wer
        else:
            incorrect = 0
            total = 0
            for prediction, reference in zip(predictions, targets):
                measures = jiwer.process_words(reference, prediction)
                incorrect += measures.substitutions + measures.deletions + measures.insertions
                total += measures.substitutions + measures.deletions + measures.hits

            score = incorrect / total

        results = {"wer": round(float(score), n_decimals)}

        if output_keys:
            results = {k: v for k, v in results.items() if k in output_keys}

        return results
