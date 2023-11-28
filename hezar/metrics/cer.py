from dataclasses import dataclass

from ..configs import MetricConfig
from ..constants import Backends, MetricType
from ..registry import register_metric
from ..utils import is_backend_available
from .metric import Metric


if is_backend_available(Backends.JIWER):
    import jiwer
    import jiwer.transforms as tr

_DESCRIPTION = "Character Error Rate (CER) using `jiwer`. Commonly used for Speech Recognition and OCR systems"

_required_backends = [
    Backends.JIWER,
]


@dataclass
class CERConfig(MetricConfig):
    """
    Configuration class for CER (Character Error Rate) metric.

    Args:
        name (MetricType): The type of metric, CER in this case.
        sentence_delimiter (str): Delimiter for separating sentences in texts.
        concatenate_texts (bool): Flag to concatenate texts before computing CER.
        output_keys (tuple): Keys to filter the metric results for output.
    """
    name = MetricType.CER
    objective: str = "minimize"
    sentence_delimiter: str = " "
    concatenate_texts: bool = False
    output_keys: tuple = ("cer",)


@register_metric("cer", config_class=CERConfig, description=_DESCRIPTION)
class CER(Metric):
    """
    CER metric for evaluating Character Error Rate using `jiwer`.

    Args:
        config (CERConfig): Metric configuration object.
        **kwargs: Extra configuration parameters passed as kwargs to update the `config`.
    """
    required_backends = _required_backends

    def __init__(self, config: CERConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.transform = tr.Compose(
            [
                tr.RemoveMultipleSpaces(),
                tr.Strip(),
                tr.ReduceToSingleSentence(self.config.sentence_delimiter),
                tr.ReduceToListOfListOfChars(),
            ]
        )

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
        Computes the Character Error Rate (CER) for the given predictions against targets.

        Args:
            predictions: Predicted texts.
            targets: Ground truth texts.
            concatenate_texts (bool): Flag to concatenate texts before computing CER.
            n_decimals (int): Number of decimals for the final score.
            output_keys (tuple): Filter the output keys.

        Returns:
            dict: A dictionary of the metric results, with keys specified by `output_keys`.
        """
        concatenate_texts = concatenate_texts or self.config.concatenate_texts
        n_decimals = n_decimals or self.config.n_decimals

        if concatenate_texts:
            score = jiwer.process_words(
                targets,
                predictions,
                reference_transform=self.transform,
                hypothesis_transform=self.transform,
            ).wer

        else:
            incorrect = 0
            total = 0
            for prediction, reference in zip(predictions, targets):
                measures = jiwer.process_words(
                    reference,
                    prediction,
                    reference_transform=self.transform,
                    hypothesis_transform=self.transform,
                )
                incorrect += measures.substitutions + measures.deletions + measures.insertions
                total += measures.substitutions + measures.deletions + measures.hits

            score = incorrect / total

        results = {"cer": round(float(score), n_decimals)}

        if output_keys:
            results = {k: v for k, v in results.items() if k in output_keys}

        return results
