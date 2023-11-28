from dataclasses import dataclass

from ..configs import MetricConfig
from ..constants import Backends, MetricType
from ..registry import register_metric
from ..utils import is_backend_available
from .metric import Metric


if is_backend_available(Backends.ROUGE):
    from rouge_score import rouge_scorer, scoring

_DESCRIPTION = "Rouge estimation. Commonly used for Text Summarization"

_required_backends = [
    Backends.ROUGE,
]


@dataclass
class ROUGEConfig(MetricConfig):
    """
    Configuration class for ROUGE metric.

    Args:
        name (MetricType): The type of metric, ROUGE in this case.
        use_stemmer (bool): Flag to enable stemming when computing ROUGE.
        use_aggregator (bool): Flag to enable score aggregation for multiple references.
        multi_ref (bool): Flag to indicate if multiple references are present.
        output_keys (tuple): Keys to filter the metric results for output.
    """
    name = MetricType.ROUGE
    objective: str = "maximize"
    use_stemmer: bool = False
    use_aggregator: bool = True
    multi_ref: bool = True
    output_keys: tuple = ("rouge1", "rouge2", "rougeL", "rougeLsum",)


@register_metric("rouge", config_class=ROUGEConfig, description=_DESCRIPTION)
class ROUGE(Metric):
    """
    ROUGE metric for evaluating text summarization using `rouge_score`.

    Args:
        config (ROUGEConfig): Metric configuration object.
        **kwargs: Extra configuration parameters passed as kwargs to update the `config`.
    """
    required_backends = _required_backends

    def __init__(self, config: ROUGEConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        self.scorer = rouge_scorer.RougeScorer(
            rouge_types=rouge_types,
            use_stemmer=self.config.use_stemmer,
        )

    def compute(
        self,
        predictions=None,
        targets=None,
        use_aggregator=None,
        n_decimals=None,
        output_keys=None,
        **kwargs,
    ):
        """
        Computes the ROUGE scores for the given predictions against targets.

        Args:
            predictions: Predicted summaries.
            targets: Ground truth summaries.
            use_aggregator (bool): Flag to enable score aggregation for multiple references.
            n_decimals (int): Number of decimals for the final score.
            output_keys (tuple): Filter the output keys.

        Returns:
            dict: A dictionary of the metric results, with keys specified by `output_keys`.
        """
        aggregator = scoring.BootstrapAggregator()

        for ref, pred in zip(targets, predictions):
            if self.config.multi_ref:
                score = self.scorer.score_multi(ref, pred)
            else:
                score = self.scorer.score(ref, pred)

            aggregator.add_scores(score)

        results = aggregator.aggregate()
        for key in results:
            results[key] = results[key].mid.fmeasure

        if output_keys:
            results = {k: round(v, 4) for k, v in results.items() if k in output_keys}

        return results
