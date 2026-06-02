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
    output_keys: tuple = (
        "rouge1",
        "rouge2",
        "rougeL",
        "rougeLsum",
    )


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
        predictions: list[str],
        targets: list[str],
        use_aggregator: bool | None = None,
        n_decimals: int | None = None,
        output_keys: tuple | None = None,
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
        use_aggregator = use_aggregator if use_aggregator is not None else self.config.use_aggregator
        n_decimals = n_decimals if n_decimals is not None else self.config.n_decimals
        output_keys = output_keys or self.config.output_keys

        score_fn = self.scorer.score_multi if self.config.multi_ref else self.scorer.score

        if use_aggregator:
            aggregator = scoring.BootstrapAggregator()
            for ref, pred in zip(targets, predictions, strict=True):
                aggregator.add_scores(score_fn(ref, pred))
            agg = aggregator.aggregate()
            results = {k: agg[k].mid.fmeasure for k in agg}
        else:
            sums = {}
            n = 0
            for ref, pred in zip(targets, predictions, strict=True):
                scores = score_fn(ref, pred)
                for k, v in scores.items():
                    sums[k] = sums.get(k, 0.0) + v.fmeasure
                n += 1
            results = {k: (v / n if n else 0.0) for k, v in sums.items()}

        results = {k: round(v, n_decimals) for k, v in results.items() if (not output_keys or k in output_keys)}

        return results
