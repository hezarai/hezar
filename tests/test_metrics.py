from unittest import TestCase, skipIf

from hezar.utils import list_available_metrics
from hezar.builders import build_metric

INVALID_OUTPUT_TYPE = "Metric output must be a dictionary!"
INVALID_FIELDS = "Invalid fields in metric outputs!"


class MetricsTestCase(TestCase):
    @skipIf("accuracy" not in list_available_metrics(), "`accuracy` is not available!")
    def test_accuracy(self):
        metric = build_metric("accuracy")
        predictions = [0, 2, 1, 3]
        references = [0, 1, 2, 3]
        results = metric.compute(predictions, references)
        self.assertEqual(type(results), dict, INVALID_OUTPUT_TYPE)
        self.assertEqual(set(results.keys()), set(metric.config.output_keys), INVALID_FIELDS)

    @skipIf("bleu" not in list_available_metrics(), "`bleu` is not available!")
    def test_bleu(self):
        metric = build_metric("bleu")
        pred_1 = ["It", "is", "a", "guide", "to", "action", "which", "ensures", "that", "the", "military", "always",
                  "obeys", "the", "commands", "of", "the", "party"]
        ref_1a = ["It", "is", "a", "guide", "to", "action", "that", "ensures", "that", "the", "military", "will",
                  "forever", "heed", "Party", "commands"]
        ref_1b = ["It", "is", "the", "guiding", "principle", "which", "guarantees", "the", "military", "forces",
                  "always", "being", "under", "the", "command", "of", "the", "Party"]
        ref_1c = ["It", "is", "the", "practical", "guide", "for", "the", "army", "always", "to", "heed", "the",
                  "directions", "of", "the", "party"]
        pred_2 = ["he", "read", "the", "book", "because", "he", "was", "interested", "in", "world", "history"]
        ref_2a = ["he", "was", "interested", "in", "world", "history", "because", "he", "read", "the", "book"]

        references = [[ref_1a, ref_1b, ref_1c], [ref_2a]]
        predictions = [pred_1, pred_2]

        results = metric.compute(predictions, references)

        self.assertEqual(type(results), dict, INVALID_OUTPUT_TYPE)
        self.assertEqual(set(results.keys()), set(metric.config.output_keys), INVALID_FIELDS)

    @skipIf("cer" not in list_available_metrics(), "`cer` is not available!")
    def test_cer(self):
        metric = build_metric("cer")
        predictions = ["بهتره این یکی رو بیخیال شیم"]
        references = ["بهتره این یکی رو بیخیال بشیم"]
        results = metric.compute(predictions, references)
        self.assertEqual(type(results), dict, INVALID_OUTPUT_TYPE)
        self.assertEqual(set(results.keys()), set(metric.config.output_keys), INVALID_FIELDS)

    @skipIf("f1" not in list_available_metrics(), "`f1` is not available!")
    def test_f1(self):
        metric = build_metric("f1")
        predictions = [0, 1, 2, 0, 1, 2]
        references = [0, 2, 1, 0, 0, 1]
        results = metric.compute(predictions, references)
        self.assertEqual(type(results), dict, INVALID_OUTPUT_TYPE)
        self.assertEqual(set(results.keys()), set(metric.config.output_keys), INVALID_FIELDS)

    @skipIf("precision" not in list_available_metrics(), "`precision` is not available!")
    def test_precision(self):
        metric = build_metric("precision")
        predictions = [0, 1, 2, 0, 1, 2]
        references = [0, 2, 1, 0, 0, 1]
        results = metric.compute(predictions, references)
        self.assertEqual(type(results), dict, INVALID_OUTPUT_TYPE)
        self.assertEqual(set(results.keys()), set(metric.config.output_keys), INVALID_FIELDS)

    @skipIf("recall" not in list_available_metrics(), "`recall` is not available!")
    def test_recall(self):
        metric = build_metric("recall")
        predictions = [0, 1, 2, 0, 1, 2]
        references = [0, 2, 1, 0, 0, 1]
        results = metric.compute(predictions, references)
        self.assertEqual(type(results), dict, INVALID_OUTPUT_TYPE)
        self.assertEqual(set(results.keys()), set(metric.config.output_keys), INVALID_FIELDS)

    @skipIf("rouge" not in list_available_metrics(), "`rouge` is not available!")
    def test_rouge(self):
        metric = build_metric("rouge")
        predictions = ["از این معیار برای سنجش خروجی مدل های ترجمه یا خلاصه سازی استفاده می شود"]
        references = ["در واقع از این معیار برای سنجش خروجی مدل های ترجمه یا خلاصه سازی استفاده می شود"]
        results = metric.compute(predictions, references)
        self.assertEqual(type(results), dict, INVALID_OUTPUT_TYPE)
        self.assertEqual(set(results.keys()), set(metric.config.output_keys), INVALID_FIELDS)

    @skipIf("seqeval" not in list_available_metrics(), "`seqeval` is not available!")
    def test_seqeval(self):
        metric = build_metric("seqeval")
        predictions = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        references = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        results = metric.compute(predictions, references)
        self.assertEqual(type(results), dict, INVALID_OUTPUT_TYPE)
        self.assertEqual(set(results.keys()), set(metric.config.output_keys), INVALID_FIELDS)

    @skipIf("wer" not in list_available_metrics(), "`wer` is not available!")
    def test_wer(self):
        metric = build_metric("wer")
        predictions = ["بهتره این یکی رو بیخیال شیم"]
        references = ["بهتره این یکی رو بیخیال بشیم"]
        results = metric.compute(predictions, references)
        self.assertEqual(type(results), dict, INVALID_OUTPUT_TYPE)
        self.assertEqual(set(results.keys()), set(metric.config.output_keys), INVALID_FIELDS)
