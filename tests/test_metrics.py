from unittest import TestCase

from hezar.builders import build_metric


class MetricsTestCase(TestCase):
    def test_accuracy(self):
        metric = build_metric("accuracy")
        predictions = []
        references = [[]]
        results = metric.compute(predictions, references)


    def test_bleu(self):
        metric = build_metric("bleu")
        predictions = []
        references = [[]]
        results = metric.compute(predictions, references)

    def test_cer(self):
        metric = build_metric("cer")
        predictions = []
        references = [[]]
        results = metric.compute(predictions, references)

    def test_f1(self):
        metric = build_metric("f1")
        predictions = []
        references = [[]]
        results = metric.compute(predictions, references)

    def test_precision(self):
        metric = build_metric("precision")
        predictions = []
        references = [[]]
        results = metric.compute(predictions, references)

    def test_recall(self):
        metric = build_metric("recall")
        predictions = []
        references = [[]]
        results = metric.compute(predictions, references)

    def test_rouge(self):
        metric = build_metric("rouge")
        predictions = []
        references = [[]]
        results = metric.compute(predictions, references)

    def test_seqeval(self):
        metric = build_metric("seqeval")
        predictions = []
        references = [[]]
        results = metric.compute(predictions, references)

    def test_wer(self):
        metric = build_metric("wer")
        predictions = []
        references = [[]]
        results = metric.compute(predictions, references)
