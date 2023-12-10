import os

import pytest

from hezar.builders import build_metric
from hezar.utils import list_available_metrics


INVALID_OUTPUT_TYPE = "Metric output must be a dictionary!"
INVALID_FIELDS = "Invalid fields in metric outputs!"


@pytest.mark.skipif("accuracy" not in list_available_metrics(), reason="`accuracy` is not available!")
def test_accuracy():
    metric = build_metric("accuracy")
    predictions = [0, 2, 1, 3]
    references = [0, 1, 2, 3]
    results = metric.compute(predictions, references)
    assert isinstance(results, dict), INVALID_OUTPUT_TYPE
    assert set(results.keys()) == set(metric.config.output_keys), INVALID_FIELDS


@pytest.mark.skipif("bleu" not in list_available_metrics(), reason="`bleu` is not available!")
def test_bleu():
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

    assert isinstance(results, dict), INVALID_OUTPUT_TYPE
    assert set(results.keys()) == set(metric.config.output_keys), INVALID_FIELDS


@pytest.mark.skipif("cer" not in list_available_metrics(), reason="`cer` is not available!")
def test_cer():
    metric = build_metric("cer")
    predictions = ["بهتره این یکی رو بیخیال شیم"]
    references = ["بهتره این یکی رو بیخیال بشیم"]
    results = metric.compute(predictions, references)
    assert isinstance(results, dict), INVALID_OUTPUT_TYPE
    assert set(results.keys()) == set(metric.config.output_keys), INVALID_FIELDS


@pytest.mark.skipif("f1" not in list_available_metrics(), reason="`f1` is not available!")
def test_f1():
    metric = build_metric("f1")
    predictions = [0, 1, 2, 0, 1, 2]
    references = [0, 2, 1, 0, 0, 1]
    results = metric.compute(predictions, references)
    assert isinstance(results, dict), INVALID_OUTPUT_TYPE
    assert set(results.keys()) == set(metric.config.output_keys), INVALID_FIELDS


@pytest.mark.skipif("precision" not in list_available_metrics(), reason="`precision` is not available!")
def test_precision():
    metric = build_metric("precision")
    predictions = [0, 1, 2, 0, 1, 2]
    references = [0, 2, 1, 0, 0, 1]
    results = metric.compute(predictions, references)
    assert isinstance(results, dict), INVALID_OUTPUT_TYPE
    assert set(results.keys()) == set(metric.config.output_keys), INVALID_FIELDS


@pytest.mark.skipif("recall" not in list_available_metrics(), reason="`recall` is not available!")
def test_recall():
    metric = build_metric("recall")
    predictions = [0, 1, 2, 0, 1, 2]
    references = [0, 2, 1, 0, 0, 1]
    results = metric.compute(predictions, references)
    assert isinstance(results, dict), INVALID_OUTPUT_TYPE
    assert set(results.keys()) == set(metric.config.output_keys), INVALID_FIELDS


@pytest.mark.skipif("rouge" not in list_available_metrics(), reason="`rouge` is not available!")
def test_rouge():
    metric = build_metric("rouge")
    predictions = ["از این معیار برای سنجش خروجی مدل های ترجمه یا خلاصه سازی استفاده می شود"]
    references = ["در واقع از این معیار برای سنجش خروجی مدل های ترجمه یا خلاصه سازی استفاده می شود"]
    results = metric.compute(predictions, references)
    assert isinstance(results, dict), INVALID_OUTPUT_TYPE
    assert set(results.keys()) == set(metric.config.output_keys), INVALID_FIELDS


@pytest.mark.skipif("seqeval" not in list_available_metrics(), reason="`seqeval` is not available!")
def test_seqeval():
    metric = build_metric("seqeval")
    predictions = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    references = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    results = metric.compute(predictions, references)
    assert isinstance(results, dict), INVALID_OUTPUT_TYPE
    assert set(results.keys()) == set(metric.config.output_keys), INVALID_FIELDS


@pytest.mark.skipif("wer" not in list_available_metrics(), reason="`wer` is not available!")
def test_wer():
    metric = build_metric("wer")
    predictions = ["بهتره این یکی رو بیخیال شیم"]
    references = ["بهتره این یکی رو بیخیال بشیم"]
    results = metric.compute(predictions, references)
    assert isinstance(results, dict), INVALID_OUTPUT_TYPE
    assert set(results.keys()) == set(metric.config.output_keys), INVALID_FIELDS
