import numpy as np
import torch


def _reconstruct(labels, blank=0):
    new_labels = []
    # merge same labels
    previous = None
    for label in labels:
        if label != previous:
            new_labels.append(label)
            previous = label
    # delete blank
    new_labels = [label for label in new_labels if label != blank]

    return new_labels


def greedy_decode(emission_log_prob, blank=0):
    labels = np.argmax(emission_log_prob, axis=-1)
    labels = _reconstruct(labels, blank=blank)
    return labels


def ctc_decode(log_probs, id2label=None, blank=0):
    emission_log_probs = np.transpose(log_probs.cpu().detach().numpy(), (1, 0, 2))
    batch_size, max_length, _ = emission_log_probs.shape

    # size of emission_log_probs: (batch, length, class)
    decoded_ids = []
    for emission_log_prob in emission_log_probs:
        ids = greedy_decode(emission_log_prob, blank=blank)
        if id2label:
            ids = [id2label[label] for label in ids]
        ids.extend([blank] * (max_length - len(ids)))
        decoded_ids.append(ids)
    return torch.tensor(decoded_ids)
