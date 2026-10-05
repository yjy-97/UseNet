import numpy as np
import torch


def collate_fn(data, max_len=None):
    batch_size = len(data)
    features, labels = zip(*data)

    lengths = [X.shape[0] for X in features]
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(batch_size, max_len, features[0].shape[-1])
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)

    padding_masks = padding_mask(
        torch.tensor(lengths, dtype=torch.int16), max_len=max_len
    )

    return X, targets, padding_masks


def padding_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()
    return (
        torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)
        .repeat(batch_size, 1)
        .lt(lengths.unsqueeze(1))
    )


def normalize_batch_ts(batch):
    mean_values = batch.mean(axis=1, keepdims=True)
    std_values = batch.std(axis=1, keepdims=True)
    std_values[std_values == 0] = 1.0
    normalized_batch = (batch - mean_values) / std_values
    return normalized_batch

