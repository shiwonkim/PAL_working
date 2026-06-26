from typing import Union

import numpy as np
import torch.nn.functional as F
from tqdm import trange

import src.utils.metrics as metrics


def compute_score(
    x_feats,
    y_feats,
    metric="mutual_knn",
    topk: Union[str, int] = 10,
    normalize=True,
    show_progress: bool = True,
    **metric_kwargs,
):
    """
    Use different layer combinations of x_feats and y_feats to find the best alignment.

    Args:
        x_feats: a torch tensor of shape N x L x D
        y_feats: a torch tensor of shape N x L x D
    Returns:
        best_alignment_score: the best alignment score
        best_alignment: the indices of the best alignment
    """
    assert x_feats.shape[0] == y_feats.shape[0]
    n_samples = x_feats.shape[0]
    if type(topk) is str:
        if topk == "sqrt":
            topk = int(np.ceil(np.sqrt(n_samples)))
        elif topk == "sturges":
            topk = int(np.ceil(np.log2(n_samples) + 1))
        elif topk == "rice":
            topk = int(np.ceil(2 * n_samples ** (1 / 3)))
        else:
            raise ValueError(topk)

    best_alignment_indices = None
    best_alignment_score = -1
    alignment_list = []

    # if we only have the last layer
    if len(x_feats.shape) == 2:
        x_feats = x_feats.unsqueeze(1)
    if len(y_feats.shape) == 2:
        y_feats = y_feats.unsqueeze(1)

    for i in trange(-1, x_feats.shape[1], disable=not show_progress):
        x = x_feats.flatten(1, 2) if i == -1 else x_feats[:, i, :]

        for j in range(-1, y_feats.shape[1]):
            y = y_feats.flatten(1, 2) if j == -1 else y_feats[:, j, :]

            kwargs = {} | metric_kwargs
            if "knn" in metric:
                kwargs["topk"] = topk

            if normalize:
                x = F.normalize(x, p=2, dim=-1)
                y = F.normalize(y, p=2, dim=-1)

            score = metrics.AlignmentMetrics.measure(metric, x, y, **kwargs)
            alignment_list.append({"indices": (i, j), "alignment_score": score})

            if score > best_alignment_score:
                best_alignment_score = score
                best_alignment_indices = (i, j)

    return best_alignment_score, best_alignment_indices, alignment_list
