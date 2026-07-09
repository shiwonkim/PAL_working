from typing import Dict, List, Optional

import pandas as pd
import torch

from src.utils.utils import safe_normalize


def compute_ground_truth_mapping(df, image_column="image_name"):
    """
    Constructs a mapping from each row index to the set of indices that share the same image name.

    Args:
        df (pd.DataFrame): DataFrame containing at least the image_column.
        image_column (str): Column name to group on.

    Returns:
        dict: A mapping where each key is a row index and the value is a set of indices sharing the same image name.
    """
    # Group by the image column; groups is a dict: image_name -> pd.Index of rows.
    groups = df.groupby(image_column).groups
    ground_truth = {}
    for idx in df.index:
        image_name = df.loc[idx, image_column]
        ground_truth[idx] = set(groups[image_name])
    return ground_truth


def retrieval_metrics_df(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    df: Optional[pd.DataFrame] = None,
    image_column: str = "image_name",
    k_values: List[int] = [1, 5, 10],
    batch_size: int = 64,
) -> Dict[str, float]:
    """Recall@k, Precision@k and MAP@k for image→text retrieval."""
    image_embeds = safe_normalize(image_embeds, p=2, dim=1)
    text_embeds = safe_normalize(text_embeds, p=2, dim=1)

    # build ground-truth mapping:  idx -> set(idx, …)
    N = image_embeds.size(0)
    if df is not None:
        ground_truth = compute_ground_truth_mapping(df, image_column)
    else:
        ground_truth = {i: {i} for i in range(N)}

    # every query must have ≥1 relevant item
    assert all(
        len(v) for v in ground_truth.values()
    ), "Some queries have no ground truth!"

    max_k = max(k_values)
    recall_hits = {k: 0.0 for k in k_values}
    precision_sum = {k: 0.0 for k in k_values}
    map_sum = {k: 0.0 for k in k_values}

    # batched similarity and metrics
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sims = image_embeds[start:end] @ text_embeds.T  # [B, N]
        _, topk = torch.topk(sims, k=max_k, dim=1)  # indices

        for row, ranked_tensor in enumerate(topk):
            q_idx = start + row
            gt_set = ground_truth[q_idx]
            ranked = ranked_tensor.tolist()
            rel_flags = [idx in gt_set for idx in ranked]  # bool list

            for k in k_values:
                rel_in_k = sum(rel_flags[:k])

                # Recall@k (hit-rate)
                if rel_in_k > 0:
                    recall_hits[k] += 1

                # Precision@k
                precision_sum[k] += rel_in_k / k

                # AP@k (average precision truncated at k)
                hits, ap = 0, 0.0
                for r, is_rel in enumerate(rel_flags[:k], start=1):
                    if is_rel:
                        hits += 1
                        ap += hits / r
                ap /= min(len(gt_set), k)  # denom cap at k
                map_sum[k] += ap

    # macro-averages
    metrics = {}
    for k in k_values:
        metrics[f"R@{k}"] = recall_hits[k] / N
        metrics[f"P@{k}"] = precision_sum[k] / N
        metrics[f"MAP@{k}"] = map_sum[k] / N

    return metrics


def text_to_image_retrieval_metrics(
    text_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    df: pd.DataFrame,
    image_column: str = "image_name",
    k_values: List[int] = [1, 5, 10],
    batch_size: int = 64,
) -> Dict[str, float]:
    """Recall@k / Precision@k / MAP@k for text→image retrieval.

    The row-wise eval features duplicate each image once per caption, so naively
    swapping the image/text args into ``retrieval_metrics_df`` ranks over those
    duplicates: identical copies of the true image fill consecutive ranks, which
    collapses R@1 onto R@5. Here the gallery is the set of UNIQUE images (one
    embedding per ``image_column`` value) and every caption query has exactly one
    relevant image — the standard text→image protocol.
    """
    text_embeds = safe_normalize(text_embeds, p=2, dim=1)
    image_embeds = safe_normalize(image_embeds, p=2, dim=1)

    # Deduplicate the gallery to unique images. Rows sharing an image carry the
    # same image embedding, so the first occurrence of each image suffices.
    dfr = df.reset_index(drop=True)
    first_rows = dfr.drop_duplicates(subset=image_column).index.tolist()
    gallery = image_embeds[first_rows]  # (M, D) unique images
    path_to_gid = {dfr.loc[ri, image_column]: gi for gi, ri in enumerate(first_rows)}
    # Each caption query maps to the single gallery id of its image.
    query_gid = [path_to_gid[dfr.loc[i, image_column]] for i in range(len(dfr))]

    N = text_embeds.size(0)
    M = gallery.size(0)
    max_k = min(max(k_values), M)
    recall_hits = {k: 0.0 for k in k_values}
    precision_sum = {k: 0.0 for k in k_values}
    map_sum = {k: 0.0 for k in k_values}

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sims = text_embeds[start:end] @ gallery.T  # [B, M]
        _, topk = torch.topk(sims, k=max_k, dim=1)
        for row, ranked in enumerate(topk.tolist()):
            gt = query_gid[start + row]  # exactly one relevant image
            rel_flags = [g == gt for g in ranked]
            for k in k_values:
                rel_in_k = sum(rel_flags[:k])
                if rel_in_k > 0:
                    recall_hits[k] += 1
                precision_sum[k] += rel_in_k / k
                hits, ap = 0, 0.0
                for r, is_rel in enumerate(rel_flags[:k], start=1):
                    if is_rel:
                        hits += 1
                        ap += hits / r
                map_sum[k] += ap  # denominator is 1 (single relevant image)

    metrics = {}
    for k in k_values:
        metrics[f"R@{k}"] = recall_hits[k] / N
        metrics[f"P@{k}"] = precision_sum[k] / N
        metrics[f"MAP@{k}"] = map_sum[k] / N

    return metrics
