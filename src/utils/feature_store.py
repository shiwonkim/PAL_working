"""FeatureStore — the single home for feature-cache path building and I/O.

Goal 3 of the refactor pulls feature extraction / caching / mmap-loading out of
``AlignmentTrainer`` so that extraction, training, and evaluation become separable
stages over a shared cache (and so the LAION / continual-learning memory work has
a clean home). It consumes a :class:`~src.utils.feature_spec.FeatureSpec` for the
CLS-vs-token policy.

Built incrementally; this first slice owns only the cache *path* contract (model
name sanitisation + ``{model}-{dataset}-{suffix}.npy`` layout). The suffix string
is still supplied by callers — unifying suffix construction through FeatureSpec is
a later step, because the existing suffixes have stage/modality-specific quirks
that must be reproduced byte-for-byte to keep the ~795 GB of cached features valid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


class FeatureStore:
    """Feature-cache path/IO. Cache layout: ``<save_path>/features/<file>.npy``."""

    @staticmethod
    def model_name(m_name: str) -> str:
        """Sanitise an encoder name into the cache filename stem."""
        return m_name.replace("/", "_").replace("-", "_")

    @staticmethod
    def cache_path(
        m_name: str,
        d_name: str,
        save_path: Union[str, Path],
        suffix: str = "",
    ) -> Path:
        """Path of the cache file for ``(model, dataset, suffix)``."""
        stem = f"{FeatureStore.model_name(m_name)}-{d_name}-{suffix}.npy"
        return Path(save_path) / "features" / stem
