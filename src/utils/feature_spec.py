"""FeatureSpec — the single CLS-vs-token policy for one modality at one stage.

The CLS/token axis is one decision (``token_level``) that fans out into a pool
mode, an encoder layer to slice, and whether the text side carries a padding
mask. Today every stage re-derives it from raw config: extraction reads
``features.token_level``-implied pooling, the train loop branches on mask
presence, zero-shot ANDs ``evaluation.token_level_zero_shot`` with
``training.token_level``, retrieval uses ``training.token_level`` alone. The
rules are subtly different per stage and duplicated, which is how a stage ends
up on the wrong feature distribution (the classic "forgot token_level_zero_shot
→ random-looking scores" bug).

``FeatureSpec`` makes that decision once, named per stage, so consumers read
typed fields instead of re-deriving the branch. Behavior is preserved: each
``for_*`` classmethod reproduces exactly the rule that stage used before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_POOL_KEY = {"image": "pool_img", "text": "pool_txt"}
_LAYER_KEY = {"image": "layer_img", "text": "layer_txt"}


@dataclass(frozen=True)
class FeatureSpec:
    """How features for one modality should be produced/consumed at one stage."""

    modality: str                 # "image" | "text"
    token_level: bool             # token sequence (B,T,D) vs pooled (B,D)
    pool: Optional[str]           # "none" when token_level, else the config pool
    layer_index: Optional[int]    # encoder layer to slice (token mode)
    img_size: Optional[int]       # image-only resolution (drives the cache tag)
    needs_mask: bool              # token_level and modality == "text"

    @classmethod
    def _build(
        cls, config: dict, modality: str, token_level: bool,
        layer_index: Optional[int] = None,
    ) -> "FeatureSpec":
        if modality not in _POOL_KEY:
            raise ValueError(f"unknown modality {modality!r}")
        feats = config["features"]
        pool = "none" if token_level else feats.get(_POOL_KEY[modality])
        if layer_index is None:
            layer_index = feats.get(_LAYER_KEY[modality])
        return cls(
            modality=modality,
            token_level=token_level,
            pool=pool,
            layer_index=layer_index,
            img_size=feats.get("img_size") if modality == "image" else None,
            needs_mask=token_level and modality == "text",
        )

    @classmethod
    def for_training(
        cls, config: dict, modality: str, layer_index: Optional[int] = None,
    ) -> "FeatureSpec":
        token_level = bool(config["training"].get("token_level", False))
        return cls._build(config, modality, token_level, layer_index)

    @classmethod
    def for_retrieval(
        cls, config: dict, modality: str, layer_index: Optional[int] = None,
    ) -> "FeatureSpec":
        # Retrieval follows the training mode directly.
        token_level = bool(config["training"].get("token_level", False))
        return cls._build(config, modality, token_level, layer_index)

    @classmethod
    def for_zero_shot(
        cls, config: dict, modality: str, layer_index: Optional[int] = None,
    ) -> "FeatureSpec":
        # Zero-shot is token-level only when explicitly opted in AND the layers
        # were trained token-level — token templates on a CLS-trained head (or
        # vice-versa) give the wrong distribution.
        token_level = bool(
            config["evaluation"].get("token_level_zero_shot", False)
        ) and bool(config["training"].get("token_level", False))
        return cls._build(config, modality, token_level, layer_index)
