"""(De)serialization of alignment layers for state_dict-based checkpoints.

Alignment checkpoints store a self-describing dict (not a pickled ``nn.Module``),
so the module is rebuilt through ``AlignmentFactory`` and loaded via
``load_state_dict`` — robust to the layer class being renamed or moved::

    {
        "format":     "alignment_state_dict_v1",
        "class_name": "PALAlignmentLayer",
        "input_dim":  1024,
        "kwargs":     {... alignment_layer_kwargs ...},
        "modality":   "image" | "text" | None,
        "state_dict": OrderedDict(...),
    }

Legacy pickled-module checkpoints (pre-refactor) are NOT supported here — load
those with the original pre-refactor code. The one-off migration tooling and the
BridgeAnchor->PAL class-name aliasing it needed have been removed now that every
checkpoint in use is new-format with the current PAL names; see git history if
an old pickled checkpoint ever needs converting again.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from src.models.alignment.alignment_factory import AlignmentFactory

ALIGNMENT_FORMAT = "alignment_state_dict_v1"


def serialize_alignment_layer(
    module: nn.Module,
    *,
    class_name: str,
    input_dim: int,
    kwargs: Mapping[str, Any],
    modality: Optional[str] = None,
) -> dict:
    """Build the new-format checkpoint entry for one alignment layer.

    ``modality`` is only recorded for layers that actually use it
    (``set_modality``); for the rest it is stored as ``None``.
    """
    return {
        "format": ALIGNMENT_FORMAT,
        "class_name": class_name,
        "input_dim": int(input_dim),
        "kwargs": dict(kwargs),
        "modality": modality if hasattr(module, "set_modality") else None,
        "state_dict": module.state_dict(),
    }


def is_new_format(entry: Any) -> bool:
    """True if ``entry`` is a new-format alignment dict (vs a legacy module)."""
    return isinstance(entry, dict) and entry.get("format") == ALIGNMENT_FORMAT


def load_alignment_layer(
    entry: Any,
    modality: str,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Rebuild an alignment layer from a new-format checkpoint entry.

    ``modality`` ("image"/"text") is applied via ``set_modality`` when the layer
    supports it, matching how the trainer builds layers. Legacy pickled-module
    checkpoints are not supported — load them with the pre-refactor code.
    """
    if not is_new_format(entry):
        raise ValueError(
            "Not a new-format alignment checkpoint (expected a dict with "
            f"format={ALIGNMENT_FORMAT!r}). Legacy pickled-module checkpoints are "
            "no longer supported here; load them with the original pre-refactor code."
        )
    module = AlignmentFactory.create(
        entry["class_name"],
        input_dim=entry["input_dim"],
        **entry["kwargs"],
    ).float()
    # set_modality must run before load_state_dict so the parameter structure
    # matches what was saved (FreezeAlign/SAIL select submodules).
    if hasattr(module, "set_modality"):
        module.set_modality(entry.get("modality") or modality)
    module.load_state_dict(entry["state_dict"])
    return module.to(device).eval()
