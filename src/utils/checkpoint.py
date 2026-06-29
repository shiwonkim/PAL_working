"""(De)serialization of alignment layers for state_dict-based checkpoints.

Old checkpoints stored the whole pickled ``nn.Module`` under
``ckpt["alignment_image"]`` / ``ckpt["alignment_text"]``. That requires
``torch.load(weights_only=False)`` and breaks the moment a layer class is renamed
or moved (goal 2 of the refactor). The new format stores a self-describing dict::

    {
        "format":     "alignment_state_dict_v1",
        "class_name": "PALAlignmentLayer",
        "input_dim":  1024,
        "kwargs":     {... alignment_layer_kwargs ...},
        "modality":   "image" | "text" | None,
        "state_dict": OrderedDict(...),
    }

so the module is rebuilt through ``AlignmentFactory`` and loaded via
``load_state_dict``. ``load_alignment_layer`` reads BOTH formats during the
transition; the legacy branch is isolated and will be removed once every
checkpoint has been migrated (see ``scripts/migrate_checkpoints.py``).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from loguru import logger

from src.models.alignment.alignment_factory import AlignmentFactory

ALIGNMENT_FORMAT = "alignment_state_dict_v1"

# Class-name aliases for checkpoints saved before the BridgeAnchor → PAL rename
# and before the CLS/token PAL classes were merged into one PALAlignmentLayer.
# Lets older new-format checkpoints keep loading; drop once all checkpoints store
# the current PAL name.
CLASS_NAME_ALIASES = {
    "BridgeAnchorAlignmentLayer": "PALAlignmentLayer",
    "BridgeAnchorTokenAlignmentLayer": "PALAlignmentLayer",
    "PALTokenAlignmentLayer": "PALAlignmentLayer",
}


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
    """Rebuild an alignment layer from a checkpoint entry.

    Handles both the new state_dict format and the legacy pickled module.
    ``modality`` ("image"/"text") is applied via ``set_modality`` when the layer
    supports it, matching how the trainer builds layers.
    """
    if is_new_format(entry):
        class_name = CLASS_NAME_ALIASES.get(entry["class_name"], entry["class_name"])
        module = AlignmentFactory.create(
            class_name,
            input_dim=entry["input_dim"],
            **entry["kwargs"],
        ).float()
        # set_modality must run before load_state_dict so the parameter
        # structure matches what was saved (FreezeAlign/SAIL select submodules).
        if hasattr(module, "set_modality"):
            module.set_modality(entry.get("modality") or modality)
        module.load_state_dict(entry["state_dict"])
    else:
        # LEGACY: pickled nn.Module from torch.save(model). Remove this branch
        # once every checkpoint has been migrated to the new format.
        logger.warning(
            "Loading a LEGACY pickled-module alignment checkpoint. Migrate it with "
            "scripts/migrate_checkpoints.py; this code path will be removed."
        )
        module = entry
        if hasattr(module, "set_modality"):
            module.set_modality(modality)

    return module.to(device).eval()
