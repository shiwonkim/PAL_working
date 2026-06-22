"""Migrate legacy pickled-module checkpoints to the new state_dict format.

Old checkpoints store ``alignment_image`` / ``alignment_text`` as whole pickled
``nn.Module`` objects. This script rebuilds the self-describing new-format entry
(see ``src/utils/checkpoint.py``) from the saved module + config, leaving every
other key (epoch, optimizer, config, loss, ...) untouched.

NEVER writes in place over the frozen originals under ``~/STRUCTURE`` — always
emits to a separate ``--out`` path / ``--out-dir``.

Usage:
    # single checkpoint
    python scripts/migrate_checkpoints.py CKPT.pth --out OUT.pth

    # many, mirrored under an output dir (relative paths preserved)
    python scripts/migrate_checkpoints.py "results/**/checkpoint-*.pth" --out-dir migrated/
"""
import argparse
import glob
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")  # repo root, so pickled `src.alignment.*` classes resolve

from src.utils.checkpoint import (  # noqa: E402
    CLASS_NAME_ALIASES,
    is_new_format,
    serialize_alignment_layer,
)

ALIGNMENT_KEYS = {"alignment_image": "image", "alignment_text": "text"}


def _input_dim_of(module):
    dim = getattr(module, "input_dim", None)
    if dim is not None:
        return int(dim)
    if hasattr(module, "anchors"):
        return int(module.anchors.shape[1])
    for p in module.parameters():
        if p.dim() >= 2:
            return int(p.shape[-1])
    raise RuntimeError(f"cannot infer input_dim for {type(module).__name__}")


def migrate_checkpoint(ckpt: dict) -> tuple[dict, int]:
    """Return (migrated_ckpt, num_layers_converted). Non-destructive on input."""
    cfg = ckpt.get("config", {})
    training = cfg.get("training", {})
    class_name = training.get("alignment_layer_name")
    # Old checkpoints' configs store pre-rename names; write the PAL name.
    if class_name is not None:
        class_name = CLASS_NAME_ALIASES.get(class_name, class_name)
    kwargs = training.get("alignment_layer_kwargs", {})

    converted = 0
    for key, modality in ALIGNMENT_KEYS.items():
        entry = ckpt.get(key)
        if entry is None or is_new_format(entry):
            continue  # missing or already migrated
        if class_name is None:
            raise RuntimeError(
                f"checkpoint has no config.training.alignment_layer_name; "
                f"cannot rebuild {key}"
            )
        ckpt[key] = serialize_alignment_layer(
            entry,
            class_name=class_name,
            input_dim=_input_dim_of(entry),
            kwargs=kwargs,
            modality=modality,
        )
        converted += 1
    return ckpt, converted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="checkpoint path or glob (use quotes for globs)")
    ap.add_argument("--out", help="output path (single input only)")
    ap.add_argument("--out-dir", help="output dir; input relative paths are mirrored")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern, recursive=True))
    if not paths:
        sys.exit(f"no checkpoints matched: {args.pattern}")
    if args.out and len(paths) > 1:
        sys.exit("--out works with a single input; use --out-dir for multiple")
    if not args.out and not args.out_dir:
        sys.exit("provide --out or --out-dir")

    for p in paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        ckpt, n = migrate_checkpoint(ckpt)

        if args.out:
            out = Path(args.out)
        else:
            out = Path(args.out_dir) / p
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, out)
        print(f"migrated {n} layer(s): {p} -> {out}")


if __name__ == "__main__":
    main()
