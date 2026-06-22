"""Batch-verify migration round-trips for every checkpoint type.

For each matched checkpoint: load the legacy pickled module, migrate it in
memory, rebuild via the new-format loader, and assert the rebuilt layer is
equivalent to the original:

  1. same class
  2. identical state_dict (same keys, bit-equal tensors)
  3. identical 2D (CLS) forward output on a fixed random input
  4. (token layers only) identical 3D + mask forward output

state_dict identity + same class + same kwargs already guarantees deterministic
forward equality; the forward checks are belt-and-suspenders and exercise the
actual rebuilt module. Layer-agnostic: every alignment layer accepts 2D input.

Usage:
    python scripts/verify_migration_roundtrip.py "results/**/checkpoint-*.pth"
"""
import argparse
import glob
import sys

import torch

sys.path.insert(0, ".")  # repo root, so pickled `src.alignment.*` classes resolve

from src.utils.checkpoint import load_alignment_layer  # noqa: E402
from scripts.migrate_checkpoints import migrate_checkpoint  # noqa: E402

SEED = 1234
B, T = 4, 16


def _state_dicts_equal(a, b):
    if a.keys() != b.keys():
        return False, f"keys differ: {set(a) ^ set(b)}"
    for k in a:
        if not torch.equal(a[k], b[k]):
            return False, f"tensor '{k}' differs (max|Δ|={ (a[k]-b[k]).abs().max().item():.3e})"
    return True, ""


@torch.no_grad()
def _forward_equal(legacy, rebuilt, dim, modality):
    g = torch.Generator().manual_seed(SEED)
    # 2D CLS path — supported by every alignment layer
    z2d = torch.randn(B, dim, generator=g)
    o_leg, o_new = legacy(z2d), rebuilt(z2d)
    if not torch.allclose(o_leg, o_new, atol=0, rtol=0):
        return False, "2D forward differs"
    # 3D + mask path — only some layers accept a mask arg
    z3d = torch.randn(B, T, dim, generator=g)
    lengths = torch.tensor([T, T - 3, T - 7, 1])
    mask = torch.arange(T)[None, :] < lengths[:, None]
    try:
        o_leg3 = legacy(z3d, mask)
        o_new3 = rebuilt(z3d, mask)
    except TypeError:
        return True, "2D ok (layer has no mask path)"
    if not torch.allclose(o_leg3, o_new3, atol=0, rtol=0):
        return False, "3D+mask forward differs"
    return True, "2D + 3D ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern, recursive=True))
    if not paths:
        sys.exit(f"no checkpoints matched: {args.pattern}")

    all_ok = True
    print(f"{'checkpoint':24s} {'class':30s} {'state_dict':10s} {'forward':22s}")
    print("-" * 90)
    for p in paths:
        label = p.split("/")[1].split("-")[-1][:22]
        try:
            legacy_ckpt = torch.load(p, map_location="cpu", weights_only=False)
            leg_img = legacy_ckpt["alignment_image"]
            cls_name = type(leg_img).__name__

            # migrate a fresh copy so the legacy module stays intact for comparison
            migrated, _ = migrate_checkpoint(
                torch.load(p, map_location="cpu", weights_only=False)
            )
            for key, modality in (("alignment_image", "image"), ("alignment_text", "text")):
                legacy = load_alignment_layer(legacy_ckpt[key], modality, "cpu")
                rebuilt = load_alignment_layer(migrated[key], modality, "cpu")
                assert type(legacy).__name__ == type(rebuilt).__name__, "class mismatch"
                sd_ok, sd_msg = _state_dicts_equal(legacy.state_dict(), rebuilt.state_dict())
                fw_ok, fw_msg = _forward_equal(
                    legacy, rebuilt, rebuilt.input_dim, modality
                )
                if not (sd_ok and fw_ok):
                    all_ok = False
                    print(f"{label:24s} {cls_name:30s} "
                          f"{'OK' if sd_ok else 'FAIL':10s} {fw_msg if not fw_ok else '':22s}"
                          f"  <-- {key} {sd_msg}{'' if fw_ok else ' / '+fw_msg}")
                    break
            else:
                print(f"{label:24s} {cls_name:30s} {'OK':10s} {fw_msg:22s}")
        except Exception as e:
            all_ok = False
            print(f"{label:24s} {'?':30s} {'ERROR':10s} {type(e).__name__}: {str(e)[:40]}")

    print("-" * 90)
    print("OVERALL:", "ALL PASS ✅" if all_ok else "FAILURES ❌")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
