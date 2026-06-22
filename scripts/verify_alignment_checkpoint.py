"""Capture / compare golden forward outputs for an alignment checkpoint.

Used to verify the state_dict checkpoint refactor (goal 1) does not change behavior:
1. BEFORE refactor:  `python scripts/verify_alignment_checkpoint.py CKPT --save golden.pt`
2. AFTER  refactor:  `python scripts/verify_alignment_checkpoint.py CKPT --compare golden.pt`

Deterministic inputs are built from each module's input_dim, covering both the CAP
3D token path (+text mask) and the 2D CLS-fallback path. Loading currently assumes the
OLD pickled-module format (`ckpt["alignment_image"]` is an nn.Module); once the
bidirectional loader exists, route through it instead.
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")  # repo root, so pickled `src.alignment.*` classes resolve

from src.utils.checkpoint import load_alignment_layer  # noqa: E402

SEED = 1234
B, T = 4, 16  # fixed batch / token count for the golden inputs


def load_modules(ckpt_path):
    # Route through the bidirectional loader so this verifies whichever format
    # the checkpoint is in (legacy pickled module OR migrated state_dict).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    img = load_alignment_layer(ckpt["alignment_image"], "image", "cpu")
    txt = load_alignment_layer(ckpt["alignment_text"], "text", "cpu")
    return img, txt


def input_dim_of(module):
    if hasattr(module, "anchors"):
        return module.anchors.shape[1]
    if hasattr(module, "input_dim"):
        return module.input_dim
    # fall back to the first 2D parameter's last dim
    for p in module.parameters():
        if p.dim() >= 2:
            return p.shape[-1]
    raise RuntimeError("could not infer input_dim")


def make_inputs(d_img, d_txt):
    g = torch.Generator().manual_seed(SEED)
    img_tok = torch.randn(B, T, d_img, generator=g)          # (B,T,D) image patches
    txt_tok = torch.randn(B, T, d_txt, generator=g)          # (B,T,D) text tokens
    # deterministic mask: varying valid lengths per row, at least 1 valid
    lengths = torch.tensor([T, T - 3, T - 7, 1])
    mask = (torch.arange(T)[None, :] < lengths[:, None])      # (B,T) bool
    img_cls = torch.randn(B, d_img, generator=g)             # (B,D) 2D fallback
    txt_cls = torch.randn(B, d_txt, generator=g)
    return img_tok, txt_tok, mask, img_cls, txt_cls


@torch.no_grad()
def forward_outputs(img, txt):
    d_img, d_txt = input_dim_of(img), input_dim_of(txt)
    img_tok, txt_tok, mask, img_cls, txt_cls = make_inputs(d_img, d_txt)
    out = {
        "img_token": img(img_tok),
        "txt_token_masked": txt(txt_tok, mask),
        "img_cls2d": img(img_cls),
        "txt_cls2d": txt(txt_cls),
        "_meta": {"d_img": d_img, "d_txt": d_txt, "B": B, "T": T, "seed": SEED},
    }
    return out


def summarize(out):
    for k, v in out.items():
        if k.startswith("_"):
            continue
        print(f"  {k:18s} shape={tuple(v.shape)} sum={v.double().sum().item():.6f} "
              f"mean={v.double().mean().item():.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--save", help="write golden outputs to this path")
    ap.add_argument("--compare", help="compare against a saved golden file")
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--rtol", type=float, default=0.0)
    args = ap.parse_args()

    img, txt = load_modules(args.ckpt)
    out = forward_outputs(img, txt)
    print(f"=== {args.ckpt}")
    print(f"  classes: img={type(img).__name__} txt={type(txt).__name__}")
    summarize(out)

    if args.save:
        torch.save(out, args.save)
        print(f"saved golden → {args.save}")

    if args.compare:
        ref = torch.load(args.compare, map_location="cpu", weights_only=False)
        ok = True
        for k, v in out.items():
            if k.startswith("_"):
                continue
            rv = ref[k]
            if not torch.allclose(v, rv, atol=args.atol, rtol=args.rtol):
                md = (v - rv).abs().max().item()
                print(f"  MISMATCH {k}: max|Δ|={md:.3e}")
                ok = False
        print("RESULT:", "MATCH ✅" if ok else "MISMATCH ❌")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
