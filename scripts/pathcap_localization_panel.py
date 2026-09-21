"""9-class CRC grounding panel: for EACH class, pick a correctly-classified representative
tile (highest-confidence correct prediction within a seeded random sample) and show PAL's
grounding heatmap for that (correct) class. Reuses pathcap_localization.py.

Run (structure env, GPU):
  python scripts/pathcap_localization_panel.py --out fig.png [--sample 40]
"""
import argparse, os, random, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/shiwon/PAL_working")
from scripts.pathcap_localization import (
    build_models, image_tokens, patch_anchor_sims, text_profiles,
    image_embedding, heat, overlay, DEV,
)
from src.evaluation.zero_shot_metadata import DATASETS_TO_CLASSES, DATASETS_TO_TEMPLATES

CRC_DIR = "/home/shiwon/STRUCTURE/data/crc100k/CRC-VAL-HE-7K"
# CRC-VAL folder codes in ImageFolder (alphabetical) order == DATASETS_TO_CLASSES['crc100k'] order
CODES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

@torch.no_grad()
def pick_and_ground(vision, transform, ai, at, language, tokenizer, prof, classes,
                    img_size, sample, seed):
    """For each class, find the best correctly-classified tile; return (tile, prob-map, conf)."""
    results = []
    for ci, code in enumerate(CODES):
        files = sorted(os.listdir(os.path.join(CRC_DIR, code)))
        random.seed(seed + ci)
        cand = random.sample(files, min(sample, len(files)))
        best = None  # (conf, path, feats)
        for fn in cand:
            path = os.path.join(CRC_DIR, code, fn)
            feats = image_tokens(vision, transform, Image.open(path).convert("RGB"), img_size)
            emb = image_embedding(feats, ai)
            logits = (F.normalize(emb, dim=-1) @ F.normalize(prof, dim=-1).T)[0]
            pred = int(logits.argmax())
            if pred == ci:  # correctly classified
                # rank by cosine margin (true - best other): a meaningful "most confident"
                other = logits.clone(); other[ci] = -1e9
                margin = float(logits[ci] - other.max())
                if best is None or margin > best[0]:
                    best = (margin, path, feats)
        results.append((ci, code, best))
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    vision, transform, language, tokenizer, ai, at = build_models(args.img_size)
    classes = DATASETS_TO_CLASSES["crc100k"]; templates = DATASETS_TO_TEMPLATES["crc100k"]
    prof = text_profiles(language, tokenizer, at, classes, templates)   # (C, K)

    res = pick_and_ground(vision, transform, ai, at, language, tokenizer, prof, classes,
                          args.img_size, args.sample, args.seed)
    hw = (args.img_size, args.img_size)

    # layout: 3 classes per row, each = (original | grounding overlay) -> 3 rows x 6 cols
    fig, axs = plt.subplots(3, 6, figsize=(6 * 2.6, 3 * 2.8)); axs = axs.ravel()
    for a in axs: a.axis("off")
    for ci, code, best in res:
        slot = (ci // 3) * 6 + (ci % 3) * 2
        name = classes[ci]
        if best is None:
            axs[slot].set_title(f"{name}\n(no correct tile in sample)", fontsize=8); continue
        margin, path, feats = best
        base = np.asarray(Image.open(path).convert("RGB").resize(hw))
        S_pa = patch_anchor_sims(feats, ai)
        sim = S_pa @ prof.T
        probs = F.softmax(sim / 0.02, dim=1)
        grid = int(round(S_pa.shape[0] ** 0.5))
        axs[slot].imshow(base); axs[slot].set_title(f"{name}\n[{code}]  ✓ margin={margin:.3f}", fontsize=8)
        axs[slot].axis("off")
        overlay(axs[slot + 1], base, heat(probs[:, ci], grid, hw, renorm=False),
                "grounding", vmin=0, vmax=float(probs.max()))
    fig.suptitle("CRC zero-shot grounding — correctly-classified representative tile per class "
                 f"(seed={args.seed}, best-of-{args.sample})", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"SAVED {args.out}", flush=True)

if __name__ == "__main__":
    main()
