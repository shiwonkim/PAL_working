"""PAL grounding / localization viz for the PathCap UNI+PubMedBERT (token-level, K=512) model.

Shows WHERE a text (a CRC zero-shot class, or a free caption / word) lands on a pathology
image — the spatial evidence behind PAL's zero-shot decision.

Mechanism (token-level CAP, same as the seg pipeline's anchor_codebook/factorized path):
  image: UNI patch z_p (P,D) -> S_pa[p,k] = cos(z_p, anchor_k)          (P, K=512)
  text : class/caption -> PubMedBERT -> PAL text CAP profile prof (C, K)
  map  : sim[p,c] = Σ_k S_pa[p,k]·prof[c,k]  (anchor bridge)            (P, C)
  -> reshape to grid, upsample, overlay. Predicted class = argmax of CAP-pooled image emb.

Usage (structure env, GPU):
  python scripts/pathcap_localization.py --image <path> --mode class  --out fig.png
  python scripts/pathcap_localization.py --image <path> --mode caption --caption "..." --out fig.png
"""
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/shiwon/PAL_working")
from src.models.backbones.vision_models import load_lvm
from src.models.backbones.text_models import load_llm, load_tokenizer
from src.utils.checkpoint import load_alignment_layer
from src.evaluation.zero_shot_classifier import build_zero_shot_classifier
from src.evaluation.zero_shot_metadata import (
    DATASETS_TO_CLASSES, DATASETS_TO_TEMPLATES, DATASETS_TO_SYNONYMS,
)

DEV = "cuda"
CKPT = ("results/alignment-microsoft_BiomedNLP_BiomedBERT_base_uncased_abstract_fulltext-"
        "hf_hub:MahmoodLab_UNI-pathcap_n200000/(23, 12)_nan_seed42/checkpoints/checkpoint-epoch232.pth")
LVM = "hf-hub:MahmoodLab/UNI"
LLM = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
IMG_LAYER, TXT_LAYER = 23, 12

def build_models(img_size):
    vision, transform = load_lvm(LVM, img_size=img_size, device=DEV)
    vision = vision.to(DEV).eval()
    language = load_llm(LLM).to(DEV).eval()
    tokenizer = load_tokenizer(LLM)
    ckpt = torch.load(CKPT, weights_only=False, map_location="cpu")
    ai = load_alignment_layer(ckpt["alignment_image"], "image", DEV).eval()
    at = load_alignment_layer(ckpt["alignment_text"], "text", DEV).eval()
    return vision, transform, language, tokenizer, ai, at

@torch.no_grad()
def image_tokens(vision, transform, pil, img_size):
    x = transform(pil).unsqueeze(0).to(DEV)
    feats = vision(x)[f"blocks.{IMG_LAYER}.add_1"][0].float()   # (T, D), token0 = CLS
    return feats

@torch.no_grad()
def patch_anchor_sims(feats, ai):
    z = F.normalize(feats[1:], dim=-1)                          # (P, D) patches only
    a = F.normalize(ai.anchors.float(), dim=-1)                 # (K, D)
    return z @ a.T                                              # (P, K)

@torch.no_grad()
def text_profiles(language, tokenizer, at, classnames, templates):
    return build_zero_shot_classifier(
        language_model=language, tokenizer=tokenizer, classnames=classnames,
        templates=templates, dataset=None, layer_index=TXT_LAYER, alignment_layer=at,
        num_classes_per_batch=8, device=DEV, pool_txt="none", save_path=None,
        token_level=True,
    ).float().to(DEV)                                          # (C, K)

@torch.no_grad()
def image_embedding(feats, ai):
    """CAP-pooled image embedding used for the zero-shot decision, in anchor space (1, K)."""
    emb = ai(feats.unsqueeze(0))                               # PAL token forward -> pooled
    return emb.reshape(1, -1).float()

def heat(sim_p, grid, out_hw, renorm=True):
    """(P,) -> upsampled (H,W). renorm=True: per-map min-max to [0,1]; else raw values."""
    m = sim_p.reshape(grid, grid)[None, None].float()
    up = F.interpolate(m, size=out_hw, mode="bilinear", align_corners=False)[0, 0]
    if renorm:
        up = up - up.min(); up = up / (up.max() + 1e-8)
    return up.cpu().numpy()

def overlay(ax, base_rgb, hmap, title, alpha=0.5, vmin=None, vmax=None):
    ax.imshow(base_rgb); ax.imshow(hmap, cmap="jet", alpha=alpha, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9); ax.axis("off")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mode", choices=["class", "caption"], default="class")
    ap.add_argument("--caption", default="")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--use-synonyms", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vision, transform, language, tokenizer, ai, at = build_models(args.img_size)
    pil = Image.open(args.image).convert("RGB")
    base = np.asarray(pil.resize((args.img_size, args.img_size)))
    feats = image_tokens(vision, transform, pil, args.img_size)
    S_pa = patch_anchor_sims(feats, ai)                        # (P, K)
    P = S_pa.shape[0]; grid = int(round(P ** 0.5)); hw = (args.img_size, args.img_size)

    if args.mode == "class":
        classes = DATASETS_TO_CLASSES["crc100k"]
        templates = DATASETS_TO_TEMPLATES["crc100k"]
        prof = text_profiles(language, tokenizer, at, classes, templates)   # (C, K)
        sim = S_pa @ prof.T                                   # (P, C)  factorized/anchor bridge
        # Per-patch softmax over classes -> comparable probability maps on a COMMON [0,1]
        # scale, so the predicted class genuinely stands out (vs independent per-map renorm).
        probs = F.softmax(sim / 0.02, dim=1)                  # (P, C)
        # predicted class from CAP-pooled image emb (the actual zs decision)
        img_emb = image_embedding(feats, ai)
        logits = F.normalize(img_emb, dim=-1) @ F.normalize(prof, dim=-1).T
        pred = int(logits.argmax()); pred_name = classes[pred]
        # figure: original + predicted-class overlay + argmax map + all-class grid
        ncol = 4; nrow = 4   # 16 axes: 0=img 1=pred 2=argmax 3=blank, 4..12 = 9 classes
        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
        axs = axs.ravel()
        for a in axs: a.axis("off")
        vmax = float(probs.max())   # common scale across all class prob-maps
        axs[0].imshow(base); axs[0].set_title("image", fontsize=9); axs[0].axis("off")
        overlay(axs[1], base, heat(probs[:, pred], grid, hw, renorm=False),
                f"PRED: {pred_name}\n(zs evidence, prob)", vmin=0, vmax=vmax)
        # argmax-over-classes map (which class each patch most resembles)
        am = sim.argmax(1).reshape(grid, grid).cpu().numpy()
        axs[2].imshow(base); axs[2].imshow(
            np.kron(am, np.ones((hw[0] // grid, hw[1] // grid))), cmap="tab10", alpha=0.5,
            vmin=0, vmax=9); axs[2].set_title("argmax class / patch", fontsize=9); axs[2].axis("off")
        axs[3].axis("off")
        for i, c in enumerate(classes):
            overlay(axs[4 + i], base, heat(probs[:, i], grid, hw, renorm=False),
                    f"{c}" + ("  ★" if i == pred else ""), vmin=0, vmax=vmax)
        fig.suptitle(f"CRC zero-shot grounding — pred={pred_name}", fontsize=11)
    else:
        cap = args.caption.strip()
        assert cap, "--caption required in caption mode"
        words = [w.strip(" .,;:") for w in cap.split() if len(w.strip(" .,;:")) > 2]
        queries = [cap] + words
        prof = text_profiles(language, tokenizer, at, queries, ["{}"])       # (Q, K)
        sim = S_pa @ prof.T                                   # (P, Q)
        n = len(queries); ncol = 4; nrow = (n + 1 + ncol - 1) // ncol
        fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3)); axs = axs.ravel()
        axs[0].imshow(base); axs[0].set_title("image", fontsize=9); axs[0].axis("off")
        overlay(axs[1], base, heat(sim[:, 0], grid, hw), f'FULL: "{cap[:40]}..."')
        for j, w in enumerate(words):
            overlay(axs[2 + j], base, heat(sim[:, 1 + j], grid, hw), f'"{w}"')
        for k in range(2 + len(words), len(axs)): axs[k].axis("off")
        fig.suptitle("Caption / word grounding", fontsize=11)

    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"SAVED {args.out}", flush=True)

if __name__ == "__main__":
    main()
