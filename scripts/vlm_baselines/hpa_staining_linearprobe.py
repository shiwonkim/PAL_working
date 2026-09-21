"""HPA staining LINEAR PROBE (vision-only) — data-efficiency comparison vs iSight/PLIP/CONCH.

iSight & fine-tuned PLIP/CONCH used the FULL 10.5M train set (full fine-tune). Here we freeze
everything and train only a linear head (sklearn LogisticRegression) on ~10K labels, to measure
how much staining info the frozen VISION representation carries. Text encoder / anchors are NOT
used (linear probe is vision-only).

Features compared (both frozen):
  - UNI2-raw   : UNI2-h CLS token (1536-d), no PAL              -> "encoder alone"
  - PAL-pooled : UNI2 tokens -> PAL alignment_image CAP (512-d) -> "PAL-aligned vision rep"
                 computed for BOTH checkpoints (generic 200K, cap1 100K)

Tasks (macro/micro acc on val 2000): intensity(4, all) / location(3, pos) / quantity(3, pos).
Probe train = ~N per intensity-class from HPA train (val-disjoint). GPU forwards ~12K+2K images.

Usage: CUDA_VISIBLE_DEVICES=<g> python scripts/vlm_baselines/hpa_staining_linearprobe.py --per-class 3000
"""
import argparse, os, sys, re
import numpy as np, torch, torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
sys.path.insert(0, "/home/shiwon/PAL_working")
from src.models.backbones.vision_models import load_lvm
from src.utils.checkpoint import load_alignment_layer

DEV = "cuda"
SEL = "/home/data/2026_hpa10m/dataset/selections"
IMG_DIR = "/home/shiwon/STRUCTURE/data/hpa10m/images"
LVM = "hf-hub:MahmoodLab/UNI2-h"
CKPTS = {
    "generic200k": "results/alignment-microsoft_BiomedNLP_BiomedBERT_base_uncased_abstract_fulltext-hf_hub:MahmoodLab_UNI2_h-lunar-darkness-224/(23, 12)_nan_seed42/checkpoints/checkpoint-epoch246.pth",
    "cap1_100k":   "results/alignment-microsoft_BiomedNLP_BiomedBERT_base_uncased_abstract_fulltext-hf_hub:MahmoodLab_UNI2_h-eager-planet-229/(23, 12)_nan_seed42/checkpoints/checkpoint-epoch223.pth",
    "genbal_96k":  "results/alignment-microsoft_BiomedNLP_BiomedBERT_base_uncased_abstract_fulltext-hf_hub:MahmoodLab_UNI2_h-desert-blaze-230/(23, 12)_nan_seed42/checkpoints/checkpoint-epoch246.pth",
    # PathCap-trained (cross-domain transfer test: no HPA data seen in training)
    "pathcap_220k":"results/alignment-microsoft_BiomedNLP_BiomedBERT_base_uncased_abstract_fulltext-hf_hub:MahmoodLab_UNI2_h-comic-haze-232/(23, 12)_nan_seed42/checkpoints/checkpoint-epoch78.pth",
}
INTENSITY = ["negative", "weak", "moderate", "strong"]
LOCATION  = ["nuclear", "cytoplasmic/membranous", "cytoplasmic/membranous,nuclear"]
QUANTITY  = ["<25%", "25%-75%", ">75%"]

class ImgDS(Dataset):
    def __init__(self, paths, tfm): self.paths=paths; self.tfm=tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return self.tfm(Image.open(self.paths[i]).convert("RGB")), i

@torch.no_grad()
def extract(vision, tfm, ais, paths, img_layer, bs=64):
    """Return CLS (N,1536), mean-pooled patch tokens (N,1536) and
    {ckpt: PAL-pooled (N,512)} for a list of image paths.

    The mean-patch feature is a pooling control: PAL sees all 265 tokens through
    CAP while the CLS baseline sees one token, so a PAL-vs-CLS gap mixes the
    alignment effect with the "looked at every token" effect. Mean-pooling the
    patch tokens (skipping the encoder's num_prefix_tokens leading non-patch tokens —
    UNI2-h: 1 CLS + 8 register, so [9:]; DINOv2 / UNI v1: [1:])
    keeps the raw encoder but gives it the same whole-image pooling.
    """
    n_prefix = int(getattr(vision, "num_prefix_tokens", 1))
    N=len(paths); cls=torch.zeros(N,1536); meanp=torch.zeros(N,1536)
    pooled={k: None for k in ais}
    for imgs, idx in DataLoader(ImgDS(paths, tfm), batch_size=bs, num_workers=8):
        toks = vision(imgs.to(DEV))[f"blocks.{img_layer}.add_1"].float()   # (B,265,1536)
        cls[idx] = toks[:,0,:].cpu()                                       # CLS token
        meanp[idx] = toks[:,n_prefix:,:].mean(dim=1).cpu()                 # patch tokens only
        for k, ai in ais.items():
            e = ai(toks); e = e.reshape(e.shape[0], -1).cpu()
            if pooled[k] is None: pooled[k]=torch.zeros(N, e.shape[1])
            pooled[k][idx] = e
    return cls.numpy(), meanp.numpy(), {k: v.numpy() for k,v in pooled.items()}

def probe(Xtr, ytr, Xte, yte):
    sc=StandardScaler().fit(Xtr)
    clf=LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    p=clf.predict(sc.transform(Xte))
    return balanced_accuracy_score(yte,p), accuracy_score(yte,p)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--per-class", type=int, default=3000)
    ap.add_argument("--img-layer", type=int, default=23); args=ap.parse_args()

    tr=pd.read_csv(f"{SEL}/hpa10m_train_whole256_seed42.csv")
    va=pd.read_csv(f"{SEL}/hpa10m_val_whole256_seed42.csv")
    # probe train pool: balanced by intensity (covers negatives + positives for all 3 tasks)
    pool=(tr.groupby("staining_intensity", group_keys=False)
            .apply(lambda g: g.sample(min(args.per_class,len(g)), random_state=42)).reset_index(drop=True))
    print(f"probe train pool: {len(pool)} (per-class~{args.per_class}); val: {len(va)}", flush=True)

    vision, tfm = load_lvm(LVM, img_size=224, device=DEV); vision=vision.to(DEV).eval()
    ais={k: load_alignment_layer(torch.load(p, weights_only=False, map_location="cpu")["alignment_image"],"image",DEV).eval()
         for k,p in CKPTS.items()}

    trp=[os.path.join(IMG_DIR, os.path.basename(p)) for p in pool["image_path"]]
    vap=[os.path.join(IMG_DIR, os.path.basename(p)) for p in va["image_path"]]
    print("extracting train features ...", flush=True); tr_cls, tr_mp, tr_pool = extract(vision, tfm, ais, trp, args.img_layer)
    print("extracting val features ...", flush=True);   va_cls, va_mp, va_pool = extract(vision, tfm, ais, vap, args.img_layer)

    feats={"UNI2-raw(CLS)": (tr_cls, va_cls),
           "UNI2-meanpatch": (tr_mp, va_mp)}
    for k in ais: feats[f"PAL-{k}"]=(tr_pool[k], va_pool[k])

    # tissue: 65-way over all rows (classes present in both train pool and val)
    TISSUE=sorted(set(pool["tissue"].dropna()) & set(va["tissue"].dropna()))
    # malignancy: binary, derived from the tissue name (cancer type -> malignant)
    CANCER=re.compile(r"cancer|carcinoma|melanoma|lymphoma|glioma|carcinoid|sarcoma|tumou?r|malignant|leukemia|blastoma|myeloma", re.I)
    MALIG=["malignant","benign"]
    for d in (pool, va):
        d["malig_label"]=["malignant" if CANCER.search(str(t)) else "benign" for t in d["tissue"]]
    tasks=[("tissue","tissue",TISSUE,None),
           ("malignancy","malig_label",MALIG,None),
           ("intensity","staining_intensity",INTENSITY,None),
           ("location","staining_location",LOCATION,"pos"),
           ("quantity","staining_quantity",QUANTITY,"pos")]
    print("\n=== LINEAR PROBE (macro / micro acc) ===", flush=True)
    for tname, col, classes, mode in tasks:
        # masks (positive-only for location/quantity)
        trm = pool[col].isin(classes); vam = va[col].isin(classes)
        ytr=np.array([classes.index(v) for v in pool[col][trm]])
        yte=np.array([classes.index(v) for v in va[col][vam]])
        print(f"[{tname}] train n={trm.sum()} val n={vam.sum()} C={len(classes)}", flush=True)
        for fname,(Xtr,Xte) in feats.items():
            ma,mi=probe(Xtr[trm.values], ytr, Xte[vam.values], yte)
            print(f"    {fname:14s}: macro={ma:.4f} micro={mi:.4f}", flush=True)
    print("PROBE_DONE", flush=True)

if __name__=="__main__":
    main()
