"""Natural-image zero-shot classification UPPER BOUND: CLIP.

Reviewer request: CLIP as an upper bound on the natural-image zs-classification track.
Reproduces the 5 datasets our pipeline reports (stl10, cifar100, caltech101, dtd, eurosat)
with the EXACT same loaders (torchvision 0.16.2, seed=42 subsets for caltech/eurosat —
identical across the `structure` and `vlm_eval` envs), and the same class names +
templates from src.evaluation.zero_shot_metadata. Template ensembling = mean of per-template
L2-normed text embeddings (same protocol as the pipeline's zero-shot classifier).

Headline metric = macro (balanced) accuracy, matching the pipeline's zs tables; micro also shown.
BLIP is NOT included: its original paper reports no zero-shot classification (retrieval only).

Run in `vlm_eval` (open_clip). NOT `structure`.
    CUDA_VISIBLE_DEVICES=<g> python vlm_natural_zs.py ViT-L-14      # main-table (vitL) scale
    CUDA_VISIBLE_DEVICES=<g> python vlm_natural_zs.py ViT-B-16      # vitB scale
"""
import os, sys
import numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torchvision.datasets as dsets
from sklearn.metrics import accuracy_score, balanced_accuracy_score
sys.path.insert(0, "/home/shiwon/PAL_working")
from src.evaluation.zero_shot_metadata import DATASETS_TO_CLASSES, DATASETS_TO_TEMPLATES

DEV = "cuda"
DATA = "/home/shiwon/STRUCTURE/data"

# ─────────── val datasets, mirroring src/datasets/data_utils.py exactly (transform=None → PIL) ───────────
def val_dataset(name):
    if name == "stl10":
        return dsets.STL10(root=DATA, split="test", download=False)
    if name == "cifar100":
        return dsets.CIFAR100(root=DATA, train=False, download=False)
    if name == "dtd":
        return dsets.DTD(root=DATA, split="test", download=False)
    if name == "caltech101":
        base = dsets.ImageFolder(root=os.path.join(DATA, "caltech-101/101_ObjectCategories"))
        tgt = np.array(base.targets); train = []
        for t in np.unique(tgt):
            np.random.seed(42)
            n = min(30, int((tgt == t).sum()))
            train.extend(np.random.choice(np.where(tgt == t)[0], size=n, replace=False))
        val = list(set(range(len(tgt))) - set(train))
        return Subset(base, val)
    if name == "eurosat":
        base = dsets.EuroSAT(root=DATA, download=False)
        tgt = np.array(base.targets); val = []
        for t in np.unique(tgt):
            np.random.seed(42)
            sub = np.random.choice(np.where(tgt == t)[0], size=1500, replace=False)
            val.extend(sub[1000:])
        return Subset(base, val)
    raise ValueError(name)

class PPDS(Dataset):
    def __init__(self, ds, pp): self.ds = ds; self.pp = pp
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, y = self.ds[i]
        return self.pp(img.convert("RGB")), y

@torch.no_grad()
def zs_classifier(classnames, templates, encode_text):
    """(D, C) weight matrix: mean of L2-normed per-template text embeds, then L2-norm."""
    W = []
    for c in classnames:
        e = encode_text([t.format(c) for t in templates])          # (T, D)
        e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        W.append(torch.nn.functional.normalize(e, dim=-1))
    return torch.stack(W, 1).to(DEV)                                 # (D, C)

@torch.no_grad()
def eval_clip(arch):
    import open_clip
    m, _, pp = open_clip.create_model_and_transforms(arch, pretrained="openai", force_quick_gelu=True)
    m = m.to(DEV).eval(); tok = open_clip.get_tokenizer(arch)
    enc_txt = lambda ts: m.encode_text(tok(ts).to(DEV))
    print(f"===== CLIP {arch} (openai) zero-shot =====", flush=True)
    accs = []
    for name in ["stl10", "cifar100", "caltech101", "dtd", "eurosat"]:
        ds = val_dataset(name)
        classnames = DATASETS_TO_CLASSES[name]; templates = DATASETS_TO_TEMPLATES[name]
        W = zs_classifier(classnames, templates, enc_txt)
        ys, ps = [], []
        for imgs, y in DataLoader(PPDS(ds, pp), batch_size=256, num_workers=8):
            f = m.encode_image(imgs.to(DEV))
            f = torch.nn.functional.normalize(f, dim=-1)
            ps.append((f @ W).argmax(1).cpu()); ys.append(y)
        y = torch.cat(ys).numpy(); p = torch.cat(ps).numpy()
        macro = balanced_accuracy_score(y, p); micro = accuracy_score(y, p)
        accs.append(macro)
        print(f"  {name:12s} n={len(y):6d} C={len(classnames):3d} | macro={macro:.4f} micro={micro:.4f}", flush=True)
    print(f"  {'MEAN(macro)':12s} = {np.mean(accs):.4f}", flush=True)

if __name__ == "__main__":
    arch = sys.argv[1] if len(sys.argv) > 1 else "ViT-L-14"
    eval_clip(arch)
    print("VLM_NATURAL_ZS_DONE", flush=True)
