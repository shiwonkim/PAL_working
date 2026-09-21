"""HPA TISSUE-TYPE zero-shot classification with a trained PAL checkpoint.

Companion to hpa_staining_zs.py. Tests the grounding hypothesis: tissue morphology is
visible at a 224 whole-core thumbnail, whereas staining location/quantity are subcellular
and are not. If the hypothesis holds, tissue zero-shot should be far above random (1/65),
while staining location/quantity are only modestly above their random baselines.

65-way over the `tissue` metadata field (val, n=2000). Prompts use the tissue metadata name
in an ensemble of neutral IHC templates (mean text prototype), matching neither caption style
exactly so generic and cap1 checkpoints are compared on equal prompts. Reports top-1 and top-5,
macro (balanced) and micro accuracy.

Usage:
  CUDA_VISIBLE_DEVICES=<g> python scripts/vlm_baselines/hpa_tissue_zs.py \
      --ckpt <checkpoint.pth> --img-layer 23 --txt-layer 12 --label generic
"""
import argparse, os, sys, re
import numpy as np, torch, torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score
sys.path.insert(0, "/home/shiwon/PAL_working")
from src.models.backbones.vision_models import load_lvm
from src.models.backbones.text_models import load_llm, load_tokenizer
from src.utils.checkpoint import load_alignment_layer

DEV = "cuda"
VAL_CSV = "/home/data/2026_hpa10m/dataset/selections/hpa10m_val_whole256_seed42.csv"
IMG_DIR = "/home/shiwon/STRUCTURE/data/hpa10m/images"
LVM = "hf-hub:MahmoodLab/UNI2-h"
LLM = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# Neutral IHC template ensemble; {t} filled with the tissue metadata name.
TEMPLATES = [
    "Immunohistochemical staining of human {t}.",
    "An immunohistochemistry image of human {t}.",
    "Histopathology of human {t}.",
    "A tissue microarray core of human {t}.",
]

# Malignancy is derived from the tissue name (HPA has no explicit label): a cancer
# type -> malignant, a named normal organ -> benign. iSight reports this as a task too.
CANCER=re.compile(r"cancer|carcinoma|melanoma|lymphoma|glioma|carcinoid|sarcoma|tumou?r|malignant|leukemia|blastoma|myeloma", re.I)
MALIG_PROMPTS = {
    "malignant": ["Immunohistochemical staining of human cancer tissue.",
                  "Immunohistochemical staining of human malignant tumor.",
                  "An immunohistochemistry image of human carcinoma.",
                  "Histopathology of malignant human tissue."],
    "benign":    ["Immunohistochemical staining of human normal tissue.",
                  "Immunohistochemical staining of human benign tissue.",
                  "An immunohistochemistry image of normal human tissue.",
                  "Histopathology of benign human tissue."],
}

class ImgDS(Dataset):
    def __init__(self, paths, tfm): self.paths=paths; self.tfm=tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tfm(Image.open(self.paths[i]).convert("RGB")), i

@torch.no_grad()
def image_embeddings(vision, tfm, ai, paths, img_layer, bs=64):
    embs=[]
    for imgs, idx in DataLoader(ImgDS(paths, tfm), batch_size=bs, num_workers=8):
        feats = vision(imgs.to(DEV))[f"blocks.{img_layer}.add_1"].float()  # (B,T,D)
        e = ai(feats)
        embs.append(F.normalize(e.reshape(e.shape[0], -1), dim=-1).cpu())
    return torch.cat(embs)

@torch.no_grad()
def text_prototypes(language, tok, at, classes, txt_layer):
    protos=[]
    for name in classes:
        vs=[]
        for tmpl in TEMPLATES:
            p = tmpl.format(t=name)
            enc = tok(p, return_tensors="pt", padding="longest", truncation=True, max_length=128).to(DEV)
            out = language(**enc, output_hidden_states=True)
            toks = out.hidden_states[txt_layer].float()
            mask = enc["attention_mask"]
            e = at(toks, mask=mask)
            vs.append(F.normalize(e.reshape(1,-1), dim=-1))
        protos.append(F.normalize(torch.stack(vs).mean(0), dim=-1))
    return torch.cat(protos).to(DEV)  # (C,K)

@torch.no_grad()
def text_protos_from_dict(language, tok, at, class_prompts, txt_layer):
    protos=[]
    for name, prompts in class_prompts.items():
        vs=[]
        for p in prompts:
            enc = tok(p, return_tensors="pt", padding="longest", truncation=True, max_length=128).to(DEV)
            out = language(**enc, output_hidden_states=True)
            toks = out.hidden_states[txt_layer].float()
            e = at(toks, mask=enc["attention_mask"])
            vs.append(F.normalize(e.reshape(1,-1), dim=-1))
        protos.append(F.normalize(torch.stack(vs).mean(0), dim=-1))
    return torch.cat(protos).to(DEV)

def topk_acc(sim, y, k):
    topk = sim.topk(k, dim=1).indices.cpu().numpy()  # (N,k)
    return np.mean([yi in row for yi, row in zip(y, topk)])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img-layer", type=int, default=23)
    ap.add_argument("--txt-layer", type=int, default=12)
    ap.add_argument("--label", default="")
    ap.add_argument("--min-count", type=int, default=0,
                    help="drop tissue classes with < this many val rows (0 = keep all 65)")
    args=ap.parse_args()

    df=pd.read_csv(VAL_CSV)
    if args.min_count > 0:
        keep = df["tissue"].value_counts()
        keep = set(keep[keep >= args.min_count].index)
        df = df[df["tissue"].isin(keep)].reset_index(drop=True)
    classes = sorted(df["tissue"].dropna().unique().tolist())
    paths=[os.path.join(IMG_DIR, os.path.basename(p)) for p in df["image_path"]]

    vision, tfm = load_lvm(LVM, img_size=224, device=DEV); vision=vision.to(DEV).eval()
    language=load_llm(LLM).to(DEV).eval(); tok=load_tokenizer(LLM)
    ck=torch.load(args.ckpt, weights_only=False, map_location="cpu")
    ai=load_alignment_layer(ck["alignment_image"],"image",DEV).eval()
    at=load_alignment_layer(ck["alignment_text"],"text",DEV).eval()

    print(f"=== HPA TISSUE zero-shot [{args.label}] | {len(classes)}-way, n={len(df)} (random top1={1/len(classes):.3f}) ===", flush=True)
    img_emb=image_embeddings(vision, tfm, ai, paths, args.img_layer)
    proto=text_prototypes(language, tok, at, classes, args.txt_layer)
    ie=img_emb.to(DEV)
    sim = ie @ proto.T                       # (N,C)
    y=np.array([classes.index(v) for v in df["tissue"]])
    pred=sim.argmax(1).cpu().numpy()
    macro=balanced_accuracy_score(y,pred); micro=accuracy_score(y,pred)
    top5=topk_acc(sim, y, 5)
    print(f"  tissue  C={len(classes)} n={len(y)} | top1 macro={macro:.4f} micro={micro:.4f} | top5 micro={top5:.4f}", flush=True)

    # --- malignancy (binary, derived from tissue name) ---
    mcls=list(MALIG_PROMPTS.keys())  # [malignant, benign]
    mproto=text_protos_from_dict(language, tok, at, MALIG_PROMPTS, args.txt_layer)  # (2,K)
    ym=np.array([0 if CANCER.search(str(t)) else 1 for t in df["tissue"]])
    mpred=(ie @ mproto.T).argmax(1).cpu().numpy()
    mmac=balanced_accuracy_score(ym,mpred); mmic=accuracy_score(ym,mpred)
    print(f"  malignancy C=2 n={len(ym)} ({100*(ym==0).mean():.0f}% malignant) | macro={mmac:.4f} micro={mmic:.4f}", flush=True)
    print("HPA_TISSUE_ZS_DONE", flush=True)

if __name__=="__main__":
    main()
