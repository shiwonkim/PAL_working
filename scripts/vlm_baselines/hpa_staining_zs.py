"""HPA staining-attribute ZERO-SHOT classification with a trained PAL checkpoint.

Evaluates a token-level PAL model (UNI2-h + PubMedBERT) on the HPA val set for the three
staining attributes iSight predicts — but zero-shot (no attribute labels seen in training),
via text prompts matched to the generic-caption template. This is the "PAL does iSight's task
zero-shot" comparison.

Tasks (macro accuracy):
  intensity : negative / weak / moderate / strong   (all 2000 val)
  location  : nuclear / cytoplasmic-membranous / both   (positive only; 'none' excluded)
  quantity  : <25% / 25%-75% / >75%                 (positive only)

Reuses the pipeline exactly: builds UNI2-h image features (cached) + PAL image layer -> CAP
pooled image embedding; text prompts -> PubMedBERT + PAL text layer -> class prototypes;
cosine -> argmax. Runs in `structure` env.

Usage:
  CUDA_VISIBLE_DEVICES=<g> python scripts/vlm_baselines/hpa_staining_zs.py \
      --ckpt <checkpoint.pth> --img-layer 23 --txt-layer 12
"""
import argparse, os, sys
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

# Prompts matched to the generic_caption template surface forms.
# {tissue} left generic ("human tissue") so the prompt isolates the attribute.
INTENSITY = {
    "negative": ["Immunohistochemical staining of human tissue shows no staining in tumor cells"],
    "weak":     ["Immunohistochemical staining of human tissue shows low levels of weak staining in tumor cells"],
    "moderate": ["Immunohistochemical staining of human tissue shows medium levels of moderate staining in tumor cells"],
    "strong":   ["Immunohistochemical staining of human tissue shows high levels of strong staining in tumor cells"],
}
LOCATION = {  # positive only
    "nuclear":                ["Immunohistochemical staining of human tissue shows nuclear staining in tumor cells"],
    "cytoplasmic/membranous": ["Immunohistochemical staining of human tissue shows cytoplasmic/membranous staining in tumor cells"],
    "cytoplasmic/membranous,nuclear": ["Immunohistochemical staining of human tissue shows nuclear and cytoplasmic/membranous staining in tumor cells"],
}
QUANTITY = {  # positive only
    "<25%":    ["Immunohistochemical staining of human tissue shows staining in approximately <25% of tumor cells"],
    "25%-75%": ["Immunohistochemical staining of human tissue shows staining in approximately 25-75% of tumor cells"],
    ">75%":    ["Immunohistochemical staining of human tissue shows staining in approximately >75% of tumor cells"],
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
        e = ai(feats)                                                       # CAP pooled (B,K)
        embs.append(F.normalize(e.reshape(e.shape[0], -1), dim=-1).cpu())
    return torch.cat(embs)

@torch.no_grad()
def text_prototypes(language, tok, at, class_prompts, txt_layer):
    protos=[]
    for name, prompts in class_prompts.items():
        vs=[]
        for p in prompts:
            enc = tok(p, return_tensors="pt", padding="longest", truncation=True, max_length=128).to(DEV)
            out = language(**enc, output_hidden_states=True)
            toks = out.hidden_states[txt_layer].float()                    # (1,T,D), cast bf16->f32
            mask = enc["attention_mask"]
            e = at(toks, mask=mask)                                        # CAP pooled (1,K)
            vs.append(F.normalize(e.reshape(1,-1), dim=-1))
        protos.append(F.normalize(torch.stack(vs).mean(0), dim=-1))
    return torch.cat(protos).to(DEV)                                       # (C,K)

def evaluate(tag, img_emb, df, mask, label_col, class_prompts, language, tok, at, txt_layer):
    classes=list(class_prompts.keys())
    proto=text_prototypes(language, tok, at, class_prompts, txt_layer)     # (C,K)
    sub=df[mask]; ie=img_emb[mask.values].to(DEV)
    y=np.array([classes.index(v) for v in sub[label_col]])
    pred=(ie @ proto.T).argmax(1).cpu().numpy()
    print(f"  {tag:10s} n={len(y):5d} C={len(classes)} | macro={balanced_accuracy_score(y,pred):.4f} micro={accuracy_score(y,pred):.4f}", flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img-layer", type=int, default=23)
    ap.add_argument("--txt-layer", type=int, default=12)
    ap.add_argument("--label", default="")
    args=ap.parse_args()

    df=pd.read_csv(VAL_CSV)
    paths=[os.path.join(IMG_DIR, os.path.basename(p)) for p in df["image_path"]]
    vision, tfm = load_lvm(LVM, img_size=224, device=DEV); vision=vision.to(DEV).eval()
    language=load_llm(LLM).to(DEV).eval(); tok=load_tokenizer(LLM)
    ck=torch.load(args.ckpt, weights_only=False, map_location="cpu")
    ai=load_alignment_layer(ck["alignment_image"],"image",DEV).eval()
    at=load_alignment_layer(ck["alignment_text"],"text",DEV).eval()

    print(f"=== HPA staining zero-shot [{args.label}] ===", flush=True)
    img_emb=image_embeddings(vision, tfm, ai, paths, args.img_layer)
    # per-attribute mask: only rows whose label value is one of that attribute's classes
    # (drops 'none'/NaN — e.g. negatives have no location/quantity).
    def valid(col, classes): return df[col].isin(classes)
    evaluate("intensity", img_emb, df, valid("staining_intensity", list(INTENSITY)), "staining_intensity", INTENSITY, language, tok, at, args.txt_layer)
    evaluate("location",  img_emb, df, valid("staining_location", list(LOCATION)), "staining_location", LOCATION, language, tok, at, args.txt_layer)
    evaluate("quantity",  img_emb, df, valid("staining_quantity", list(QUANTITY)), "staining_quantity", QUANTITY, language, tok, at, args.txt_layer)
    print("HPA_STAINING_ZS_DONE", flush=True)

if __name__=="__main__":
    main()
