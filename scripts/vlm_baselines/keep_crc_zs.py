"""KEEP (Astaxanthin/KEEP) pathology VLM — CRC-100K zero-shot classification.

Same protocol as our CONCH/PLIP/QuiltNet CRC eval:
  * CRC-VAL-HE-7K via ImageFolder (ADI..TUM = class 0..8, matches DATASETS_TO_CLASSES['crc100k'])
  * text prototype = mean of L2-normed per-(template) text embeddings, then L2-norm
  * 22 CONCH official templates
  * two prompt protocols: plain (single class name × 22 tmpl) and synonym (all class synonyms × 22 tmpl)
  * report macro (balanced) + micro accuracy

KEEP: ViT-L/16 vision + BERT text, loaded via AutoModel(trust_remote_code=True),
encode_image / encode_text; tokenizer max_length=256, ImageNet-norm 224 center-crop.

Run in vlm_eval env.  CUDA_VISIBLE_DEVICES=<g> python keep_crc_zs.py
"""
import os, sys, numpy as np, torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from transformers import AutoModel, AutoTokenizer
sys.path.insert(0, "/home/shiwon/PAL_working")
from src.evaluation.zero_shot_metadata import (
    DATASETS_TO_CLASSES, DATASETS_TO_TEMPLATES, DATASETS_TO_SYNONYMS,
)

DEV = "cuda"
CRC = "/home/shiwon/STRUCTURE/data/crc100k/CRC-VAL-HE-7K"

transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

print("loading KEEP ...", flush=True)
model = AutoModel.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True).to(DEV).eval()
tokenizer = AutoTokenizer.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)

@torch.no_grad()
def encode_text(strings):
    tok = tokenizer(list(strings), max_length=256, padding="max_length",
                    truncation=True, return_tensors="pt").to(DEV)
    e = model.encode_text(tok)
    return torch.nn.functional.normalize(e.float(), dim=-1)

@torch.no_grad()
def prototype(name_variants, templates):
    """name_variants: list of surface strings for ONE class (>=1). Prototype = mean over
    all (variant × template) L2-normed text embeddings, then renormalized."""
    strs = [tp.format(v) for v in name_variants for tp in templates]
    e = encode_text(strs).mean(0)
    return torch.nn.functional.normalize(e, dim=-1)

@torch.no_grad()
def classifier(per_class_variants, templates):
    W = [prototype(v, templates) for v in per_class_variants]
    return torch.stack(W, 1).to(DEV)                          # (D, C)

@torch.no_grad()
def run(tag, per_class_variants, templates):
    clf = classifier(per_class_variants, templates)
    ds = ImageFolder(CRC, transform=transform)
    preds, ys = [], []
    for imgs, y in DataLoader(ds, batch_size=256, num_workers=8):
        ie = model.encode_image(imgs.to(DEV))
        ie = torch.nn.functional.normalize(ie.float(), dim=-1)
        preds.append((ie @ clf).argmax(1).cpu().numpy()); ys.append(np.asarray(y))
    p = np.concatenate(preds); y = np.concatenate(ys)
    print(f"KEEP crc100k [{tag}]: macro(balanced)={balanced_accuracy_score(y,p):.4f}  "
          f"micro={accuracy_score(y,p):.4f}  (n={len(y)})", flush=True)

if __name__ == "__main__":
    classes = DATASETS_TO_CLASSES["crc100k"]
    templates = DATASETS_TO_TEMPLATES["crc100k"]           # 22 CONCH templates
    synonyms = DATASETS_TO_SYNONYMS["crc100k"]             # per-class synonym lists
    # plain: single class name × 22 templates
    run("22tmpl", [[c] for c in classes], templates)
    # synonym: all class synonyms × 22 templates (our headline protocol)
    run("22tmpl+syn", synonyms, templates)
    print("KEEP_CRC_ZS_DONE", flush=True)
