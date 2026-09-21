"""Re-evaluate VLM baselines with OFFICIAL prompt ensembles.
CRC100K: CONCH's official prompts (crc100k_prompts_all_per_class.json) --
  per-class synonyms x 22 templates, prototype = mean over all.
SICAP: same 22 templates + CONCH-style Gleason synonyms (no official file).
Same ensemble applied to every model (dataset-specific, model-agnostic)."""
import json, sys, numpy as np, torch, h5py
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, balanced_accuracy_score
dev = "cuda"
CRC = "/home/shiwon/STRUCTURE/data/crc100k/CRC-VAL-HE-7K"
SIC = "/home/shiwon/STRUCTURE/data/sicap/eval"

crc = json.load(open("/home/shiwon/conch_crc100k_prompts.json"))["0"]
TEMPLATES = crc["templates"]                       # 22 templates w/ "CLASSNAME"
CRC_CN = crc["classnames"]                          # dict: ADI..TUM -> [synonyms]
SIC_CN = json.load(open("/home/shiwon/sicap_prompts.json"))["0"]["classnames"]

def prompts_for(classnames_dict):
    """Return (class_order, [ [prompt,...] per class ]) using synonyms x templates."""
    order = list(classnames_dict.keys())            # already ImageFolder-alphabetical
    per_class = []
    for c in order:
        pl = [t.replace("CLASSNAME", s) for s in classnames_dict[c] for t in TEMPLATES]
        per_class.append(pl)
    return order, per_class

@torch.no_grad()
def build_protos(per_class, enc_txt):
    W = []
    for pl in per_class:
        e = enc_txt(pl); e = e / e.norm(dim=-1, keepdim=True)
        p = e.mean(0); p = p / p.norm(); W.append(p)
    return torch.stack(W, 1).to(dev)                # (D, C)

class PCamNone: pass  # (PCam omitted per request)

@torch.no_grad()
def classify(ds, protos, enc_img):
    P, Y = [], []
    for imgs, y in DataLoader(ds, batch_size=256, num_workers=8):
        ie = enc_img(imgs); ie = ie / ie.norm(dim=-1, keepdim=True)
        P.append((ie @ protos).argmax(1).cpu().numpy()); Y.append(np.asarray(y))
    p = np.concatenate(P); y = np.concatenate(Y)
    return accuracy_score(y, p), balanced_accuracy_score(y, p), len(y)

def run_model(tag, pp, enc_img, enc_txt):
    for name, root, cn in [("crc100k", CRC, CRC_CN), ("sicap", SIC, SIC_CN)]:
        _, per_class = prompts_for(cn)
        protos = build_protos(per_class, enc_txt)
        ds = ImageFolder(root, transform=pp)
        mic, mac, n = classify(ds, protos, enc_img)
        print(f"{tag} {name}: micro={mic:.3f}  macro={mac:.3f}  (n={n})", flush=True)

# ---- CONCH ----
try:
    import os
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
    tokh = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    m, pp = create_model_from_pretrained('conch_ViT-B-16', 'hf_hub:MahmoodLab/conch', hf_auth_token=tokh)
    m = m.to(dev).eval(); tk = get_tokenizer()
    def et(pl): return m.encode_text(tokenize(texts=pl, tokenizer=tk).to(dev))
    def ei(x): return m.encode_image(x.to(dev), proj_contrast=True, normalize=True)
    run_model("CONCH", pp, ei, et)
except Exception as e: print("CONCH FAILED:", repr(e), flush=True)

# ---- QuiltNet ----
try:
    import open_clip
    m, _, pp = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32'); m = m.to(dev).eval()
    tk = open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')
    run_model("QuiltNet", pp, lambda x: m.encode_image(x.to(dev)), lambda pl: m.encode_text(tk(pl).to(dev)))
except Exception as e: print("QuiltNet FAILED:", repr(e), flush=True)

# ---- CLIP (natural) ----
try:
    import open_clip
    m, _, pp = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai'); m = m.to(dev).eval()
    tk = open_clip.get_tokenizer('ViT-B-16')
    run_model("CLIP", pp, lambda x: m.encode_image(x.to(dev)), lambda pl: m.encode_text(tk(pl).to(dev)))
except Exception as e: print("CLIP FAILED:", repr(e), flush=True)

# ---- PLIP ----
try:
    from transformers import CLIPModel, CLIPProcessor
    pm = CLIPModel.from_pretrained("vinid/plip").to(dev).eval(); proc = CLIPProcessor.from_pretrained("vinid/plip")
    def ppp(pil): return proc(images=pil, return_tensors="pt")["pixel_values"][0]
    def ei(x): return pm.get_image_features(pixel_values=x.to(dev))
    def et(pl):
        tt = proc(text=pl, return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(dev)
        return pm.get_text_features(**tt)
    run_model("PLIP", ppp, ei, et)
except Exception as e: print("PLIP FAILED:", repr(e), flush=True)
print("VLM_OFFICIAL_DONE")
