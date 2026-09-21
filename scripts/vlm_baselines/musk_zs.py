"""MUSK (Nature 2025) zero-shot classification on CRC100K + SICAP, matching the
other VLM baselines: 22 CONCH templates x {single class name / per-class synonyms}.
Runs in the `musk` conda env (torch 2.0.1 / timm 0.9.8)."""
import os, sys, json, numpy as np, torch, torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models import create_model
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from transformers import XLMRobertaTokenizer
sys.path.insert(0, "/home/shiwon/PAL_working")
from src.evaluation.zero_shot_metadata import DATASETS_TO_CLASSES, DATASETS_TO_SYNONYMS
from musk import utils, modeling  # noqa: F401 (registers musk_large_patch16_384)

dev = "cuda"
CRC = "/home/shiwon/STRUCTURE/data/crc100k/CRC-VAL-HE-7K"
SIC = "/home/shiwon/STRUCTURE/data/sicap/eval"
TEMPLATES = [t.replace("CLASSNAME", "{}") for t in
             json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "conch_crc100k_prompts.json")))["0"]["templates"]]
SPM = "/home/shiwon/MUSK/musk/models/tokenizer.spm"
if not os.path.exists(SPM):
    from huggingface_hub import hf_hub_download
    SPM = hf_hub_download("xiangjx/musk", "tokenizer.spm")

# ---- load model ----
model = create_model("musk_large_patch16_384")
utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')
model = model.to(dev, dtype=torch.float16).eval()
tokenizer = XLMRobertaTokenizer(SPM)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(384, interpolation=3, antialias=True),
    torchvision.transforms.CenterCrop((384, 384)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
])

@torch.no_grad()
def enc_txt(texts):
    embs = []
    for i in range(0, len(texts), 256):
        chunk = texts[i:i+256]
        ids, pads = [], []
        for t in chunk:
            tid, pad = utils.xlm_tokenizer(t, tokenizer, max_len=100)
            ids.append(tid); pads.append(pad)
        ids = torch.tensor(ids).to(dev); pads = torch.tensor(pads).to(dev)
        e = model(text_description=ids, padding_mask=pads, with_head=True,
                  out_norm=True, return_global=True)[1]
        embs.append(e.float().cpu())
    return torch.cat(embs)

@torch.no_grad()
def enc_img(x):
    return model(image=x.to(dev, dtype=torch.float16), with_head=True, out_norm=True,
                 ms_aug=False, return_global=True)[0].float()

def protos_common(cn):
    W = []
    for c in cn:
        e = enc_txt([t.format(c) for t in TEMPLATES]); e = e / e.norm(dim=-1, keepdim=True)
        p = e.mean(0); W.append(p / p.norm())
    return torch.stack(W, 1).to(dev)

def protos_syn(groups):
    W = []
    for syns in groups:
        pl = [t.format(s) for s in syns for t in TEMPLATES]
        e = enc_txt(pl); e = e / e.norm(dim=-1, keepdim=True)
        p = e.mean(0); W.append(p / p.norm())
    return torch.stack(W, 1).to(dev)

@torch.no_grad()
def classify(root, protos):
    ds = ImageFolder(root, transform=transform); P, Y = [], []
    for imgs, y in DataLoader(ds, batch_size=64, num_workers=8):
        ie = enc_img(imgs); ie = ie / ie.norm(dim=-1, keepdim=True)
        P.append((ie @ protos).argmax(1).cpu().numpy()); Y.append(np.asarray(y))
    p = np.concatenate(P); y = np.concatenate(Y)
    return accuracy_score(y, p), balanced_accuracy_score(y, p), len(y)

for name, root in [("crc100k", CRC), ("sicap", SIC)]:
    pc = protos_common(DATASETS_TO_CLASSES[name])
    mic, mac, n = classify(root, pc)
    print(f"MUSK {name} [common]:  micro={mic:.3f}  macro={mac:.3f}  (n={n})", flush=True)
    ps = protos_syn(DATASETS_TO_SYNONYMS[name])
    mic, mac, n = classify(root, ps)
    print(f"MUSK {name} [synonym]: micro={mic:.3f}  macro={mac:.3f}  (n={n})", flush=True)
print("MUSK_ZS_DONE")
