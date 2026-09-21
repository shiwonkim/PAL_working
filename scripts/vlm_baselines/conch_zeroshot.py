import os, sys, numpy as np, torch, h5py
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
sys.path.insert(0, "/home/shiwon/PAL_working")
from src.evaluation.zero_shot_metadata import DATASETS_TO_CLASSES, DATASETS_TO_TEMPLATES

dev="cuda"; tok=open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
model, preprocess = create_model_from_pretrained('conch_ViT-B-16','hf_hub:MahmoodLab/conch',hf_auth_token=tok)
model=model.to(dev).eval(); tokenizer=get_tokenizer()

@torch.no_grad()
def zs_classifier(classnames, templates):
    W=[]
    for c in classnames:
        t=tokenize(texts=[tp.format(c) for tp in templates], tokenizer=tokenizer).to(dev)
        e=model.encode_text(t); e=e/e.norm(dim=-1,keepdim=True)
        e=e.mean(0); e=e/e.norm(); W.append(e)
    return torch.stack(W,1).to(dev)   # (D, C)

@torch.no_grad()
def run(name, ds, classnames, templates):
    clf=zs_classifier(classnames, templates)
    loader=DataLoader(ds, batch_size=256, num_workers=8)
    preds=[]; ys=[]
    for imgs,y in loader:
        ie=model.encode_image(imgs.to(dev), proj_contrast=True, normalize=True)
        logits=ie@clf                      # (B, C)
        preds.append(logits.argmax(1).cpu().numpy()); ys.append(np.asarray(y))
    p=np.concatenate(preds); y=np.concatenate(ys)
    print(f"CONCH {name}: top1_micro={accuracy_score(y,p):.3f}  top1_macro(balanced)={balanced_accuracy_score(y,p):.3f}  (n={len(y)})", flush=True)

# crc100k (ImageFolder ADI..TUM = 0..8, DATASETS_TO_CLASSES 순서 동일)
crc=ImageFolder("/home/shiwon/STRUCTURE/data/crc100k/CRC-VAL-HE-7K", transform=preprocess)
run("crc100k", crc, DATASETS_TO_CLASSES["crc100k"], DATASETS_TO_TEMPLATES["crc100k"])

# pcam (h5: x (N,96,96,3), y (N,1,1,1); 0=normal,1=tumor)
class PCam(Dataset):
    def __init__(self):
        self.x=h5py.File("/home/shiwon/STRUCTURE/data/pcam/camelyonpatch_level_2_split_test_x.h5","r")["x"]
        self.y=h5py.File("/home/shiwon/STRUCTURE/data/pcam/camelyonpatch_level_2_split_test_y.h5","r")["y"]
    def __len__(self): return self.x.shape[0]
    def __getitem__(self,i):
        return preprocess(Image.fromarray(self.x[i])), int(np.asarray(self.y[i]).reshape(-1)[0])
run("pcam", PCam(), DATASETS_TO_CLASSES["pcam"], DATASETS_TO_TEMPLATES["pcam"])
print("CONCH_ZS_DONE")
