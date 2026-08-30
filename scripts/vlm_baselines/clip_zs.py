import os, sys, numpy as np, torch, h5py
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, balanced_accuracy_score
sys.path.insert(0, "/home/shiwon/PAL_working")
from src.evaluation.zero_shot_metadata import DATASETS_TO_CLASSES, DATASETS_TO_TEMPLATES
dev="cuda"
CRC="/home/shiwon/STRUCTURE/data/crc100k/CRC-VAL-HE-7K"
PX="/home/shiwon/STRUCTURE/data/pcam/camelyonpatch_level_2_split_test_x.h5"
PY_="/home/shiwon/STRUCTURE/data/pcam/camelyonpatch_level_2_split_test_y.h5"
SIC="/home/shiwon/STRUCTURE/data/sicap/eval"

class PCam(Dataset):
    def __init__(self,pp): self.x=h5py.File(PX,"r")["x"]; self.y=h5py.File(PY_,"r")["y"]; self.pp=pp
    def __len__(self): return self.x.shape[0]
    def __getitem__(self,i): return self.pp(Image.fromarray(self.x[i])), int(np.asarray(self.y[i]).reshape(-1)[0])

import open_clip
m,_,pp=open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai'); m=m.to(dev).eval()
tok=open_clip.get_tokenizer('ViT-B-16')

@torch.no_grad()
def zs_clf(cn,tp):
    W=[]
    for c in cn:
        e=m.encode_text(tok([t.format(c) for t in tp]).to(dev)); e=e/e.norm(dim=-1,keepdim=True)
        e=e.mean(0); e=e/e.norm(); W.append(e)
    return torch.stack(W,1).to(dev)
@torch.no_grad()
def run(name, ds):
    clf=zs_clf(DATASETS_TO_CLASSES[name], DATASETS_TO_TEMPLATES[name]); P=[];Y=[]
    for imgs,y in DataLoader(ds,batch_size=256,num_workers=8):
        ie=m.encode_image(imgs.to(dev)); ie=ie/ie.norm(dim=-1,keepdim=True)
        P.append((ie@clf).argmax(1).cpu().numpy()); Y.append(np.asarray(y))
    p=np.concatenate(P);y=np.concatenate(Y)
    print(f"CLIP(ViT-B/16 openai) {name}: top1_micro={accuracy_score(y,p):.3f}  macro={balanced_accuracy_score(y,p):.3f}  (n={len(y)})",flush=True)

run("crc100k", ImageFolder(CRC, transform=pp))
run("pcam", PCam(pp))
run("sicap", ImageFolder(SIC, transform=pp))
print("CLIP_ZS_DONE")
