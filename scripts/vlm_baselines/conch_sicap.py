import os, sys, numpy as np, torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
sys.path.insert(0, "/home/shiwon/PAL_working")
from src.evaluation.zero_shot_metadata import DATASETS_TO_CLASSES, DATASETS_TO_TEMPLATES
dev="cuda"; tokh=open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
model, pp = create_model_from_pretrained('conch_ViT-B-16','hf_hub:MahmoodLab/conch',hf_auth_token=tokh)
model=model.to(dev).eval(); tokenizer=get_tokenizer()
@torch.no_grad()
def zs(cn,tp):
    W=[]
    for c in cn:
        t=tokenize(texts=[x.format(c) for x in tp], tokenizer=tokenizer).to(dev)
        e=model.encode_text(t); e=e/e.norm(dim=-1,keepdim=True); e=e.mean(0); e=e/e.norm(); W.append(e)
    return torch.stack(W,1).to(dev)
ds=ImageFolder("/home/shiwon/STRUCTURE/data/sicap/eval", transform=pp)
clf=zs(DATASETS_TO_CLASSES["sicap"], DATASETS_TO_TEMPLATES["sicap"])
P=[];Y=[]
with torch.no_grad():
    for imgs,y in DataLoader(ds,batch_size=256,num_workers=8):
        ie=model.encode_image(imgs.to(dev),proj_contrast=True,normalize=True)
        P.append((ie@clf).argmax(1).cpu().numpy()); Y.append(np.asarray(y))
p=np.concatenate(P);y=np.concatenate(Y)
print(f"CONCH sicap: top1_micro={accuracy_score(y,p):.3f}  macro={balanced_accuracy_score(y,p):.3f}  (n={len(y)})")
print("CONCH_SICAP_DONE")
