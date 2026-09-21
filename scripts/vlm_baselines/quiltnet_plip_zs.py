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

class PCam(Dataset):
    def __init__(self,pp):
        self.x=h5py.File(PX,"r")["x"]; self.y=h5py.File(PY_,"r")["y"]; self.pp=pp
    def __len__(self): return self.x.shape[0]
    def __getitem__(self,i): return self.pp(Image.fromarray(self.x[i])), int(np.asarray(self.y[i]).reshape(-1)[0])

@torch.no_grad()
def zs_clf(classnames, templates, enc_txt):
    W=[]
    for c in classnames:
        e=enc_txt([tp.format(c) for tp in templates]); e=e/e.norm(dim=-1,keepdim=True)
        e=e.mean(0); e=e/e.norm(); W.append(e)
    return torch.stack(W,1).to(dev)

@torch.no_grad()
def run(tag,name,ds,cn,tp,enc_img,enc_txt):
    clf=zs_clf(cn,tp,enc_txt); loader=DataLoader(ds,batch_size=256,num_workers=8)
    P=[];Y=[]
    for imgs,y in loader:
        ie=enc_img(imgs); ie=ie/ie.norm(dim=-1,keepdim=True)
        P.append((ie@clf).argmax(1).cpu().numpy()); Y.append(np.asarray(y))
    p=np.concatenate(P);y=np.concatenate(Y)
    print(f"{tag} {name}: top1_micro={accuracy_score(y,p):.3f}  macro={balanced_accuracy_score(y,p):.3f}  (n={len(y)})",flush=True)

def eval_model(tag, pp, enc_img, enc_txt):
    run(tag,"crc100k",ImageFolder(CRC,transform=pp),DATASETS_TO_CLASSES["crc100k"],DATASETS_TO_TEMPLATES["crc100k"],enc_img,enc_txt)
    run(tag,"pcam",PCam(pp),DATASETS_TO_CLASSES["pcam"],DATASETS_TO_TEMPLATES["pcam"],enc_img,enc_txt)

# ---- QuiltNet (open_clip) ----
try:
    import open_clip
    m,_,pp=open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32'); m=m.to(dev).eval()
    tok=open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')
    eval_model("QuiltNet", pp, lambda x:m.encode_image(x.to(dev)), lambda t:m.encode_text(tok(t).to(dev)))
except Exception as e:
    print("QuiltNet FAILED:", repr(e), flush=True)

# ---- PLIP (transformers CLIP) ----
try:
    from transformers import CLIPModel, CLIPProcessor
    pm=CLIPModel.from_pretrained("vinid/plip").to(dev).eval()
    proc=CLIPProcessor.from_pretrained("vinid/plip")
    def ppp(pil): return proc(images=pil,return_tensors="pt")["pixel_values"][0]
    def ei(x): return pm.get_image_features(pixel_values=x.to(dev))
    def et(t):
        tt=proc(text=t,return_tensors="pt",padding=True,truncation=True).to(dev)
        return pm.get_text_features(**tt)
    eval_model("PLIP", ppp, ei, et)
except Exception as e:
    print("PLIP FAILED:", repr(e), flush=True)
print("QN_PLIP_DONE")
