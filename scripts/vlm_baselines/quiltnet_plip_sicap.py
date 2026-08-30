import os, sys, numpy as np, torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score
sys.path.insert(0, "/home/shiwon/PAL_working")
from src.evaluation.zero_shot_metadata import DATASETS_TO_CLASSES, DATASETS_TO_TEMPLATES
dev="cuda"; SIC="/home/shiwon/STRUCTURE/data/sicap/eval"
CN=DATASETS_TO_CLASSES["sicap"]; TP=DATASETS_TO_TEMPLATES["sicap"]
@torch.no_grad()
def zs_clf(enc_txt):
    W=[]
    for c in CN:
        e=enc_txt([t.format(c) for t in TP]); e=e/e.norm(dim=-1,keepdim=True); e=e.mean(0); e=e/e.norm(); W.append(e)
    return torch.stack(W,1).to(dev)
@torch.no_grad()
def run(tag, pp, enc_img, enc_txt):
    ds=ImageFolder(SIC, transform=pp); clf=zs_clf(enc_txt); P=[];Y=[]
    for imgs,y in DataLoader(ds,batch_size=256,num_workers=8):
        ie=enc_img(imgs); ie=ie/ie.norm(dim=-1,keepdim=True)
        P.append((ie@clf).argmax(1).cpu().numpy()); Y.append(np.asarray(y))
    p=np.concatenate(P);y=np.concatenate(Y)
    print(f"{tag} sicap: top1_micro={accuracy_score(y,p):.3f}  macro={balanced_accuracy_score(y,p):.3f}  (n={len(y)})",flush=True)
try:
    import open_clip
    m,_,pp=open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32'); m=m.to(dev).eval()
    tok=open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')
    run("QuiltNet", pp, lambda x:m.encode_image(x.to(dev)), lambda t:m.encode_text(tok(t).to(dev)))
except Exception as e: print("QuiltNet FAILED:", repr(e), flush=True)
try:
    from transformers import CLIPModel, CLIPProcessor
    pm=CLIPModel.from_pretrained("vinid/plip").to(dev).eval(); proc=CLIPProcessor.from_pretrained("vinid/plip")
    def ppp(pil): return proc(images=pil,return_tensors="pt")["pixel_values"][0]
    def ei(x): return pm.get_image_features(pixel_values=x.to(dev))
    def et(t): 
        tt=proc(text=t,return_tensors="pt",padding="max_length",max_length=77,truncation=True).to(dev)
        return pm.get_text_features(**tt)
    run("PLIP", ppp, ei, et)
except Exception as e: print("PLIP FAILED:", repr(e), flush=True)
print("QN_PLIP_SICAP_DONE")
