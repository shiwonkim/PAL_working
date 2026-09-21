import os, numpy as np, torch, pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
dev="cuda"
SEL="/home/shiwon/STRUCTURE/data/quilt1m/selections/quilt_openpath-quilt_val_full_seed42.csv"
IMG="/home/shiwon/STRUCTURE/data/quilt1m/images"
df=pd.read_csv(SEL); caps=df['caption'].astype(str).tolist()
paths=[os.path.join(IMG,os.path.basename(p)) for p in df['image_path']]; N=len(df)

def recalls(sim):
    out={}; gt=np.arange(N)[:,None]
    rank_i2t=((-sim).argsort(1)==gt).argmax(1)
    rank_t2i=((-sim.T).argsort(1)==gt).argmax(1)
    for k in (1,5,10):
        out[f"I2T-R@{k}"]=float((rank_i2t<k).mean()); out[f"T2I-R@{k}"]=float((rank_t2i<k).mean())
    return out

m=CLIPModel.from_pretrained("vinid/plip").to(dev).eval()
proc=CLIPProcessor.from_pretrained("vinid/plip")
class DS(Dataset):
    def __len__(s): return N
    def __getitem__(s,i):
        try: return proc(images=Image.open(paths[i]).convert('RGB'),return_tensors="pt")["pixel_values"][0],i
        except: return torch.zeros(3,224,224),i
IE=torch.zeros(N,512)
with torch.no_grad():
    for imgs,idx in DataLoader(DS(),batch_size=256,num_workers=8):
        e=m.get_image_features(pixel_values=imgs.to(dev)); IE[idx]=(e/e.norm(dim=-1,keepdim=True)).cpu()
    TE=torch.zeros(N,512)
    for i in range(0,N,256):
        tt=proc(text=caps[i:i+256],return_tensors="pt",padding="max_length",max_length=77,truncation=True).to(dev)
        e=m.get_text_features(**tt); TE[i:i+len(e)]=(e/e.norm(dim=-1,keepdim=True)).cpu()
sim=(IE@TE.T).numpy(); r=recalls(sim)
print(f"PLIP: I2T R@1/5/10={r['I2T-R@1']:.3f}/{r['I2T-R@5']:.3f}/{r['I2T-R@10']:.3f}  T2I R@1/5/10={r['T2I-R@1']:.3f}/{r['T2I-R@5']:.3f}/{r['T2I-R@10']:.3f}")
print("PLIP_RT_DONE")
