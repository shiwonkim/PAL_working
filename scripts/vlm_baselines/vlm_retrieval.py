import os, sys, numpy as np, torch, pandas as pd
from PIL import Image
sys.path.insert(0,"/home/shiwon/PAL_working")
dev="cuda"; tok_hf=open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
SEL="/home/shiwon/STRUCTURE/data/quilt1m/selections/quilt_openpath-quilt_val_full_seed42.csv"
IMG="/home/shiwon/STRUCTURE/data/quilt1m/images"
df=pd.read_csv(SEL); caps=df['caption'].astype(str).tolist()
paths=[os.path.join(IMG,os.path.basename(p)) for p in df['image_path']]; N=len(df)

def recalls(sim):  # sim (N_img, N_txt), diagonal = positive
    out={}
    # I2T: each row (image) rank texts
    r=(-sim).argsort(1); gt=np.arange(N)[:,None]
    rank_i2t=(r==gt).argmax(1)
    # T2I: each col (text) rank images
    c=(-sim.T).argsort(1); rank_t2i=(c==gt).argmax(1)
    for k in (1,5,10):
        out[f"I2T-R@{k}"]=float((rank_i2t<k).mean()); out[f"T2I-R@{k}"]=float((rank_t2i<k).mean())
    return out

@torch.no_grad()
def encode(name, enc_img, enc_txt, preprocess, batch=256):
    from torch.utils.data import Dataset, DataLoader
    class DS(Dataset):
        def __len__(s): return N
        def __getitem__(s,i):
            try: return preprocess(Image.open(paths[i]).convert('RGB')),i
            except: return torch.zeros(3,224,224),i
    IE=torch.zeros(N,512)
    for imgs,idx in DataLoader(DS(),batch_size=batch,num_workers=8):
        e=enc_img(imgs); IE[idx]=(e/e.norm(dim=-1,keepdim=True)).cpu()
    TE=torch.zeros(N,512)
    for i in range(0,N,batch):
        e=enc_txt(caps[i:i+batch]); TE[i:i+len(e)]=(e/e.norm(dim=-1,keepdim=True)).cpu()
    sim=(IE@TE.T).numpy()
    r=recalls(sim)
    print(f"{name}: I2T R@1/5/10={r['I2T-R@1']:.3f}/{r['I2T-R@5']:.3f}/{r['I2T-R@10']:.3f}  T2I R@1/5/10={r['T2I-R@1']:.3f}/{r['T2I-R@5']:.3f}/{r['T2I-R@10']:.3f}",flush=True)

# CONCH
try:
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
    m,pp=create_model_from_pretrained('conch_ViT-B-16','hf_hub:MahmoodLab/conch',hf_auth_token=tok_hf); m=m.to(dev).eval(); tk=get_tokenizer()
    encode("CONCH", lambda x:m.encode_image(x.to(dev),proj_contrast=True,normalize=True),
           lambda t:m.encode_text(tokenize(texts=t,tokenizer=tk).to(dev)), pp)
except Exception as e: print("CONCH FAIL",repr(e),flush=True)
# QuiltNet
try:
    import open_clip
    m,_,pp=open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32'); m=m.to(dev).eval(); tk=open_clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')
    encode("QuiltNet", lambda x:m.encode_image(x.to(dev)), lambda t:m.encode_text(tk(t).to(dev)), pp)
except Exception as e: print("QuiltNet FAIL",repr(e),flush=True)
# PLIP
try:
    from transformers import CLIPModel, CLIPProcessor
    m=CLIPModel.from_pretrained("vinid/plip").to(dev).eval(); proc=CLIPProcessor.from_pretrained("vinid/plip")
    encode("PLIP", lambda x:m.get_image_features(pixel_values=x.to(dev)),
           lambda t:m.get_text_features(**proc(text=t,return_tensors="pt",padding=True,truncation=True).to(dev)),
           lambda pil: proc(images=pil,return_tensors="pt")["pixel_values"][0])
except Exception as e: print("PLIP FAIL",repr(e),flush=True)
print("VLM_RT_DONE")
