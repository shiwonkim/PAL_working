import os, numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

SEL="/home/shiwon/STRUCTURE/data/quilt1m/selections/quilt_openpath-quilt_full_seed42.csv"
IMG="/home/shiwon/STRUCTURE/data/quilt1m/images"
OUT="/home/shiwon/STRUCTURE/data/quilt1m/selections/conch_scores_full.csv"
dev="cuda"
tok_hf=open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()

print("loading CONCH...", flush=True)
model, preprocess = create_model_from_pretrained('conch_ViT-B-16','hf_hub:MahmoodLab/conch',hf_auth_token=tok_hf)
model=model.to(dev).eval(); tokenizer=get_tokenizer()

df=pd.read_csv(SEL); N=len(df)
caps=df['caption'].astype(str).tolist()
paths=[os.path.join(IMG, os.path.basename(p)) for p in df['image_path']]
print(f"pairs: {N}", flush=True)

# --- text embeddings (batched) ---
print("encoding text...", flush=True)
te_all=torch.zeros(N,512)
with torch.no_grad():
    for i in range(0,N,512):
        t=tokenize(texts=caps[i:i+512], tokenizer=tokenizer).to(dev)
        e=model.encode_text(t); e=e/e.norm(dim=-1,keepdim=True)
        te_all[i:i+len(e)]=e.cpu()

# --- image embeddings + cosine (batched) ---
class DS(Dataset):
    def __len__(self): return N
    def __getitem__(self,i):
        try: img=preprocess(Image.open(paths[i]).convert('RGB'))
        except Exception: img=torch.zeros(3,224,224)
        return img,i
loader=DataLoader(DS(),batch_size=256,num_workers=8,pin_memory=True)
scores=np.full(N,np.nan,dtype=np.float32)
print("encoding images + cosine...", flush=True)
done=0
with torch.no_grad():
    for imgs,idx in loader:
        ie=model.encode_image(imgs.to(dev),proj_contrast=True,normalize=True).cpu()
        te=te_all[idx]
        scores[idx.numpy()]=(ie*te).sum(-1).numpy()
        done+=len(idx)
        if done % 5120 < 256: print(f"  {done}/{N}", flush=True)

df['conch_score']=scores
df.to_csv(OUT,index=False)
s=pd.Series(scores).dropna()
print("\n===== CONCH score 분포 (n=%d) =====" % len(s))
for q in [0,1,5,10,25,50,75,90,95,99,100]:
    print(f"  p{q:>3}: {np.percentile(s,q):.4f}")
print(f"  mean {s.mean():.4f}  std {s.std():.4f}")
print("\n소스별 평균:")
for src in df['source'].unique():
    ss=df[df['source']==src]['conch_score'].dropna()
    print(f"  {src}: mean {ss.mean():.4f}  median {ss.median():.4f}  n={len(ss)}")
print("\n=== 최고 스코어 8 ===")
for _,r in df.nlargest(8,'conch_score').iterrows(): print(f"  {r['conch_score']:.3f} [{r['source']}] {str(r['caption'])[:90]}")
print("\n=== 최저 스코어 8 ===")
for _,r in df.nsmallest(8,'conch_score').iterrows(): print(f"  {r['conch_score']:.3f} [{r['source']}] {str(r['caption'])[:90]}")
print("\nSCORE_DONE ->", OUT)
