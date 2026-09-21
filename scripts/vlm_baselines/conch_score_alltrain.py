"""CONCH-score the ENTIRE Quilt-1M train split (no metadata filter).
Image-side dedup: encode each unique image once, map back to all its rows.
Output: conch_scores_alltrain.csv (lookup rows + conch_score)."""
import os, numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

QP  = "/home/shiwon/STRUCTURE/data/quilt1m"
LK  = f"{QP}/quilt_1M_lookup.csv"
IMG = f"{QP}/images"
OUT = f"{QP}/selections/conch_scores_alltrain.csv"
dev = "cuda"
tok_hf = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()

print("loading CONCH...", flush=True)
model, preprocess = create_model_from_pretrained('conch_ViT-B-16', 'hf_hub:MahmoodLab/conch', hf_auth_token=tok_hf)
model = model.to(dev).eval(); tokenizer = get_tokenizer()

df = pd.read_csv(LK)
df = df[df["split"] == "train"].reset_index(drop=True)
N = len(df)
caps = df["caption"].astype(str).tolist()
df["_base"] = df["image_path"].apply(lambda p: os.path.basename(str(p)))
print(f"train pairs: {N}", flush=True)

# --- text embeddings (batched, per-row) ---
print("encoding text...", flush=True)
te_all = torch.zeros(N, 512)
with torch.no_grad():
    for i in range(0, N, 512):
        t = tokenize(texts=caps[i:i+512], tokenizer=tokenizer).to(dev)
        e = model.encode_text(t); e = e / e.norm(dim=-1, keepdim=True)
        te_all[i:i+len(e)] = e.cpu()
        if i % 51200 == 0: print(f"  text {i}/{N}", flush=True)

# --- image dedup: unique images only ---
uniq = df["_base"].drop_duplicates().reset_index(drop=True)
U = len(uniq)
pos = {b: k for k, b in enumerate(uniq)}
row_to_uniq = df["_base"].map(pos).values
print(f"unique images: {U} (vs {N} rows -> {(1-U/N)*100:.1f}% fewer image forwards)", flush=True)

class DS(Dataset):
    def __len__(self): return U
    def __getitem__(self, k):
        try: img = preprocess(Image.open(os.path.join(IMG, uniq[k])).convert('RGB'))
        except Exception: img = torch.zeros(3, 224, 224)
        return img, k
loader = DataLoader(DS(), batch_size=256, num_workers=12, pin_memory=True)

ie_all = torch.zeros(U, 512)
print("encoding images (unique)...", flush=True)
done = 0
with torch.no_grad():
    for imgs, idx in loader:
        ie = model.encode_image(imgs.to(dev), proj_contrast=True, normalize=True).cpu()
        ie_all[idx] = ie
        done += len(idx)
        if done % 25600 < 256: print(f"  img {done}/{U}", flush=True)

# --- cosine per row (row text . its image) ---
ie_rows = ie_all[torch.tensor(row_to_uniq, dtype=torch.long)]
scores = (ie_rows * te_all).sum(-1).numpy().astype(np.float32)
df = df.drop(columns=["_base"])
df["conch_score"] = scores
df.to_csv(OUT, index=False)

s = pd.Series(scores)
print("\n===== CONCH score 분포 (n=%d) =====" % len(s))
for q in [0,1,5,10,25,50,75,90,95,99,100]:
    print(f"  p{q:>3}: {np.percentile(s,q):.4f}")
print(f"  mean {s.mean():.4f}  std {s.std():.4f}")
# 어떤 임계에서 100K가 걸리나
thr100k = np.partition(scores, N-100000)[N-100000]
print(f"\n  top-100K score 컷: {thr100k:.4f}  (이 이상인 쌍 {int((scores>=thr100k).sum())})")
print("SCORE_DONE ->", OUT, flush=True)
