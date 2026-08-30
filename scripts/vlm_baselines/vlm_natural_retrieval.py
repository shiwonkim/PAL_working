"""Natural-image retrieval UPPER BOUNDS: CLIP + BLIP (ITC only).

Reviewer asked for web-scale VLMs (CLIP/BLIP) as upper bounds on the natural-image
retrieval track. This reports I2T/T2I R@{1,5,10} on flickr30k-test and coco_karpathy-test
using the SAME protocol as PAL's `evaluate_retrieval`:

  * per-caption rows (image duplicated once per caption) — evaluation.drop_duplicates=false
  * I2T ground truth = all rows sharing an image_path  (retrieval_metrics_df)
  * T2I gallery      = unique images, one relevant per caption (text_to_image_retrieval_metrics)

Both helpers are copied verbatim from src/evaluation/retrieval.py so results are
byte-identical to the pipeline (minus the frozen-encoder+alignment vs full-VLM difference).
BLIP is ITC-only (image/text projection cosine) = same dual-encoder condition as CLIP/PAL,
NOT the ITM cross-attention rerank.

Run in the `vlm_eval` conda env (open_clip + transformers), NOT `structure`.
    CUDA_VISIBLE_DEVICES=0 python vlm_natural_retrieval.py [clip|blip|both]
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

DEV = "cuda"
DATA = "/home/shiwon/STRUCTURE/data"

# ─────────────────────────── retrieval helpers (verbatim copy) ───────────────────────────
def _norm(x):  # explicit L2; embeddings are already normalized but keep idempotent
    return torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-12)

def compute_ground_truth_mapping(df, image_column="image_path"):
    groups = df.groupby(image_column).groups
    return {idx: set(groups[df.loc[idx, image_column]]) for idx in df.index}

def retrieval_metrics_df(image_embeds, text_embeds, df, image_column="image_path",
                         k_values=(1, 5, 10), batch_size=256):
    image_embeds = _norm(image_embeds); text_embeds = _norm(text_embeds)
    N = image_embeds.size(0)
    ground_truth = compute_ground_truth_mapping(df, image_column)
    assert all(len(v) for v in ground_truth.values()), "query with no GT!"
    max_k = max(k_values)
    recall_hits = {k: 0.0 for k in k_values}
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sims = image_embeds[start:end] @ text_embeds.T
        _, topk = torch.topk(sims, k=max_k, dim=1)
        for row, ranked_tensor in enumerate(topk):
            gt_set = ground_truth[start + row]
            ranked = ranked_tensor.tolist()
            rel_flags = [idx in gt_set for idx in ranked]
            for k in k_values:
                if sum(rel_flags[:k]) > 0:
                    recall_hits[k] += 1
    return {f"R@{k}": recall_hits[k] / N for k in k_values}

def text_to_image_retrieval_metrics(text_embeds, image_embeds, df, image_column="image_path",
                                    k_values=(1, 5, 10), batch_size=256):
    text_embeds = _norm(text_embeds); image_embeds = _norm(image_embeds)
    dfr = df.reset_index(drop=True)
    first_rows = dfr.drop_duplicates(subset=image_column).index.tolist()
    gallery = image_embeds[first_rows]
    path_to_gid = {dfr.loc[ri, image_column]: gi for gi, ri in enumerate(first_rows)}
    query_gid = [path_to_gid[dfr.loc[i, image_column]] for i in range(len(dfr))]
    N = text_embeds.size(0); M = gallery.size(0)
    max_k = min(max(k_values), M)
    recall_hits = {k: 0.0 for k in k_values}
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sims = text_embeds[start:end] @ gallery.T
        _, topk = torch.topk(sims, k=max_k, dim=1)
        for row, ranked in enumerate(topk.tolist()):
            gt = query_gid[start + row]
            rel_flags = [g == gt for g in ranked]
            for k in k_values:
                if sum(rel_flags[:k]) > 0:
                    recall_hits[k] += 1
    return {f"R@{k}": recall_hits[k] / N for k in k_values}

# ─────────────────────────── df builders (mirror dataset classes) ───────────────────────────
def flickr_df():
    root = os.path.join(DATA, "flickr30k")
    df = pd.read_csv(os.path.join(root, "results.csv"), delimiter="|")
    df.columns = [c.strip() for c in df.columns]
    split_ann = []
    for split in ["train", "val", "test"]:
        with open(os.path.join(root, f"{split}.txt")) as f:
            split_ann += [(l.rstrip(), split) for l in f]
    ds = pd.DataFrame(split_ann, columns=["image_name", "split"])
    ds["image_name"] = ds["image_name"] + ".jpg"
    ds["image_path"] = ds["image_name"].apply(lambda x: os.path.join(root, "images", x))
    df = df.merge(ds, on="image_name", how="left")
    df.dropna(subset="comment", inplace=True)
    df = df[df["split"] == "test"].reset_index(drop=True)
    df = df.rename(columns={"comment": "caption"})
    df["caption"] = df["caption"].astype(str).str.strip()
    return df[["image_path", "caption"]]

def coco_karpathy_df():
    coco = os.path.join(DATA, "COCO")
    anno = os.path.join(coco, "annotations", "captions_val2014.json")
    img_dir = os.path.join(coco, "val2014")
    with open(anno) as f:
        data = json.load(f)
    idmap = {im["id"]: im["file_name"] for im in data["images"]}
    rows = [(os.path.join(img_dir, idmap[a["image_id"]]), a["caption"])
            for a in data["annotations"] if a["image_id"] in idmap]
    df = pd.DataFrame(rows, columns=["image_path", "caption"]).dropna(subset="caption")
    with open(os.path.join(coco, "karpathy_test_ids.json")) as f:
        test_ids = set(json.load(f))
    keep = df["image_path"].apply(
        lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]) in test_ids)
    return df[keep].reset_index(drop=True)[["image_path", "caption"]]

# ─────────────────────────── encoders ───────────────────────────
class ImgDS(Dataset):
    def __init__(self, paths, pp): self.paths = paths; self.pp = pp
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try: return self.pp(Image.open(self.paths[i]).convert("RGB")), i
        except Exception: return torch.zeros(3, 224, 224), i

@torch.no_grad()
def embed_rows(df, enc_img, enc_txt, pp, dim, batch=256):
    """Return (image_embeds, text_embeds) aligned to df rows; image encoded once per unique path."""
    uniq = list(dict.fromkeys(df["image_path"].tolist()))
    ie = torch.zeros(len(uniq), dim)
    for imgs, idx in DataLoader(ImgDS(uniq, pp), batch_size=batch, num_workers=8):
        e = enc_img(imgs.to(DEV)); ie[idx] = torch.nn.functional.normalize(e, dim=-1).cpu()
    p2e = {p: ie[i] for i, p in enumerate(uniq)}
    img_rows = torch.stack([p2e[p] for p in df["image_path"]])
    caps = df["caption"].astype(str).tolist()
    te = torch.zeros(len(caps), dim)
    for i in range(0, len(caps), batch):
        e = enc_txt(caps[i:i + batch]); te[i:i + len(e)] = torch.nn.functional.normalize(e, dim=-1).cpu()
    return img_rows, te

def report(name, dsname, df, img_e, txt_e):
    i2t = retrieval_metrics_df(img_e, txt_e, df)
    t2i = text_to_image_retrieval_metrics(txt_e, img_e, df)
    print(f"{name:6s} {dsname:14s} | I2T R@1/5/10 {i2t['R@1']:.3f}/{i2t['R@5']:.3f}/{i2t['R@10']:.3f}"
          f"  T2I R@1/5/10 {t2i['R@1']:.3f}/{t2i['R@5']:.3f}/{t2i['R@10']:.3f}", flush=True)

def run_clip(dfs, arch="ViT-B-16", tag="CLIP", dim=512):
    import open_clip
    # openai weights use QuickGELU; force it so the arch matches the checkpoint.
    m, _, pp = open_clip.create_model_and_transforms(
        arch, pretrained="openai", force_quick_gelu=True)
    m = m.to(DEV).eval(); tok = open_clip.get_tokenizer(arch)
    ei = lambda x: m.encode_image(x)
    et = lambda t: m.encode_text(tok(t).to(DEV))
    for dn, df in dfs: report(tag, dn, df, *embed_rows(df, ei, et, pp, dim))

def run_blip(dfs, ck="Salesforce/blip-itm-base-coco", tag="BLIP"):
    from transformers import BlipForImageTextRetrieval, BlipProcessor
    proc = BlipProcessor.from_pretrained(ck)
    m = BlipForImageTextRetrieval.from_pretrained(ck).to(DEV).eval()
    def ei(x):  # ITC image feature = normalize(vision_proj(cls))
        v = m.vision_model(pixel_values=x)[0]
        return m.vision_proj(v[:, 0, :])
    def et(tl):  # ITC text feature = normalize(text_proj(cls)), text-only mode
        enc = proc(text=list(tl), return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEV)
        t = m.text_encoder(input_ids=enc.input_ids, attention_mask=enc.attention_mask)[0]
        return m.text_proj(t[:, 0, :])
    pp = lambda pil: proc(images=pil, return_tensors="pt")["pixel_values"][0]
    dim = m.text_proj.out_features
    for dn, df in dfs: report(tag, dn, df, *embed_rows(df, ei, et, pp, dim))

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    dfs = [("flickr30", flickr_df()), ("coco_karpathy", coco_karpathy_df())]
    for dn, df in dfs:
        print(f"[{dn}] rows={len(df)} unique_images={df['image_path'].nunique()}", flush=True)
    if which in ("clip", "both"): run_clip(dfs)
    if which in ("blip", "both", "blip_coco"):
        run_blip(dfs, "Salesforce/blip-itm-base-coco", "BLIP-coco")   # COCO-finetuned
    if which in ("blip_base", "both"):
        run_blip(dfs, "Salesforce/blip-itm-base", "BLIP-base")        # pretrain only (zero-shot)
    print("VLM_NATURAL_RT_DONE", flush=True)
