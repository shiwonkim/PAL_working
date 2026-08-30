"""PathCap-val IN-DOMAIN retrieval: PAL vs pathology VLMs (CONCH / QuiltNet / PLIP).

Same protocol as the PAL pipeline's evaluate_retrieval: reuses the exact helper functions
(retrieval_metrics_df / text_to_image_retrieval_metrics) via vlm_natural_retrieval, on the
SAME PathCap-val df (1,500 images, 1 caption each -> clean 1:1 retrieval). So VLM numbers are
directly comparable to the PAL number from `python -m src.eval --rt pathcap`.

PathCap is IN-DOMAIN for PAL (trained on pathcap train) and OUT-OF-DOMAIN transfer for the VLMs
(CONCH/QuiltNet/PLIP were trained on other pathology corpora) — the mirror image of Quilt-val.

Run in vlm_eval env.
    CUDA_VISIBLE_DEVICES=<g> python vlm_pathcap_retrieval.py
"""
import os, sys
import pandas as pd, torch
sys.path.insert(0, "/home/shiwon/PAL_working/scripts/vlm_baselines")
from vlm_natural_retrieval import embed_rows, report  # shared helpers + protocol

DEV = "cuda"
DATA = "/home/shiwon/STRUCTURE/data/pathcap"
TOKH = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()

def pathcap_df():
    df = pd.read_csv(f"{DATA}/selections/pathcap_val_le128_seed42.csv")
    df["image_path"] = df["image_path"].apply(
        lambda p: os.path.join(DATA, "images", os.path.basename(str(p))))
    df["caption"] = df["caption"].astype(str)
    return df[["image_path", "caption"]]

def run_conch(df):
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
    m, pp = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch", hf_auth_token=TOKH)
    m = m.to(DEV).eval(); tk = get_tokenizer()
    ei = lambda x: m.encode_image(x.to(DEV), proj_contrast=True, normalize=True)
    et = lambda t: m.encode_text(tokenize(texts=list(t), tokenizer=tk).to(DEV))
    report("CONCH", "pathcap_val", df, *embed_rows(df, ei, et, pp, 512))

def run_quiltnet(df):
    import open_clip
    m, _, pp = open_clip.create_model_and_transforms("hf-hub:wisdomik/QuiltNet-B-32")
    m = m.to(DEV).eval(); tk = open_clip.get_tokenizer("hf-hub:wisdomik/QuiltNet-B-32")
    ei = lambda x: m.encode_image(x.to(DEV))
    et = lambda t: m.encode_text(tk(list(t)).to(DEV))
    report("QuiltNet", "pathcap_val", df, *embed_rows(df, ei, et, pp, 512))

def run_plip(df):
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained("vinid/plip").to(DEV).eval()
    proc = CLIPProcessor.from_pretrained("vinid/plip")
    ei = lambda x: m.get_image_features(pixel_values=x.to(DEV))
    et = lambda t: m.get_text_features(
        **proc(text=list(t), return_tensors="pt", padding="max_length",
               truncation=True, max_length=77).to(DEV))
    pp = lambda pil: proc(images=pil, return_tensors="pt")["pixel_values"][0]
    report("PLIP", "pathcap_val", df, *embed_rows(df, ei, et, pp, 512))

if __name__ == "__main__":
    df = pathcap_df()
    print(f"[pathcap_val] rows={len(df)} unique_images={df['image_path'].nunique()}", flush=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("conch", run_conch), ("quiltnet", run_quiltnet), ("plip", run_plip)]:
        if which in (name, "all"):
            try: fn(df)
            except Exception as e: print(f"{name} FAIL: {type(e).__name__}: {e}", flush=True)
    print("VLM_PATHCAP_RT_DONE", flush=True)
