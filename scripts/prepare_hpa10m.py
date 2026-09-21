"""Prepare downloaded HPA10M WebDataset shards for the PAL pipeline.

Builds a flat images/ dir + train/val selection CSVs in the QuiltCaptionDataset schema
(image_path = bare filename, caption = chosen field). ALL THREE caption variants + the
structured metadata are kept as extra columns so the `caption` column can be re-pointed
later (pathologist's choice) with NO image re-extraction.

--mode controls how each 3000px TMA core becomes a stored image (encoder input is 224 either
way; the mode decides WHAT the 224 shows):
  whole : resize shortest side -> --size (default 256), whole core (option C, thumbnail).
  crop  : native-resolution square crop of --size px centered on the tissue bbox (option A,
          cell resolution). No downsampling of the crop.
Switching options later = re-run with a different --mode; the tars stay on disk, so it just
re-reads them (no re-download, no code change). Multi-tile (option B) is a future --mode.

Train pool = hpa10m_train/ + hpa10m_train_part2/ (200 downloaded shards). Val = validation tar.

Usage (tmux; multiprocessed):
  python scripts/prepare_hpa10m.py --mode whole --size 256 --workers 16
"""
import argparse, csv, io, json, glob, os, tarfile
from multiprocessing import Pool
from PIL import Image

SRC = "/home/data/2026_hpa10m"
OUT = "/home/data/2026_hpa10m/dataset"
CAP_FIELDS = ["generic_caption", "caption_1", "caption_2"]
META_FIELDS = ["gene", "ensembl_id", "uniprot_id", "tissue", "cell_type",
               "staining_intensity", "staining_location", "staining_quantity"]

def shard_paths(split):
    if split == "val":
        return sorted(glob.glob(f"{SRC}/hpa10m_validation/*.tar"))
    return sorted(glob.glob(f"{SRC}/hpa10m_train/*.tar")) + \
           sorted(glob.glob(f"{SRC}/hpa10m_train_part2/*.tar"))

_CFG = {}
def _init(mode, size, caption_field, img_dir):
    _CFG.update(mode=mode, size=size, caption_field=caption_field, img_dir=img_dir)

def _to_stored(img, cm):
    """Return the PIL image to store, per _CFG['mode']."""
    mode, s = _CFG["mode"], _CFG["size"]
    W, H = img.size
    if mode == "whole":
        if min(W, H) <= s: return img
        sc = s / min(W, H)
        return img.resize((round(W * sc), round(H * sc)), Image.BICUBIC)
    # crop: native square of size s centered on tissue bbox (fallback: image center)
    cx, cy = W // 2, H // 2
    bb = cm.get("bboxes")
    try:
        if isinstance(bb, str): bb = json.loads(bb)
        x, y, bw, bh = bb[0]; cx, cy = int(x + bw / 2), int(y + bh / 2)
    except Exception:
        pass
    half = s // 2
    left = max(0, min(cx - half, W - s)); top = max(0, min(cy - half, H - s))
    return img.crop((left, top, left + min(s, W), top + min(s, H)))

def process_shard(tar):
    mode, cf, img_dir = _CFG["mode"], _CFG["caption_field"], _CFG["img_dir"]
    rows = []
    tf = tarfile.open(tar)
    members = {}
    for m in tf.getmembers():
        if "." not in m.name: continue
        stem, ext = m.name.rsplit(".", 1); members.setdefault(stem, {})[ext] = m
    for stem, d in members.items():
        if "jpg" not in d or "json" not in d: continue
        cm = json.load(tf.extractfile(d["json"])).get("custom_metadata", {})
        cap = cm.get(cf)
        if not cap or str(cap).strip() in ("", "nan"): continue
        fname = os.path.basename(d["jpg"].name)
        out_path = os.path.join(img_dir, fname)
        if not os.path.exists(out_path):
            try:
                img = Image.open(io.BytesIO(tf.extractfile(d["jpg"]).read())).convert("RGB")
                _to_stored(img, cm).save(out_path, "JPEG", quality=90)
            except Exception as e:
                print(f"  skip {fname}: {e}", flush=True); continue
        row = {"image_path": fname, "caption": cap, "source": os.path.basename(tar)}
        for k in CAP_FIELDS: row[k] = cm.get(k)
        for k in META_FIELDS: row[k] = cm.get(k)
        rows.append(row)
    tf.close()
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="whole", choices=["whole", "crop"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--caption-field", default="generic_caption", choices=CAP_FIELDS)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    # separate images dir per mode so switching doesn't mix whole/crop files
    img_dir = os.path.join(OUT, f"images_{args.mode}{args.size}")
    sel_dir = os.path.join(OUT, "selections")
    os.makedirs(img_dir, exist_ok=True); os.makedirs(sel_dir, exist_ok=True)
    cols = ["image_path", "caption", "source"] + CAP_FIELDS + META_FIELDS

    for split in ["val", "train"]:            # val first (small) as a smoke
        shards = shard_paths(split)
        out_csv = os.path.join(sel_dir, f"hpa10m_{split}_{args.mode}{args.size}_seed42.csv")
        total = 0
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            with Pool(args.workers, initializer=_init,
                      initargs=(args.mode, args.size, args.caption_field, img_dir)) as pool:
                for i, rows in enumerate(pool.imap_unordered(process_shard, shards), 1):
                    for r in rows: w.writerow(r)
                    total += len(rows)
                    print(f"  [{split}] shard {i}/{len(shards)}  total={total}", flush=True)
        print(f"[{split}] wrote {total} rows -> {out_csv}  (images: {img_dir})", flush=True)
    print("HPA_PREPARE_DONE", flush=True)

if __name__ == "__main__":
    main()
