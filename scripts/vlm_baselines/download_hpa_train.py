"""Download the first N existing HPA10M train shards to /home/data/2026_hpa10m.

WebDataset shards are ~0.9GB / ~1000 imgs each. Shards are numbered non-contiguously
(hpa10m_train starts at 0002 with gaps; rest in hpa10m_train_part2). We enumerate the
real shard list via the HF API and take the first N, so N shards actually land.
Xet is disabled (HF_HUB_DISABLE_XET=1 set by the caller) — the Xet CDN was timing out.
Resumable: re-running skips shards already fully downloaded.
"""
import sys, re, json, urllib.request
from huggingface_hub import snapshot_download

REPO = "nirschl-lab/hpa10m"
LOCAL = "/home/data/2026_hpa10m"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200

def list_train_shards():
    shards = []
    for folder in ("hpa10m_train", "hpa10m_train_part2"):
        url = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{folder}?recursive=false"
        try:
            data = json.load(urllib.request.urlopen(url, timeout=30))
        except Exception as e:
            print(f"  (skip {folder}: {e})", flush=True); continue
        for x in data:
            p = x.get("path", "")
            if p.endswith(".tar"):
                m = re.search(r"_(\d+)\.tar$", p)
                shards.append((int(m.group(1)) if m else 0, p))
    shards.sort()
    return [p for _, p in shards]

all_shards = list_train_shards()
pick = all_shards[:N]
print(f"total train shards visible: {len(all_shards)}; downloading first {len(pick)}", flush=True)
print(f"  range: {pick[0]} .. {pick[-1]}", flush=True)

snapshot_download(
    repo_id=REPO, repo_type="dataset", local_dir=LOCAL,
    allow_patterns=pick, max_workers=8,
)
print(f"HPA_TRAIN_DONE ({len(pick)} shards)", flush=True)
