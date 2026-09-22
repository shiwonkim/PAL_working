"""Phrase-grounding pointing-game evaluation on Flickr30k Entities.

Reuses the zero-shot-segmentation CLI's method / encoder infrastructure so the
per-patch descriptors (``get_patch_features``) and text descriptors
(``get_text_features``) are produced exactly as in segmentation, with the same
``direct`` / ``factorized`` decoding. The only differences are the dataset
(Flickr30k Entities phrases + boxes) and the metric.

Metric (pointing game, Akbari et al., CVPR 2019): for each annotated phrase we
build a patch-level phrase-image similarity map, upsample it to the original
image resolution (bilinear, matching the segmentation CLI), take the argmax
pixel, and count a hit if that point falls inside any ground-truth box for the
phrase. ``pointing_acc = hits / total_phrases``.

The image transform is a plain square resize with no crop (from
``build_vision_encoder``), so upsampling the h x w response back to the original
(H, W) aligns the argmax point with the original-coordinate boxes.

Usage:
    python -m src.evaluation.grounding \
        --config_path configs/pal/vitl_roberta/token_k512.yaml \
        --ckpt <checkpoint.pth> --method anchor_codebook --label pal \
        --layer-img 23 --layer-txt 24 \
        [--decoding direct] [--text-strategy raw] [--out_csv <path>]
"""
import argparse
import csv
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from tqdm import tqdm

from src.evaluation.zero_shot_segmentation import (
    build_language_encoder,
    build_method,
    build_vision_encoder,
    get_text_templates,
    load_config,
)
from src.utils.checkpoint import load_alignment_layer

# [/EN#123/people A young boy]  ->  (entity_id="123", phrase="A young boy")
_PHRASE_RE = re.compile(r"\[/EN#(\d+)/[^ \]]+((?:/[^ \]]+)*) ([^\]]+)\]")


def parse_phrases(sentences_path: Path):
    """Return a list of (entity_id, phrase_text) over all 5 sentences."""
    phrases = []
    with open(sentences_path, encoding="utf-8") as f:
        for line in f:
            for m in _PHRASE_RE.finditer(line):
                phrases.append((m.group(1), m.group(3).strip()))
    return phrases


def parse_boxes(annotation_path: Path):
    """Return (boxes: {entity_id -> [(x1,y1,x2,y2), ...]}, width, height).

    A single <object> may carry several <name> tags (a box shared by multiple
    entities) and an entity may span several <object> tags (multiple boxes).
    Objects without a <bndbox> (scene / nobndbox entities) contribute no box.
    """
    root = ET.parse(annotation_path).getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    boxes: dict[str, list] = {}
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        box = (
            int(bnd.find("xmin").text), int(bnd.find("ymin").text),
            int(bnd.find("xmax").text), int(bnd.find("ymax").text),
        )
        for name in obj.findall("name"):
            boxes.setdefault(name.text, []).append(box)
    return boxes, width, height


def _find_image(image_root: Path, image_id: str) -> Path:
    for cand in (
        image_root / f"{image_id}.jpg",
        image_root / "flickr30k-images" / f"{image_id}.jpg",
        image_root / "flickr30k_images" / f"{image_id}.jpg",
        image_root / "images" / f"{image_id}.jpg",
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"image {image_id}.jpg not found under {image_root}")


def _decode_sim(method, patch_feats, text_feats):
    """(P, D_m), (N, D_m) -> (P, N) similarity, matching the seg decode branch."""
    decoding = getattr(method, "decoding", "direct")
    if decoding == "factorized":
        return patch_feats @ text_feats.T
    return F.normalize(patch_feats, dim=-1) @ text_feats.T


def evaluate_pointing(
    method, vision_model, image_transform, tokenizer, language_model,
    layer_img, layer_txt, image_root, entities_root, image_ids, device,
    text_strategy="raw", max_images=None,
):
    templates = get_text_templates(text_strategy)
    n_prefix = int(getattr(vision_model, "num_prefix_tokens", 1))
    ids = image_ids if max_images is None else image_ids[:max_images]

    hits = 0
    total = 0
    n_images = 0
    for image_id in tqdm(ids, desc=f"ground/{method.name}", file=os.sys.stdout):
        ann_path = Path(entities_root) / "Annotations" / f"{image_id}.xml"
        sent_path = Path(entities_root) / "Sentences" / f"{image_id}.txt"
        if not ann_path.exists() or not sent_path.exists():
            continue
        boxes, _, _ = parse_boxes(ann_path)
        phrases = [(eid, txt) for eid, txt in parse_phrases(sent_path) if eid in boxes]
        if not phrases:
            continue

        pil = Image.open(_find_image(Path(image_root), image_id)).convert("RGB")
        W, H = pil.size
        img_t = image_transform(pil).unsqueeze(0).float().to(device)
        with torch.no_grad():
            lvm_out = vision_model(img_t)
            layer_key = list(lvm_out.keys())[layer_img]
            feats = lvm_out[layer_key].squeeze(0)  # (T, D)
            patch_feats = method.get_patch_features(feats, device, n_prefix).float()

            text_feats = method.get_text_features(
                classnames=[txt for _, txt in phrases],
                templates=templates, tokenizer=tokenizer,
                language_model=language_model, layer_txt=layer_txt, device=device,
            )
            text_feats = F.normalize(text_feats.float().to(device), dim=-1)

            sim = _decode_sim(method, patch_feats, text_feats)  # (P, N)
        P, N = sim.shape
        h = int(round(math.sqrt(P)))
        if h * h != P:
            raise RuntimeError(f"non-square patch grid: P={P}")
        # (N, 1, h, h) -> upsample to original (H, W) -> argmax pixel per phrase
        maps = sim.T.reshape(N, 1, h, h).float()
        maps = F.interpolate(maps, size=(H, W), mode="bilinear", align_corners=False)
        flat = maps.reshape(N, H * W)
        arg = flat.argmax(dim=1).cpu().numpy()
        for j, (eid, _) in enumerate(phrases):
            y, x = divmod(int(arg[j]), W)
            hit = any(x1 <= x <= x2 and y1 <= y <= y2 for (x1, y1, x2, y2) in boxes[eid])
            hits += int(hit)
            total += 1
        n_images += 1

    acc = hits / total if total else 0.0
    return {"pointing_acc": acc, "hits": hits, "total": total, "images": n_images}


def load_test_ids(split_json: Path, split: str = "test"):
    data = json.load(open(split_json))
    ids = []
    for im in data["images"]:
        if im["split"] == split:
            ids.append(Path(im["filename"]).stem)
    return ids


def load_ids_from_txt(split_file: Path):
    """One image id (or filename) per line — the Flickr30k Entities test split."""
    return [Path(l.strip()).stem for l in open(split_file) if l.strip()]


def main():
    p = argparse.ArgumentParser(description="Flickr30k Entities pointing-game eval")
    p.add_argument("--config_path", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--method", required=True,
                   help="anchor_codebook | fa | linear_perpatch")
    p.add_argument("--label", default="")
    p.add_argument("--layer-img", type=int, default=23)
    p.add_argument("--layer-txt", type=int, default=24)
    p.add_argument("--decoding", default=None, choices=["direct", "factorized"])
    p.add_argument("--text-strategy", default="raw", choices=["raw", "ensemble"])
    p.add_argument("--image-root", default="data/flickr30k")
    p.add_argument("--entities-root", default="data/flickr30k_entities")
    p.add_argument("--split-json",
                   default="data/COCO/karpathy_splits/dataset_flickr30k.json")
    p.add_argument("--split-file", default="data/flickr30k/test.txt",
                   help="Flickr30k Entities test split (one image id per line); "
                        "takes precedence over --split-json when it exists")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--out_csv", default=None)
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config_path)

    vision_model, image_transform, _ = build_vision_encoder(cfg, device)
    language_model, tokenizer = build_language_encoder(cfg, device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    alignment_image = load_alignment_layer(ckpt["alignment_image"], "image", device)
    alignment_text = load_alignment_layer(ckpt["alignment_text"], "text", device)
    method = build_method(
        args.method, alignment_image, alignment_text, cfg,
        decoding_override=args.decoding,
    )

    if args.split_file and Path(args.split_file).exists():
        image_ids = load_ids_from_txt(Path(args.split_file))
    else:
        image_ids = load_test_ids(Path(args.split_json))
    logger.info(
        f"[{args.label or args.method}] method={args.method} "
        f"decoding={getattr(method, 'decoding', 'direct')} "
        f"text={args.text_strategy} test_images={len(image_ids)}"
    )

    res = evaluate_pointing(
        method, vision_model, image_transform, tokenizer, language_model,
        args.layer_img, args.layer_txt, args.image_root, args.entities_root,
        image_ids, device, text_strategy=args.text_strategy,
        max_images=args.max_images,
    )
    logger.info(
        f"[{args.label or args.method}] pointing acc = {res['pointing_acc']:.4f} "
        f"({res['hits']}/{res['total']} phrases, {res['images']} images)"
    )

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out.exists()
        with open(out, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["label", "method", "decoding", "text_strategy",
                            "pointing_acc", "hits", "total", "images"])
            w.writerow([args.label, args.method,
                        getattr(method, "decoding", "direct"), args.text_strategy,
                        f"{res['pointing_acc']:.4f}", res["hits"], res["total"],
                        res["images"]])
        logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
