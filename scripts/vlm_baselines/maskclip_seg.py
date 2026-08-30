"""CLIP segmentation UPPER BOUND via MaskCLIP (canonical, ECCV'22).

Reviewer request: CLIP as an upper bound on the open-vocab segmentation track.
MaskCLIP = the standard training-free CLIP dense-prediction trick: in the LAST
transformer block, drop the q-k attention and use the value projection directly
(no residual, no mlp), then apply ln_post + visual proj per patch token. Raw CLIP
patches are useless for segmentation; MaskCLIP recovers dense open-vocab features.

Protocol is byte-matched to our pipeline's `run_eval` (src/evaluation/zero_shot_segmentation.py):
  * square Resize((224,224), BICUBIC) + CLIP normalize  (same 16x16 grid as DINOv2-L/14@224)
  * text = mean of L2-normed per-template embeds (raw or imagenet-80 ensemble)
  * per-patch L2-norm, cosine with text, bilinear upsample to GT, argmax
  * same confusion-matrix mIoU (fg / all), same DatasetSpec (VOC21 / Context60 / ADE151)
Dataset classes + metric + class names are copied verbatim from the pipeline module so
the datasets/labels/metric are identical; only the encoder is swapped to CLIP+MaskCLIP.

Run in `vlm_eval` (open_clip). NOT `structure`.
    CUDA_VISIBLE_DEVICES=<g> python maskclip_seg.py --dataset voc2012 --arch ViT-L-14
"""
from __future__ import annotations
import argparse, math, os, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from tqdm import tqdm
sys.path.insert(0, "/home/shiwon/PAL_working")
from src.evaluation.zero_shot_metadata import DATASETS_TO_TEMPLATES

DEV = "cuda"
IGNORE_INDEX = 255
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# ══════════════ class names / prompts (verbatim from pipeline) ══════════════
VOC2012_CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
    "chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
VOC2012_CLASS_PROMPTS = {c: c for c in VOC2012_CLASSES}
VOC2012_CLASS_PROMPTS.update({"diningtable":"dining table","pottedplant":"potted plant","tvmonitor":"tv monitor"})

PASCAL_CONTEXT_CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
    "chair","cow","table","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor",
    "bag","bed","bench","book","building","cabinet","ceiling","cloth","computer","cup","door","fence","floor",
    "flower","food","grass","ground","keyboard","light","mountain","mouse","curtain","platform","sign","plate",
    "road","rock","shelves","sidewalk","sky","snow","bedclothes","track","tree","truck","wall","water","window","wood"]
PASCAL_CONTEXT_PROMPTS = {c: c for c in PASCAL_CONTEXT_CLASSES}
PASCAL_CONTEXT_PROMPTS.update({"pottedplant":"potted plant","tvmonitor":"tv monitor","bedclothes":"bedclothes"})

ADE20K_CLASSES = ["background","wall","building","sky","floor","tree","ceiling","road","bed","windowpane","grass",
    "cabinet","sidewalk","person","earth","door","table","mountain","plant","curtain","chair","car","water","painting",
    "sofa","shelf","house","sea","mirror","rug","field","armchair","seat","fence","desk","rock","wardrobe","lamp",
    "bathtub","railing","cushion","base","box","column","signboard","chest of drawers","counter","sand","sink",
    "skyscraper","fireplace","refrigerator","grandstand","path","stairs","runway","case","pool table","pillow",
    "screen door","stairway","river","bridge","bookcase","blind","coffee table","toilet","flower","book","hill",
    "bench","countertop","stove","palm","kitchen island","computer","swivel chair","boat","bar","arcade machine",
    "hovel","bus","towel","light","truck","tower","chandelier","awning","streetlight","booth","television receiver",
    "airplane","dirt track","apparel","pole","land","bannister","escalator","ottoman","bottle","buffet","poster",
    "stage","van","ship","fountain","conveyer belt","canopy","washer","plaything","swimming pool","stool","barrel",
    "basket","waterfall","tent","bag","minibike","cradle","oven","ball","food","step","tank","trade name","microwave",
    "pot","animal","bicycle","lake","dishwasher","screen","blanket","sculpture","hood","sconce","vase","traffic light",
    "tray","ashcan","fan","pier","crt screen","plate","monitor","bulletin board","shower","radiator","glass","clock","flag"]
assert len(ADE20K_CLASSES) == 151
ADE20K_PROMPTS = {c: c for c in ADE20K_CLASSES}

@dataclass
class DatasetSpec:
    name: str; classes: Sequence[str]; prompts: Dict[str, str]
    num_classes: int; ignore_index: int; has_background: bool = True

SPECS = {
    "voc2012": DatasetSpec("voc2012", VOC2012_CLASSES, VOC2012_CLASS_PROMPTS, 21, IGNORE_INDEX, True),
    "pascal_context": DatasetSpec("pascal_context", PASCAL_CONTEXT_CLASSES, PASCAL_CONTEXT_PROMPTS, 60, IGNORE_INDEX, True),
    "ade20k": DatasetSpec("ade20k", ADE20K_CLASSES, ADE20K_PROMPTS, 151, 0, False),
}

def get_text_templates(strategy):
    if strategy == "raw": return ["{}"]
    if strategy == "ensemble": return list(DATASETS_TO_TEMPLATES["imagenet"])
    raise ValueError(strategy)

# ══════════════ datasets (verbatim from pipeline) ══════════════
class PascalContext59Dataset:
    def __init__(self, data_root):
        import scipy.io
        self.root = Path(data_root); self.images_dir = self.root/"images"; self.ann_dir = self.root/"trainval"
        sf = self.root/"val.txt"
        if sf.exists():
            self.ids = [l.strip() for l in open(sf) if l.strip()]
        else:
            self.ids = sorted(p.stem for p in self.ann_dir.glob("*.mat"))
        self._scipy_io = scipy.io
        lf = self.root/"labels.txt"
        raw_label_to_idx = {}
        for line in open(lf):
            line = line.strip()
            if not line: continue
            idx, name = line.split(":", 1); raw_label_to_idx[name.strip()] = int(idx)
        self._context_map = np.zeros(max(raw_label_to_idx.values())+2, dtype=np.int64)
        for new_idx, name in enumerate(PASCAL_CONTEXT_CLASSES):
            if name == "background": continue
            raw = raw_label_to_idx.get(name) or raw_label_to_idx.get(name.lower())
            if raw is not None and raw < self._context_map.shape[0]:
                self._context_map[raw] = new_idx
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        iid = self.ids[i]
        img = Image.open(self.images_dir/f"{iid}.jpg").convert("RGB")
        raw = self._scipy_io.loadmat(str(self.ann_dir/f"{iid}.mat"))["LabelMap"].astype(np.int64)
        rem = self._context_map[np.clip(raw, 0, self._context_map.shape[0]-1)]
        return img, Image.fromarray(rem.astype(np.uint8))

class ADE20KDataset:
    def __init__(self, data_root):
        root = Path(data_root)/"ADEChallengeData2016"
        self.img_dir = root/"images"/"validation"; self.ann_dir = root/"annotations"/"validation"
        self.ids = sorted(p.stem for p in self.img_dir.glob("*.jpg"))
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        iid = self.ids[i]
        return Image.open(self.img_dir/f"{iid}.jpg").convert("RGB"), Image.open(self.ann_dir/f"{iid}.png")

def build_dataset(name, data_root):
    if name == "voc2012":
        return VOCSegmentation(root=data_root, year="2012", image_set="val", download=False), SPECS[name]
    if name == "pascal_context":
        return PascalContext59Dataset(data_root), SPECS[name]
    if name == "ade20k":
        return ADE20KDataset(data_root), SPECS[name]
    raise ValueError(name)

# ══════════════ metric (verbatim from pipeline) ══════════════
def update_confusion_matrix(conf, gt, pred, num_classes, ignore_index):
    mask = gt != ignore_index
    gt_v = np.clip(gt[mask].astype(np.int64), 0, num_classes-1)
    pr_v = np.clip(pred[mask].astype(np.int64), 0, num_classes-1)
    idx = num_classes*gt_v + pr_v
    conf += np.bincount(idx, minlength=num_classes*num_classes).reshape(num_classes, num_classes).astype(np.int64)

def compute_iou(conf, exclude_background):
    cm = conf.astype(np.float64); tp = np.diag(cm)
    denom = tp + (cm.sum(0)-tp) + (cm.sum(1)-tp)
    valid = denom > 0; iou = np.zeros_like(tp); iou[valid] = tp[valid]/denom[valid]
    miou_all = float(iou[valid].mean()) if valid.any() else float("nan")
    if exclude_background and len(iou) > 1:
        fgv = valid.copy(); fgv[0] = False
        miou_fg = float(iou[fgv].mean()) if fgv.any() else float("nan")
    else:
        miou_fg = miou_all
    return miou_fg, miou_all

# ══════════════ MaskCLIP dense features (open_clip ViT) ══════════════
@torch.no_grad()
def maskclip_dense(visual, x):
    """Per-patch features (B, P, D_out) via canonical MaskCLIP.

    Uses the model's own `_embeds` (conv/cls/pos/dropout/ln_pre) so the embedding
    is bit-exact, then runs blocks[:-1] normally. The LAST block drops q-k attention
    and the FFN, and drops the residual: dense feature = out_proj(value) only
    (value-only, no residual, no FFN — the standard MaskCLIP reproduction). Then
    ln_post + visual.proj per token, matching open_clip's final_ln_after_pool=False
    path. `batch_first` is honored so the layout matches the installed open_clip.
    """
    bf = visual.transformer.batch_first
    x = visual._embeds(x)                                 # (N, L, D)
    if not bf:
        x = x.transpose(0, 1)                             # NLD -> LND for the blocks
    blocks = visual.transformer.resblocks
    for blk in blocks[:-1]:
        x = blk(x)
    last = blocks[-1]
    x_ln = last.ln_1(x)
    a = last.attn
    qkv = F.linear(x_ln, a.in_proj_weight, a.in_proj_bias)
    v = qkv.chunk(3, dim=-1)[2]                           # value projection
    x = F.linear(v, a.out_proj.weight, a.out_proj.bias)  # value -> out_proj (no residual, no FFN)
    if not bf:
        x = x.transpose(0, 1)                             # back to NLD
    x = visual.ln_post(x)
    if visual.proj is not None:
        x = x @ visual.proj
    P = x.shape[1] - 1
    gh = int(round(math.sqrt(P)))
    return x[:, 1:, :], gh, gh                            # drop cls token

# ══════════════ eval loop (mirrors run_eval) ══════════════
@torch.no_grad()
def _text_feats(model, tok, spec, strategy):
    templates = get_text_templates(strategy)
    W = []
    for c in [spec.prompts[c] for c in spec.classes]:
        e = model.encode_text(tok([t.format(c) for t in templates]).to(DEV))
        e = F.normalize(e, dim=-1).mean(0)
        W.append(F.normalize(e, dim=-1))
    return torch.stack(W, 0).float().to(DEV)              # (C, D_out), normalized

@torch.no_grad()
def eval_strategies(dataset, spec, model, tok, tfm, strategies, max_images=None):
    """Evaluate all strategies in ONE image pass (image forward is strategy-independent)."""
    text = {s: _text_feats(model, tok, spec, s) for s in strategies}
    conf = {s: np.zeros((spec.num_classes, spec.num_classes), dtype=np.int64) for s in strategies}
    n = len(dataset) if max_images is None else min(max_images, len(dataset))
    for i in tqdm(range(n), desc=spec.name, file=sys.stdout):
        pil_img, pil_mask = dataset[i]
        img_t = tfm(pil_img).unsqueeze(0).float().to(DEV)
        patch, gh, gw = maskclip_dense(model.visual, img_t)
        patch = F.normalize(patch.squeeze(0).float(), dim=-1)   # (P, D)
        gt = np.array(pil_mask); H, W_ = gt.shape
        for s in strategies:
            sim = patch @ text[s].T                             # (P, C)
            sim_map = sim.view(gh, gw, spec.num_classes).permute(2, 0, 1).unsqueeze(0)
            up = F.interpolate(sim_map.float(), size=(H, W_), mode="bilinear", align_corners=False)
            pred = up.squeeze(0).argmax(0).cpu().numpy().astype(np.int64)
            update_confusion_matrix(conf[s], gt, pred, spec.num_classes, spec.ignore_index)
    return {s: compute_iou(conf[s], spec.has_background) for s in strategies}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["voc2012", "pascal_context", "ade20k"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--arch", default="ViT-L-14")
    ap.add_argument("--strategies", default="raw,ensemble")
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(args.arch, pretrained="openai", force_quick_gelu=True)
    model = model.to(DEV).eval(); tok = open_clip.get_tokenizer(args.arch)
    tfm = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])

    ds, spec = build_dataset(args.dataset, args.data_root)
    strategies = args.strategies.split(",")
    print(f"=== MaskCLIP {args.arch} on {args.dataset} (n={len(ds)}, C={spec.num_classes}) ===", flush=True)
    res = eval_strategies(ds, spec, model, tok, tfm, strategies, args.max_images)
    for strat in strategies:
        fg, allm = res[strat]
        print(f"  MaskCLIP {args.arch:9s} {args.dataset:15s} {strat:9s} | mIoU-fg={fg*100:.2f}%  mIoU-all={allm*100:.2f}%", flush=True)
    print("MASKCLIP_SEG_DONE", flush=True)

if __name__ == "__main__":
    main()
