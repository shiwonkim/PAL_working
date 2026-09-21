"""Non-standard MAP (attention-pooling) segmentation eval — SEPARATE from the
main zero_shot_segmentation.py.

MAP (AttnPoolAlignmentLayer) has NO native per-patch representation: a single
learnable query attention-pools all tokens into one global vector, and the
contrastive alignment happens only at that pooled vector. To get a per-patch
descriptor for dense prediction we EXTEND it MaskCLIP-style — apply MAP's
per-token sub-transformations to each patch, SKIPPING the attention pooling:

    desc_p = out_proj( v_proj( in_proj(patch_p) ) ) + MLP-head(...)

i.e. treat each patch's value as if it were the pooled context. This DISCARDS
MAP's defining mechanism (the learned-query attention), so it is a degenerate
extension — it evaluates "MAP's value projection per-patch", not "MAP doing
dense prediction". Reported only to contrast against CAP's NATIVE per-patch
relative reps.

Reuses the machinery (encoders / dataset / run_eval / decoding) from
zero_shot_segmentation.py; only adds MAPPerPatchMethod + a thin CLI.

  cd /workspace/PAL && python -m src.evaluation.zero_shot_segmentation_map \
      --config configs/attn_pool/vitl_roberta/attn_pool_d512_token.yaml \
      --checkpoint <map_ckpt.pth> --layer-img 23 --layer-txt 24 \
      --dataset voc2012 --data-root data/pascal_voc --text-strategies ensemble
"""
import argparse
import os

import torch
from loguru import logger

from src.evaluation.zero_shot_segmentation import (
    SegmentationMethod,
    build_vision_encoder,
    build_language_encoder,
    build_dataset,
    run_eval,
    get_text_templates,
    load_config,
    print_results_table,
)
from src.evaluation.zero_shot_classifier import build_zero_shot_classifier
from src.utils.checkpoint import load_alignment_layer


class MAPPerPatchMethod(SegmentationMethod):
    """MAP attention-pooling extended to per-patch (attention skipped).

    Image side: apply in_proj -> v_proj -> out_proj -> residual MLP head to each
    patch (no attention). Text side: normal MAP text forward (pooled per class).
    Decoding is ``direct`` (per-patch L2-norm + cosine), matching the standard
    open-vocab seg protocol used everywhere else.
    """

    name = "map_perpatch"
    pool_txt = "none"

    def __init__(self, alignment_image, alignment_text, token_level: bool = True,
                 pool_txt: str = "avg"):
        self.alignment_image = alignment_image
        self.alignment_text = alignment_text
        self.decoding = "direct"
        self.token_level = token_level
        self.pool_txt = pool_txt

    def get_patch_features(self, layer_feats, device, n_prefix: int = 1):
        with torch.no_grad():
            ai = self.alignment_image
            patches = layer_feats[n_prefix:, :].to(device)  # (P, D) strip CLS + registers
            kv = ai.in_proj(patches)                     # (P, d)
            v = ai.v_proj(kv)                            # (P, d) per-patch value
            pooled = ai.out_proj(v)                      # (P, d) (attention skipped)
            desc = pooled + ai.mlp(ai.ln(pooled))        # (P, d) residual MLP head
            return desc

    def get_text_features(self, classnames, templates, tokenizer, language_model,
                          layer_txt, device):
        return build_zero_shot_classifier(
            language_model=language_model, tokenizer=tokenizer,
            classnames=classnames, templates=templates, dataset=None,
            layer_index=layer_txt, alignment_layer=self.alignment_text,
            num_classes_per_batch=8, device=device,
            pool_txt="none" if self.token_level else self.pool_txt,
            save_path=None, token_level=self.token_level,
        ).to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--layer-img", type=int, required=True)
    p.add_argument("--layer-txt", type=int, required=True)
    p.add_argument("--dataset", default="voc2012",
                   choices=["voc2012", "pascal_context", "ade20k"])
    p.add_argument("--data-root", default="data/pascal_voc")
    p.add_argument("--text-strategies", default="ensemble")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    vision_model, image_transform, img_size = build_vision_encoder(cfg, device)
    language_model, tokenizer = build_language_encoder(cfg, device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    alignment_image = load_alignment_layer(ckpt["alignment_image"], "image", device)
    alignment_text = load_alignment_layer(ckpt["alignment_text"], "text", device)
    logger.info(f"MAP alignment class: {type(alignment_image).__name__}")

    method = MAPPerPatchMethod(
        alignment_image=alignment_image, alignment_text=alignment_text,
        token_level=bool(cfg["training"].get("token_level", False)),
        pool_txt=cfg["features"].get("pool_txt", "avg"),
    )
    dataset, dataset_spec = build_dataset(args.dataset, args.data_root, download=False)

    results = []
    for strategy in [s.strip() for s in args.text_strategies.split(",") if s.strip()]:
        res = run_eval(
            method=method, strategy=strategy, dataset=dataset,
            dataset_spec=dataset_spec, vision_model=vision_model,
            image_transform=image_transform, img_size=img_size,
            tokenizer=tokenizer, language_model=language_model,
            layer_img=args.layer_img, layer_txt=args.layer_txt, device=device,
            max_images=args.max_images,
        )
        results.append({"method": method.name, "strategy": strategy,
                        "miou_fg": res["miou_fg"], "miou_all": res["miou_all"],
                        "per_class": res.get("per_class", [])})
    print_results_table(results, dataset_spec)


if __name__ == "__main__":
    main()
