"""Vision encoder loader for feature extraction.

Mirror of ``text_models.load_llm`` for the image side: builds a timm vision
model + its preprocessing transform, set up to return per-block token features.
Moved out of ``FeatureStore.get_lvm`` so both encoder loaders live under
``encoders/`` (FeatureStore keeps a thin ``get_lvm`` wrapper). The only state it
needed from the store was ``img_size`` and ``device``, now explicit args.
"""

import timm
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from torchvision.models.feature_extraction import create_feature_extractor

from src.datasets.data_utils import _ensure_rgb_image

# Per-model extra kwargs for timm.create_model. Some DINOv2-family hf-hub
# checkpoints (e.g. pathology UNI) use LayerScale (blocks.*.ls{1,2}.gamma); timm
# only builds those params when ``init_values`` is passed, so without it the
# pretrained state_dict fails to load. NOTE: do NOT set ``dynamic_img_size`` here
# — it inserts control flow into pos-embed resampling that torch.fx (used by
# create_feature_extractor below) cannot trace.
_LVM_EXTRA_KWARGS = {
    "hf-hub:MahmoodLab/UNI": {"init_values": 1e-5},
    # UNI2-h: ViT-H/14, 1536-d, SwiGLU MLP + SiLU, 8 register tokens (token layout =
    # 1 CLS + 8 reg + 256 patch = 265). dynamic_img_size MUST stay False so torch.fx can
    # trace the fixed 224 pos-embed (verified: create_feature_extractor yields
    # blocks.{i}.add_1 with 265 tokens). num_prefix_tokens=9; per-patch consumers
    # (seg/localization) must slice [9:] not [1:] — token-level CAP pools all tokens, so
    # training/retrieval need no change.
    "hf-hub:MahmoodLab/UNI2-h": {
        "patch_size": 14, "depth": 24, "num_heads": 24, "init_values": 1e-5,
        "embed_dim": 1536, "mlp_ratio": 2.66667 * 2, "num_classes": 0,
        "no_embed_class": True, "mlp_layer": SwiGLUPacked, "act_layer": nn.SiLU,
        "reg_tokens": 8, "dynamic_img_size": False,
    },
}


def load_lvm(lvm_model_name, img_size=None, device="cpu"):
    """Build a vision encoder + transform that yields per-layer token features.

    Returns ``(vision_model, transform)``. ``vision_model`` is wrapped in a
    feature extractor returning every transformer block's output
    (``blocks.{i}.add_1``); only ViT-family models are supported.
    """
    model_kwargs = dict(_LVM_EXTRA_KWARGS.get(lvm_model_name, {}))
    if img_size is not None:
        model_kwargs["img_size"] = int(img_size)
    vision_model = timm.create_model(
        lvm_model_name, pretrained=True, **model_kwargs
    )
    data_config = resolve_data_config(
        vision_model.pretrained_cfg, model=vision_model
    )
    if img_size is not None:
        data_config["input_size"] = (3, int(img_size), int(img_size))
        data_config["crop_pct"] = 1.0
    transform = create_transform(**data_config)
    transform.transforms = [_ensure_rgb_image] + transform.transforms

    # ViT-family (timm DINOv2, hf-hub UNI, …): every transformer block exposes a
    # residual-add node ``blocks.{i}.add_1``. Broadened from a name substring
    # check to ``hasattr(blocks)`` so hf-hub ViTs whose name lacks "vit"
    # (e.g. "hf-hub:MahmoodLab/UNI") are covered too.
    if hasattr(vision_model, "blocks"):
        return_nodes = [
            f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))
        ]
    else:
        raise NotImplementedError(f"unknown model {lvm_model_name}")
    vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)
    vision_model = vision_model.to(device)
    vision_model = vision_model.eval()
    return vision_model, transform
