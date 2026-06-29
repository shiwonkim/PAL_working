"""Vision encoder loader for feature extraction.

Mirror of ``text_models.load_llm`` for the image side: builds a timm vision
model + its preprocessing transform, set up to return per-block token features.
Moved out of ``FeatureStore.get_lvm`` so both encoder loaders live under
``encoders/`` (FeatureStore keeps a thin ``get_lvm`` wrapper). The only state it
needed from the store was ``img_size`` and ``device``, now explicit args.
"""

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor

from src.data.data_utils import _ensure_rgb_image


def load_lvm(lvm_model_name, img_size=None, device="cpu"):
    """Build a vision encoder + transform that yields per-layer token features.

    Returns ``(vision_model, transform)``. ``vision_model`` is wrapped in a
    feature extractor returning every transformer block's output
    (``blocks.{i}.add_1``); only ViT-family models are supported.
    """
    model_kwargs = {}
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

    if "vit" in lvm_model_name:
        return_nodes = [
            f"blocks.{i}.add_1" for i in range(len(vision_model.blocks))
        ]
    else:
        raise NotImplementedError(f"unknown model {lvm_model_name}")
    vision_model = create_feature_extractor(vision_model, return_nodes=return_nodes)
    vision_model = vision_model.to(device)
    vision_model = vision_model.eval()
    return vision_model, transform
