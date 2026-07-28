"""Vision encoder loader for feature extraction.

Mirror of ``text_models.load_llm`` for the image side: builds a timm vision
model + its preprocessing transform, set up to return per-block token features.
Moved out of ``FeatureStore.get_lvm`` so both encoder loaders live under
``encoders/`` (FeatureStore keeps a thin ``get_lvm`` wrapper). The only state it
needed from the store was ``img_size`` and ``device``, now explicit args.
"""

import timm
import torch.nn as nn
import torchvision.transforms as T
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor

from src.datasets.data_utils import _ensure_rgb_image

# HuggingFace-transformers vision encoders (not timm). These load via AutoModel
# and expose per-layer tokens through ``output_hidden_states`` rather than
# torch.fx nodes, so they take the ``_load_hf_lvm`` path below. Example: RAD-DINO
# (chest-X-ray DINOv2 ViT-B, native 518, its own CXR mean/std).
_HF_LVM_MODELS = {"microsoft/rad-dino"}


# Per-model extra kwargs for timm.create_model. Some DINOv2-family hf-hub
# checkpoints (e.g. pathology UNI) use LayerScale (blocks.*.ls{1,2}.gamma); timm
# only builds those params when ``init_values`` is passed, so without it the
# pretrained state_dict fails to load. NOTE: do NOT set ``dynamic_img_size`` here
# — it inserts control flow into pos-embed resampling that torch.fx (used by
# create_feature_extractor below) cannot trace.
_LVM_EXTRA_KWARGS = {
    "hf-hub:MahmoodLab/UNI": {"init_values": 1e-5},
}


class _HFViTFeatureExtractor(nn.Module):
    """Adapt a HuggingFace ViT (e.g. Dinov2Model / RAD-DINO) to the timm
    feature-extractor interface used by ``FeatureStore``: ``forward(x)`` returns
    an ordered dict ``{"blocks.{i}.add_1": per_block_tokens}``.

    Uses ``output_hidden_states`` and drops the first entry (the embedding
    output) so ``.values()`` are the per-block outputs in order — matching
    timm's ``blocks.{i}.add_1`` nodes, with token index 0 = CLS.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values, output_hidden_states=True)
        hidden = out.hidden_states[1:]  # skip embeddings -> one per block
        return {f"blocks.{i}.add_1": h for i, h in enumerate(hidden)}


def _load_hf_lvm(lvm_model_name, device):
    """Load a HuggingFace-transformers ViT + its preprocessing as a torchvision
    transform (so it drops into the same dataset pipeline as the timm path)."""
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(lvm_model_name).eval()
    proc = AutoImageProcessor.from_pretrained(lvm_model_name)
    # Rebuild the processor's resize/crop/normalize as a torchvision Compose;
    # RAD-DINO uses its own CXR mean/std (not ImageNet), read from the processor.
    crop = proc.crop_size["height"]
    resize = proc.size.get("shortest_edge", crop)
    transform = T.Compose([
        T.Resize(resize, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(crop),
        T.ToTensor(),
        T.Normalize(mean=proc.image_mean, std=proc.image_std),
    ])
    transform.transforms = [_ensure_rgb_image] + transform.transforms
    model = _HFViTFeatureExtractor(model).to(device).eval()
    return model, transform


def load_lvm(lvm_model_name, img_size=None, device="cpu"):
    """Build a vision encoder + transform that yields per-layer token features.

    Returns ``(vision_model, transform)``. ``vision_model`` is wrapped in a
    feature extractor returning every transformer block's output
    (``blocks.{i}.add_1``); only ViT-family models are supported. HuggingFace
    ViTs in ``_HF_LVM_MODELS`` (e.g. RAD-DINO) take the ``_load_hf_lvm`` path.
    """
    if lvm_model_name in _HF_LVM_MODELS:
        return _load_hf_lvm(lvm_model_name, device)

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
