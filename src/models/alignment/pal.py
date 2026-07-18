"""PAL (Projection-free Anchor Learning) alignment layer.

A single layer that handles both modes, chosen by the rank of the input:
  - token ``(B, T, D)``: Cross-Attention Pooling (CAP) — pools token
    contributions per anchor with a softmax over tokens (the headline PAL).
  - CLS ``(B, D)``: cosine similarity to the K anchors (the 2D path; used when
    only pooled embeddings are available, or for CLS-trained configs).

Which mode runs is decided by the features fed in (config ``token_level``), not
by the class — token-PAL and CLS-PAL are the same ``PALAlignmentLayer``.

Forward contract:
    input:  z (B, T, D) token features or (B, D) CLS embedding
            mask (B, T) optional — 1 = valid token, 0 = padding
    output: (B, K) L2-normalized profile (same shape in both modes, so
            CLIPLoss / STRUCTURE reg are unchanged)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.alignment.alignment_factory import AlignmentFactory
from src.models.alignment.base_alignment_layer import BaseAlignmentLayer


class BottleneckProjector(nn.Module):
    """Lightweight residual bottleneck projector.

    Zero-initialised on the up projection so the layer starts as identity.
    """

    def __init__(self, d_in: int, d_mid: int):
        super().__init__()
        self.down = nn.Linear(d_in, d_mid)
        self.act = nn.GELU()
        self.up = nn.Linear(d_mid, d_in)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


@AlignmentFactory.register()
class PALAlignmentLayer(BaseAlignmentLayer):
    """PAL: K learnable anchors; token input → CAP, CLS input → cosine profile."""

    def __init__(
        self,
        input_dim: int,
        num_anchors: int | None = None,
        pool_temperature: float = 0.05,
        projector_dim: int = 0,
        init_method: str = "random",
        dim_alignment: int | None = None,
        pool_method: str = "cap",
        fix_anchors: bool = False,
        topk: int | None = None,
        sim_exponent: float = 1.0,
    ):
        super().__init__(input_dim=input_dim)

        # dim_alignment alias survives STRUCTURE's YAML deep-merge (default.yaml
        # injects dim_alignment=256 into every alignment_layer_kwargs).
        if num_anchors is None:
            num_anchors = dim_alignment if dim_alignment is not None else 128
        self.num_anchors = num_anchors
        self.pool_temperature = pool_temperature
        self.projector_dim = projector_dim
        self.init_method = init_method
        self.pool_method = pool_method
        # ASIF-style fixed-anchor baseline knobs (all no-op by default so
        # learnable PAL is unchanged): fix_anchors stops anchor gradients,
        # topk sparsifies the profile, sim_exponent re-weights kept similarities.
        self.fix_anchors = fix_anchors
        self.topk = topk
        self.sim_exponent = sim_exponent

        self.anchors = nn.Parameter(torch.empty(num_anchors, input_dim))
        if init_method in ("random", "normal"):
            nn.init.normal_(self.anchors)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
        with torch.no_grad():
            self.anchors.data = F.normalize(self.anchors.data, dim=-1)
        if fix_anchors:
            self.anchors.requires_grad_(False)

        self.projector = (
            BottleneckProjector(input_dim, projector_dim) if projector_dim > 0 else None
        )

    def set_anchors_from_data(self, vecs: torch.Tensor) -> None:
        """Install fixed, data-derived anchors (ASIF-style) and freeze them.

        ``vecs`` are ``(num_anchors, input_dim)`` pooled embeddings sampled from
        ground-truth pairs; they are L2-normalized and copied into ``anchors``,
        which is then frozen. Used by the fixed-anchor (Flavor A) baseline so
        that image-anchor-k and text-anchor-k are a real (image, caption) pair,
        exactly as in ASIF — instead of the correspondence being learned.
        """
        with torch.no_grad():
            v = F.normalize(
                vecs.to(dtype=self.anchors.dtype, device=self.anchors.device), dim=-1
            )
            if v.shape != self.anchors.shape:
                raise ValueError(
                    f"anchor shape {tuple(v.shape)} != {tuple(self.anchors.shape)}"
                )
            self.anchors.data.copy_(v)
        self.anchors.requires_grad_(False)

    def _postprocess(self, profile: torch.Tensor) -> torch.Tensor:
        """ASIF-style sparsify + exponentiate the profile, then L2-normalize.

        No-op (just normalize, matching the original PAL) unless ``topk`` or
        ``sim_exponent`` are configured.
        """
        topk = getattr(self, "topk", None)
        if topk is not None and topk < profile.shape[-1]:
            vals, idx = profile.topk(topk, dim=-1)  # ASIF (i): keep top-k, zero rest
            profile = torch.zeros_like(profile).scatter_(-1, idx, vals)
        p = getattr(self, "sim_exponent", 1.0)
        if p != 1.0:
            profile = profile.clamp(min=0).pow(p)  # ASIF (ii): exponent on kept sims
        return F.normalize(profile, dim=-1)

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # CLS-fallback path for 2D input (used at eval time when only CLS is
        # available, or for backwards compat with CLS-only configs).
        if z.dim() == 2:
            z_norm = F.normalize(z, dim=-1)
            a_norm = F.normalize(self.anchors, dim=-1)
            profile = z_norm @ a_norm.T  # (B, K)
            return self._postprocess(profile)

        # Token-level path: z is (B, T, D)
        if self.projector is not None:
            z = self.projector(z)

        z_norm = F.normalize(z, dim=-1)              # (B, T, D)
        a_norm = F.normalize(self.anchors, dim=-1)   # (K, D)
        sim = z_norm @ a_norm.T                       # (B, T, K)

        if getattr(self, "pool_method", "cap") == "mean":
            if mask is not None:
                mask_f = mask.unsqueeze(-1).float()   # (B, T, 1)
                profile = (sim * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            else:
                profile = sim.mean(dim=1)             # (B, K)
            return self._postprocess(profile)

        # CAP path (default)
        logits = sim / self.pool_temperature         # (B, T, K)

        if mask is not None:
            logits = logits.masked_fill(
                ~mask.bool().unsqueeze(-1), float("-inf")
            )

        attn = F.softmax(logits, dim=1)              # softmax over tokens
        attn = attn.nan_to_num(0.0)                   # all-masked safety

        profile = (attn * sim).sum(dim=1)            # (B, K)
        return self._postprocess(profile)
