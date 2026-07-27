"""Single-query attention-pooling ("MAP") alignment baseline.

A standard learned-query cross-attention pooling head — Multihead Attention
Pooling (MAP), as used in ViT / SigLIP / CoCa: ONE learnable query cross-attends
over the token sequence with full Q/K/V projections and pools it to a single
``dim_alignment``-d embedding, followed by a residual MLP head. The attention is
implemented manually (explicit Q·Kᵀ → mask → softmax → ·V) rather than via
``nn.MultiheadAttention``, because torch 2.1's memory-efficient SDPA backend
returns NaN gradients with a key-padding mask (broke the LR finder); the manual
math path is numerically stable and identical in definition.

Purpose (rebuttal control for CAP; see ``docs/cap_pooling_ablation.ko.md``):
    Reviewer VQng (Q1/Q2) asks whether CAP's gains come from its *specific*
    similarity-weighted (anchor-cosine-softmax) design or from *any* learned
    weighted token aggregation, and requests a comparison against "standard
    cross-attention with learned queries" (Q2: CAP itself is similarity-weighted
    pooling, NOT cross-attention). This layer is that control — a genuine learned
    cross-attention pooler. Unlike CAP it:
      - is NOT projection-free (W_q/W_k/W_v/out + MLP head);
      - pools token *features* into a d-dim vector, not cosine similarities to
        anchors into a K-d relative-representation profile.
    "cross" here = the learned query vs. the tokens (same modality); the two
    modalities are aligned by the contrastive loss, not by this attention. So
    CAP matching/beating this head — esp. in low-data, where its extra
    projections overfit — supports both the CAP design and the projection-free
    claim.

Forward contract mirrors the other alignment layers:
    input:  z (B, T, D) token features or (B, D) CLS embedding
            mask (B, T) optional — 1 = valid token, 0 = padding (text only)
    output: (B, dim_alignment) L2-normalized embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.alignment.alignment_factory import AlignmentFactory
from src.models.alignment.base_alignment_layer import BaseAlignmentLayer
from src.models.alignment.linear import _masked_mean_pool


@AlignmentFactory.register()
class AttnPoolAlignmentLayer(BaseAlignmentLayer):
    """Single-query multihead attention pooling (MAP) head (manual SDPA)."""

    def __init__(
        self,
        input_dim: int,
        dim_alignment: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: int = 4,
        normalize_output: bool = True,
    ):
        super().__init__(input_dim=input_dim)
        d = int(dim_alignment)
        if d % int(num_heads) != 0:
            raise ValueError(f"dim_alignment {d} not divisible by num_heads {num_heads}")
        self.dim_alignment = d
        self.num_heads = int(num_heads)
        self.head_dim = d // self.num_heads
        self.normalize_output = normalize_output

        # Project tokens to the working width; one learnable query cross-attends
        # them with explicit Q/K/V/out projections (standard multi-head attention).
        self.in_proj = nn.Linear(input_dim, d)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d)
        )
        self._reset_attention_parameters()

    def _reset_attention_parameters(self) -> None:
        """Match ``nn.MultiheadAttention``'s init exactly for the attention core.

        nn.MHA applies a single ``xavier_uniform_`` to the combined
        ``[W_q; W_k; W_v]`` (so the fan is that of the (3d, d) matrix, not three
        separate (d, d) ones), zeros the in-proj biases, and zeros ``out_proj``'s
        bias while leaving ``out_proj``'s weight at the default Linear init. We
        replicate that so this is textbook standard MHA, init included. The extra
        ``in_proj`` (input_dim->d) and the MLP head keep PyTorch's default Linear
        init, matching the linear/mlp baselines' projections.
        """
        d = self.dim_alignment
        with torch.no_grad():
            combined = torch.empty(3 * d, d)
            nn.init.xavier_uniform_(combined)
            self.q_proj.weight.copy_(combined[:d])
            self.k_proj.weight.copy_(combined[d:2 * d])
            self.v_proj.weight.copy_(combined[2 * d:])
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def _pool(self, kv: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        # kv: (B, T, d) projected tokens
        B, T, d = kv.shape
        H, hd = self.num_heads, self.head_dim
        q = self.q_proj(self.query.expand(B, -1, -1))      # (B, 1, d)
        k = self.k_proj(kv)                                # (B, T, d)
        v = self.v_proj(kv)                                # (B, T, d)
        q = q.view(B, 1, H, hd).transpose(1, 2)            # (B, H, 1, hd)
        k = k.view(B, T, H, hd).transpose(1, 2)            # (B, H, T, hd)
        v = v.view(B, T, H, hd).transpose(1, 2)            # (B, H, T, hd)
        scores = (q @ k.transpose(-2, -1)) / (hd ** 0.5)   # (B, H, 1, T)
        if mask is not None:
            m = (~mask.bool()).view(B, 1, 1, T)            # True = padding
            scores = scores.masked_fill(m, float("-inf"))
        attn = scores.softmax(dim=-1).nan_to_num(0.0)      # (B, H, 1, T); all-pad safe
        attn = self.attn_drop(attn)
        ctx = (attn @ v).transpose(1, 2).reshape(B, 1, d)  # (B, 1, d)
        pooled = self.out_proj(ctx).squeeze(1)             # (B, d)
        return pooled + self.mlp(self.ln(pooled))          # residual MLP head

    def forward(
        self, z: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if z.dim() == 2:
            # CLS fallback: treat the single embedding as a length-1 sequence.
            z = z.unsqueeze(1)
            mask = None
        kv = self.in_proj(z)                               # (B, T, d)
        out = self._pool(kv, mask)                         # (B, d)
        if self.normalize_output:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def reduce_for_structure_reg(self, z: torch.Tensor) -> torch.Tensor:
        """2D reduction for token-level structure_reg (unused at lambda=0)."""
        return _masked_mean_pool(z, None)
