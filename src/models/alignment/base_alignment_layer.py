from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseAlignmentLayer(ABC, nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    @abstractmethod
    def forward(
        self, z: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def reduce_for_structure_reg(self, z: torch.Tensor) -> torch.Tensor:
        """Pool 3D token features ``(B, T, D)`` to 2D ``(B, D)`` for structure_reg.

        NOT provided by default: structure_reg only makes sense if a layer pools
        its ``original`` the same way its forward pools the ``aligned``, so each
        supporting layer must override this to match its own architecture. A
        layer that does not override cannot be used with token-level structure_reg
        (the trainer only calls this for 3D inputs when structure_reg is active).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support token-level structure_reg. "
            "Override reduce_for_structure_reg to pool tokens to 2D consistently "
            "with this layer's forward."
        )
