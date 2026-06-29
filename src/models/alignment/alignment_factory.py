from src.models.alignment.base_alignment_layer import BaseAlignmentLayer
from src.utils.base_factory import BaseFactory


class AlignmentFactory(BaseFactory):
    """The factory class for creating various alignment layers."""

    base_class = BaseAlignmentLayer
    registry = {}
