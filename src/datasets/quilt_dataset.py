"""Quilt-1M histopathology image-caption dataset.

Quilt-1M (Ikezogwo et al., NeurIPS 2023) ships as a single lookup CSV
(``quilt_1M_lookup.csv``) plus ``images_part_*_resized.zip`` archives whose
images extract flat, named by the CSV's ``image_path`` column. This mirrors
:class:`~src.datasets.coco_dataset.CocoCaptionDataset` — same ``self.df`` schema
(``image_path``, ``captions``) and the same ``__getitem__`` / ``load_image`` /
``LoadingType`` machinery — but builds the sample table from the CSV instead of
a COCO-style JSON, so it drops straight into the FeatureStore's image/text
extraction passes.
"""

import os

import pandas as pd
from loguru import logger

from .coco_dataset import CocoCaptionDataset, LoadingType


class QuiltCaptionDataset(CocoCaptionDataset):
    def __init__(
        self,
        csv_file: str = None,
        image_dir: str = None,
        transform=None,
        tokenizer=None,
        loading_type: LoadingType = LoadingType.STANDARD,
        split: str = "train",
        subsets=None,
        selection_path: str = None,
        **kwargs,
    ):
        # NOTE: intentionally does not call super().__init__ (that one parses a
        # COCO JSON); we reuse the parent's __getitem__/load_image/apply_tokenizer
        # and only replace how ``self.df`` is populated.
        self.annotation_file = selection_path or csv_file
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.loading_type = loading_type

        if selection_path is not None:
            # Pre-built selection (see quilt_selection.py): a filtered/subsampled
            # (image_path, caption[, source]) table. Loaded verbatim — no split /
            # subset / re-filtering — so training sets are exactly reproducible.
            sel = pd.read_csv(selection_path).dropna(subset=["caption", "image_path"])
            image_paths = sel["image_path"].apply(lambda p: os.path.join(image_dir, p))
            self.df = pd.DataFrame(
                {"image_path": image_paths.values, "captions": sel["caption"].values}
            )
            self.df.reset_index(drop=True, inplace=True)
            logger.info(f"Quilt-1M selection: {len(self.df)} pairs from {selection_path}")
        else:
            # Raw lookup CSV with optional split/subset selection (filtering OFF by
            # default -> full pool, as needed by the CONCH overlap diagnostic).
            df = pd.read_csv(csv_file)
            n0 = len(df)
            if split is not None and "split" in df.columns:
                df = df[df["split"] == split]
            if subsets is not None and "subset" in df.columns:
                df = df[df["subset"].isin(subsets)]
            df = df.dropna(subset=["caption", "image_path"])

            image_paths = df["image_path"].apply(lambda p: os.path.join(image_dir, p))
            self.df = pd.DataFrame(
                {"image_path": image_paths.values, "captions": df["caption"].values}
            )
            self.df.dropna(subset="captions", inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            logger.info(
                f"Quilt-1M: {n0} rows -> {len(self.df)} "
                f"(split={split}, subsets={subsets})"
            )
        self.apply_tokenizer()

        # PIL-only decode (load_image's fallback path). Quilt mixes .jpg/.png so
        # TurboJPEG would fall back for PNGs anyway; feature extraction is a
        # single pass, so the fast path isn't worth the mixed-format handling.
        self.jpeg_reader = None
