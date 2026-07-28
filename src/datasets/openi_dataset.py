"""OpenI / Indiana University chest-X-ray (IU-Xray) image-report dataset.

OpenI ships ~3.9k de-identified radiology reports as ``ecgen-radiology/*.xml``
plus ~7.5k PNG images. Each report XML carries ``<AbstractText Label="FINDINGS">``
/ ``IMPRESSION`` sections and one or more ``<parentImage id="CXRxxx_IM-...">``
references that map to ``<image_dir>/<id>.png``. This mirrors
:class:`~src.datasets.coco_dataset.CocoCaptionDataset` (same ``self.df`` schema
and ``__getitem__`` / ``load_image`` / ``LoadingType``), building the pair table
from the report XMLs instead of a COCO JSON, so it feeds the FeatureStore
directly for the CXR PAL track.
"""

import glob
import hashlib
import os
import xml.etree.ElementTree as ET

import pandas as pd
from loguru import logger

from .coco_dataset import CocoCaptionDataset, LoadingType


def _report_id(image_path: str) -> str:
    # "CXR1000_IM-0003-1001.png" -> "CXR1000" (split train/val by report, not
    # image, so a report's frontal+lateral views never straddle the split).
    return os.path.basename(image_path).split("_")[0]


class OpenICaptionDataset(CocoCaptionDataset):
    def __init__(
        self,
        reports_dir: str,
        image_dir: str,
        transform=None,
        tokenizer=None,
        loading_type: LoadingType = LoadingType.STANDARD,
        text_fields=("FINDINGS", "IMPRESSION"),
        split: str = "all",
        val_frac: float = 0.1,
        seed: int = 42,
        **kwargs,
    ):
        self.annotation_file = reports_dir
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.loading_type = loading_type

        rows = []
        for xml_path in sorted(glob.glob(os.path.join(reports_dir, "*.xml"))):
            try:
                root = ET.parse(xml_path).getroot()
            except ET.ParseError:
                continue
            texts = {}
            for ab in root.iter("AbstractText"):
                label = ab.get("Label")
                if label:
                    texts[label] = (ab.text or "").strip()
            caption = " ".join(texts.get(f, "") for f in text_fields).strip()
            if not caption:
                continue
            for parent in root.iter("parentImage"):
                iid = parent.get("id")
                if iid:
                    rows.append((os.path.join(image_dir, iid + ".png"), caption))

        df = pd.DataFrame(rows, columns=["image_path", "captions"])
        n_pairs = len(df)
        df = df[df["image_path"].map(os.path.exists)].reset_index(drop=True)

        if split in ("train", "val"):
            # Deterministic report-level split: hash(report_id + seed) -> [0,1).
            def _in_val(p):
                h = hashlib.md5(f"{_report_id(p)}-{seed}".encode()).hexdigest()
                return (int(h[:8], 16) / 0xFFFFFFFF) < val_frac
            in_val = df["image_path"].map(_in_val)
            df = df[in_val] if split == "val" else df[~in_val]
            df = df.reset_index(drop=True)

        self.df = df
        logger.info(
            f"OpenI: {n_pairs} pairs -> {len(self.df)} "
            f"(image-exists, split={split})"
        )
        self.apply_tokenizer()
        self.jpeg_reader = None
