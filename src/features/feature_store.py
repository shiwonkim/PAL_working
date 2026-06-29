"""FeatureStore — the single home for feature-cache path building and I/O.

Goal 3 of the refactor pulls feature extraction / caching / mmap-loading out of
``AlignmentTrainer`` so that extraction, training, and evaluation become separable
stages over a shared cache (and so the LAION / continual-learning memory work has
a clean home). It consumes a :class:`~src.features.feature_spec.FeatureSpec` for the
CLS-vs-token policy.

State: ``save_path`` (cache root), ``device``, and ``config`` (the SAME dict the
trainer holds — in-place pool/layer overrides must stay visible here). Encoders
are built lazily and only when a cache miss forces extraction; loads are
``mmap``-backed so large token caches don't blow committed RAM.

The cache-path contract (``<save_path>/features/<model>-<dataset>-<suffix>.npy``)
is preserved byte-for-byte — ~795 GB of caches exist under
``~/STRUCTURE/results/features`` and must keep resolving. The suffix string is
still supplied by callers; unifying suffix construction through FeatureSpec is a
later brick.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.coco_dataset import LoadingType
from src.models.encoders.text_models import load_llm, load_tokenizer
from src.models.encoders.vision_models import load_lvm
from src.utils.utils import set_transform_dataset


class FeatureStore:
    """Feature-cache path/IO. Cache layout: ``<save_path>/features/<file>.npy``."""

    def __init__(self, save_path: Union[str, Path], device, config: dict):
        self.save_path = save_path
        self.device = device
        self.config = config

    # ------------------------------------------------------------------
    # Cache-path contract (stateless)
    # ------------------------------------------------------------------
    @staticmethod
    def model_name(m_name: str) -> str:
        """Sanitise an encoder name into the cache filename stem."""
        return m_name.replace("/", "_").replace("-", "_")

    @staticmethod
    def cache_path(
        m_name: str,
        d_name: str,
        save_path: Union[str, Path],
        suffix: str = "",
    ) -> Path:
        """Path of the cache file for ``(model, dataset, suffix)``."""
        stem = f"{FeatureStore.model_name(m_name)}-{d_name}-{suffix}.npy"
        return Path(save_path) / "features" / stem


    # ------------------------------------------------------------------
    # Encoders (built only on cache miss)
    # ------------------------------------------------------------------
    def get_llm(self, llm_model_name: str):
        language_model = load_llm(llm_model_name)
        # since we're using huggingface's automapping
        # we don't need to move it to the device
        language_model = language_model.eval()
        tokenizer = load_tokenizer(llm_model_name)
        return language_model, tokenizer

    def get_lvm(self, lvm_model_name: str):
        # Thin wrapper over encoders.vision_models.load_lvm; the store only
        # supplies img_size + device (mirrors get_llm wrapping load_llm).
        return load_lvm(
            lvm_model_name,
            img_size=self.config["features"].get("img_size"),
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Extract-or-load
    # ------------------------------------------------------------------
    def get_text_features(
        self,
        loader,
        llm_model_name: str,
        suffix: str = "",
        dataset_name: Optional[str] = None,
        pool: Optional[str] = None,
        layer_index: Optional[int] = None,
    ):
        # pool / layer default to the config (back-compat); callers may pass them
        # explicitly so the in-place config override is no longer needed.
        pool_txt_mode = pool if pool is not None else self.config["features"]["pool_txt"]
        layer_txt = (
            layer_index if layer_index is not None
            else self.config["features"].get("layer_txt")
        )
        if hasattr(loader.dataset, "name"):
            dataset_name = loader.dataset.name
        elif dataset_name is None:
            dataset_name = type(loader.dataset).__name__
        save_path = FeatureStore.cache_path(
            m_name=llm_model_name,
            d_name=dataset_name,
            save_path=self.save_path,
            suffix=suffix,
        )


        if save_path.exists():
            # mmap=True: feature cache is file-backed, pages shared via
            # OS page cache between concurrent training processes. Drops
            # per-process Committed_AS by ~50–100 GB on ViT-L/RoBERTa-L
            # runs and avoids blowing Server B's CommitLimit of ~136 GB.
            llm_feats = torch.load(
                str(save_path), weights_only=False, mmap=True
            )["features"]
            logger.debug(f"Loaded features from: {save_path}")
            return llm_feats

        language_model, tokenizer = self.get_llm(llm_model_name=llm_model_name)
        loader.dataset.tokenizer = tokenizer
        if hasattr(loader.dataset, "loading_type"):
            # for optimizing the loading and looping
            loader.dataset.loading_type = LoadingType.TXT_ONLY
        _df = loader.dataset.df
        loader.dataset.apply_tokenizer()
        # ensure this is still the same ordering
        assert loader.dataset.df.equals(_df)
        del _df

        llm_feats = None
        offset = 0
        total_n = len(loader.dataset)
        for batch in tqdm(loader, total=len(loader), file=sys.stdout):
            _, token_inputs = batch
            token_inputs = {
                k: v.to(self.device).long() for (k, v) in token_inputs.items()
            }
            with torch.no_grad():
                if "olmo" in llm_model_name.lower():
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                        output_hidden_states=True,
                    )
                else:
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                    )
                if pool_txt_mode == "avg":
                    # swap the backsize to the first dimension
                    # (BS, Layers, Tokens, Dim)
                    feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3)
                    # make the mask compatible with the dimension
                    mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)
                    # average along the token dimension
                    feats = (feats * mask).sum(2) / mask.sum(2)
                elif pool_txt_mode == "last":
                    feats = [v[:, -1, :] for v in llm_output["hidden_states"]]
                    feats = torch.stack(feats).permute(1, 0, 2)
                elif pool_txt_mode == "none":
                    assert layer_txt is not None
                    feats = torch.stack(list(llm_output["hidden_states"]))
                    # permute to dim: (bs, layers, tokens, dim)
                    feats = feats.permute(1, 0, 2, 3)
                    # select only the layer we care about, otherwise we don't have enough memory
                    feats = feats[:, layer_txt, :, :]
                else:
                    raise NotImplementedError(f"unknown pooling {pool_txt_mode}")

                if pool_txt_mode == "none":
                    feats_cpu = feats.to(dtype=torch.float16).cpu()
                    if llm_feats is None:
                        _, T, D = feats_cpu.shape
                        llm_feats = torch.empty(
                            (total_n, T, D), dtype=torch.float16
                        )
                    bs = feats_cpu.shape[0]
                    llm_feats[offset : offset + bs] = feats_cpu
                    offset += bs
                else:
                    if llm_feats is None:
                        llm_feats = []
                    llm_feats.append(feats.cpu())
        if pool_txt_mode == "none":
            if offset < total_n:
                llm_feats = llm_feats[:offset]
        else:
            llm_feats = torch.cat(llm_feats).cpu()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {"features": llm_feats}
        if hasattr(loader.dataset, "df"):
            save_dict["dataframe"] = loader.dataset.df
        torch.save(save_dict, save_path)
        logger.debug(f"Saved features to: {save_path}")
        del language_model
        return llm_feats

    def load_or_build_text_mask(
        self, loader, llm_model_name: str, suffix: str,
    ) -> torch.Tensor:
        """Load cached text attention mask, or tokenise the loader to build one.

        The main ``get_text_features`` call doesn't persist masks — we write a
        companion file next to the features cache with the same base name
        plus ``_mask`` suffix.
        """
        dataset_name = (
            loader.dataset.name
            if hasattr(loader.dataset, "name")
            else type(loader.dataset).__name__
        )
        features_path = FeatureStore.cache_path(
            m_name=llm_model_name,
            d_name=dataset_name,
            save_path=self.save_path,
            suffix=suffix,
        )
        mask_path = features_path.with_name(
            features_path.stem + "_mask" + features_path.suffix
        )
        if mask_path.exists():
            payload = torch.load(mask_path, weights_only=False)
            logger.debug(f"Loaded text mask from: {mask_path}")
            return payload["mask"]

        # Build masks by re-running the tokenizer over the dataloader.
        _, tokenizer = self.get_llm(llm_model_name=llm_model_name)
        loader.dataset.tokenizer = tokenizer
        if hasattr(loader.dataset, "loading_type"):
            loader.dataset.loading_type = LoadingType.TXT_ONLY
        loader.dataset.apply_tokenizer()

        masks = []
        for batch in tqdm(
            loader, total=len(loader), file=sys.stdout, desc=f"text-mask[{suffix}]"
        ):
            _, token_inputs = batch
            masks.append(token_inputs["attention_mask"].cpu())
        mask = torch.cat(masks, dim=0)

        mask_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mask": mask}, mask_path)
        logger.debug(f"Saved text mask to: {mask_path}")
        return mask

    def get_image_features(
        self,
        loader,
        lvm_model_name: str,
        suffix: str = "",
        dataset_name: Optional[str] = None,
        allow_image_dedup: bool = True,
        pool: Optional[str] = None,
        layer_index: Optional[int] = None,
    ):
        # pool / layer default to the config (back-compat); callers may pass them
        # explicitly so the in-place config override is no longer needed.
        pool_mode = pool if pool is not None else self.config["features"]["pool_img"]
        layer_img = (
            layer_index if layer_index is not None
            else self.config["features"].get("layer_img")
        )
        if hasattr(loader.dataset, "name"):
            dataset_name = loader.dataset.name
        elif dataset_name is None:
            dataset_name = type(loader.dataset).__name__
        save_path = FeatureStore.cache_path(
            m_name=lvm_model_name,
            d_name=dataset_name,
            save_path=self.save_path,
            suffix=suffix,
        )


        if save_path.exists():
            # mmap=True: see get_text_features cache load for rationale.
            lvm_feats = torch.load(
                str(save_path), weights_only=False, mmap=True
            )["features"]
            logger.debug(f"Loaded features from: {save_path}")
            return lvm_feats

        vision_model, image_transform = self.get_lvm(lvm_model_name=lvm_model_name)
        set_transform_dataset(
            dataset=loader.dataset,
            image_transform=image_transform,
        )

        # Image-side dedup at extraction: pool=none caches duplicate the same
        # image 5x in COCO (one row per caption). Detect that and iterate
        # only the first-occurrence rows of each unique image_path. Saves
        # ~5x extraction time and ~5x disk on the train cache. fit() detects
        # the deduped layout via shape and skips the redundant per-row
        # dedup it would otherwise apply.
        is_deduped_extraction = False
        unique_to_full_idx = None
        if (
            pool_mode == "none"
            and allow_image_dedup
            and self._should_dedup_image_extraction(loader)
        ):
            df = loader.dataset.df
            first_idx_mask = (df.groupby("image_path").cumcount() == 0).values
            keep_indices = np.where(first_idx_mask)[0].tolist()
            n_unique = len(keep_indices)
            n_full = len(df)
            logger.info(
                f"Image dedup at extraction: {n_full:,} caption-image pairs -> "
                f"{n_unique:,} unique images "
                f"(saves {(1 - n_unique / n_full) * 100:.1f}% of vision forwards)"
            )
            unique_view = self._indexed_dataset_view(loader.dataset, keep_indices)
            from torch.utils.data import DataLoader as _DataLoader

            iter_loader = _DataLoader(
                unique_view,
                batch_size=loader.batch_size,
                shuffle=False,
                num_workers=getattr(loader, "num_workers", 0),
                pin_memory=False,
                drop_last=False,
                collate_fn=getattr(loader, "collate_fn", None),
            )
            iter_total_n = n_unique
            # Build the caption -> image-row mapping for the sidecar
            unique_paths = df.loc[first_idx_mask, "image_path"].reset_index(drop=True)
            path_to_pos = {p: i for i, p in enumerate(unique_paths)}
            unique_to_full_idx = torch.tensor(
                df["image_path"].map(path_to_pos).values, dtype=torch.long
            )
            is_deduped_extraction = True
        else:
            iter_loader = loader
            iter_total_n = len(loader.dataset)

        # Streaming allocation path for pool=none: token features are huge
        # (591K * 257 * 384 * 4 = 234 GB for full COCO float32), so we
        # pre-allocate a single float16 tensor and fill it by offset to
        # avoid torch.cat's 2x peak memory.
        lvm_feats = None
        offset = 0
        total_n = iter_total_n
        for batch in tqdm(iter_loader, total=len(iter_loader), file=sys.stdout):
            images, _ = batch
            with torch.no_grad():
                images = images.to(self.device, non_blocking=True)
                lvm_output = vision_model(images)
                if pool_mode == "cls":
                    # extract the class token for all layers
                    feats = [v[:, 0, :] for v in lvm_output.values()]
                    feats = torch.stack(feats).permute(1, 0, 2)
                elif pool_mode == "none":
                    assert layer_img is not None
                    feats = torch.stack(list(lvm_output.values()))
                    # permute to dim: (bs, layers, tokens, dim)
                    feats = feats.permute(1, 0, 2, 3)
                    # select only the layer we care about, otherwise we don't have enough memory
                    feats = feats[:, layer_img, :, :]
                else:
                    raise NotImplementedError(f"unknown pooling {pool_mode}")

                if pool_mode == "none":
                    feats_cpu = feats.to(dtype=torch.float16).cpu()
                    if lvm_feats is None:
                        _, T, D = feats_cpu.shape
                        lvm_feats = torch.empty(
                            (total_n, T, D), dtype=torch.float16
                        )
                    bs = feats_cpu.shape[0]
                    lvm_feats[offset : offset + bs] = feats_cpu
                    offset += bs
                else:
                    if lvm_feats is None:
                        lvm_feats = []
                    lvm_feats.append(feats.cpu())
        if pool_mode == "none":
            if offset < total_n:
                lvm_feats = lvm_feats[:offset]
        else:
            lvm_feats = torch.cat(lvm_feats).cpu()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {"features": lvm_feats}
        if is_deduped_extraction:
            save_dict["is_image_deduped"] = True
            save_dict["unique_to_full_idx"] = unique_to_full_idx
            # Persist the unique-row df so future loaders can rebuild
            # mappings without re-reading the original annotations.
            save_dict["dataframe"] = (
                loader.dataset.df.loc[
                    loader.dataset.df.groupby("image_path").cumcount() == 0
                ].reset_index(drop=True)
            )
        elif hasattr(loader.dataset, "df"):
            save_dict["dataframe"] = loader.dataset.df
        torch.save(save_dict, save_path)
        logger.debug(f"Saved features to: {save_path}")
        del vision_model
        return lvm_feats

    # ------------------------------------------------------------------
    # Image-dedup helpers (extraction-time)
    # ------------------------------------------------------------------
    def _indexed_dataset_view(self, base_dataset, keep_indices):
        """Dataset proxy that yields only the rows in ``keep_indices``.

        Mirrors ``_SubsetView`` (read+write attribute delegation, df
        property override) but supports an arbitrary index list instead
        of a first-N truncation. Used by image-side dedup at extraction.
        """

        class _IndexedView:
            _PROXY_ATTRS = {"_dataset", "_indices"}

            def __init__(self, dataset, indices):
                object.__setattr__(self, "_dataset", dataset)
                object.__setattr__(self, "_indices", list(indices))

            def __len__(self):
                return len(self._indices)

            def __getitem__(self, i):
                return self._dataset[self._indices[i]]

            def __getattr__(self, name):
                return getattr(self._dataset, name)

            def __setattr__(self, name, value):
                if name in type(self)._PROXY_ATTRS:
                    object.__setattr__(self, name, value)
                else:
                    setattr(self._dataset, name, value)

            @property
            def df(self):
                inner_df = getattr(self._dataset, "df", None)
                if inner_df is None:
                    return None
                return inner_df.iloc[self._indices].reset_index(drop=True)

        return _IndexedView(base_dataset, keep_indices)

    def _should_dedup_image_extraction(self, loader) -> bool:
        """Decide whether to apply image-dedup-at-extraction for this loader.

        Returns True only when:
          - the config flag ``features.image_dedup_extraction`` is on,
          - the dataset has a ``df`` with an ``image_path`` column,
          - that df has duplicate image_paths,
          - training.drop_duplicates is true and n_dup_samples == 1
            (so the trainer would otherwise dedup to first-occurrence anyway),
          - no n_random_subsample_train cap is set (the dryrun first-N path
            doesn't benefit and would interact awkwardly with dedup).
        """
        if not self.config["features"].get("image_dedup_extraction", False):
            return False
        ds = loader.dataset
        if not hasattr(ds, "df"):
            return False
        df = ds.df
        if df is None or "image_path" not in df.columns:
            return False
        if not df["image_path"].duplicated().any():
            return False
        if not self.config["training"].get("drop_duplicates", False):
            return False
        if int(self.config["training"].get("n_dup_samples", 1)) != 1:
            return False
        if self.config["training"].get("n_random_subsample_train") is not None:
            return False
        return True
