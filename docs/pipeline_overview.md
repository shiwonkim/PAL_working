# Pipeline overview — training & inference architecture

> Reference for understanding the codebase before/while refactoring.
> **File:line refs are as of 2026-06 (commit ~2ae0859)** and will drift as code
> changes — trust the *structure/concepts*, re-grep the exact lines.

---

## 0. Core design philosophy

**The encoders (ViT / RoBERTa) never run inside the training loop.** All features
are pre-extracted and cached to disk; only the **alignment layer** is trained, on
top of the cached feature tensors.

Consequences (these explain almost every design choice):
- Each epoch has no encoder forward → very fast → 1000-epoch default is feasible.
- The full feature tensor is held in memory and indexed per batch → large token
  caches cause RAM pressure (this is the root of the LAION memory issue — see
  `docs/laion_reimplementation_TODO.md`).
- A separate alignment layer is built **per modality** (same class, two instances):
  `alignment_image`, `alignment_text`, created by `AlignmentFactory` from
  `config.training.alignment_layer_name`
  (`src/trainers/alignment_trainer.py` ~1839).
- **CLS vs Token** is a single axis that runs through extraction → training →
  every eval path (see §5).

---

## 1. Training pipeline

Entry: `python src/train_alignment.py --config_path X.yaml`

```
src/train_alignment.py
  ① load + merge(defaults, overrides)                          ~103
  ② load_dataset(): coco/coco2017/flickr30 = caption datasets, ~18-79
     others wrapped in ImageTextDataset
  ③ load eval datasets (zero_shot / retrieval)                 ~176-194
  ④ dispatch:                                                  ~205-210
       training.cca=true  → CSATrainer (closed-form CCA)
       training.clip=true → CLIPEvalTrainer
       else               → AlignmentTrainer  ← main path
```

### AlignmentTrainer.fit()  (`src/trainers/alignment_trainer.py` ~1181-2061)

**(a) Feature extraction & caching** — `get_image_features` (~282-425) /
`get_text_features` (~172-280)
- pool mode decides shape:
  - `pool_*="cls"` → CLS token → `(N, D)`        (CLS methods)
  - `pool_*="avg"` → token mean → `(N, D)`
  - `pool_*="none"` + `layer_*` → single-layer tokens → `(N, T, D)`  (token methods)
- cache path `results/features/{model}-{dataset}-{suffix}.npy`; reload via `mmap=True`.
- **image dedup at extraction** (~322-358): COCO has one row per caption → images
  repeat ~5×; extraction keeps unique images + an index map (5× disk/time saved).

**(b) Layer selection** — `compute_layer_alignment` (~427-570)
- Runs only when config lacks `layer_img/txt`. **Mutual-kNN** scores every
  `(img_layer × txt_layer)` pair; `best_only` keeps the top pair.
  COCO2014: ViT-L→(23,24), ViT-S→(11,6), ViT-B→(11,12).

**(c) fit loop** (~1499-2061), per selected layer pair:
- slice features at layer; if token_level → `_load_token_features_for_layer`
  (~710-814) loads `(N,T,D)` + text mask `(N,T)`.
- build alignment layers via factory; build loss; optional LR finder
  (`base_trainer.py` ~147-314); epoch loop.
- **one batch** (`train()`, ~2065-2349):
  ```
  image_feats = image_features[i:end].to(device)          # index the cache
  aligned_img = alignment_image(image_feats[, cls_attn])  # (B,D|T,D) → (B,K)
  aligned_txt = alignment_text(text_feats[, mask])        # (B,D|T,D) → (B,K)
  loss = CLIPLoss(aligned_img, aligned_txt, ...)
  loss.backward(); clip_grad; opt.step(); sched.step()
  ```
- early stopping; best val → `save_checkpoint` stores `alignment_image`,
  `alignment_text`, optimizer, and the full `config` (~2014).

**Token detail worth knowing:** when token_level, features start as a CLS stub
`(N,1)` and are swapped for real `(N,T,D)` at train time — hence the
`image_dim = features.shape[-1]` recompute (~1818); without it `input_dim` would
be 1.

### CSATrainer (`src/trainers/csa_trainer.py` ~80-226)
Closed-form CCA (`NormalizedCCA`, `src/alignment/cca_class.py`); no token-level,
no gradient training — fits canonical variates analytically, then evaluates.

---

## 2. Alignment layer zoo

All subclass `BaseAlignmentLayer(input_dim)` (`src/alignment/base_alignment_layer.py`),
implement `forward(z, mask?)`, registered via `@AlignmentFactory.register()`
(`alignment_factory.py` + auto-import in `__init__.py` via
`initialize_package_factory`, `src/utils/base_factory.py`). Output is always
**`(B, K)` L2-normalized profile** → fed to CLIPLoss. Trainer builds one per
modality and calls `set_modality(...)` if present (~1839-1856).

| method | class (file) | forward gist |
|---|---|---|
| Linear | `LinearAlignmentLayer` (`linear_alignment_layer.py`) | `Linear(z)`; 3D → masked mean-pool; opt L2 |
| MLP | `MLPAlignmentLayer` (`mlp_alignment_layer.py`) | stacked ReLU MLP |
| STRUCTURE "MLP" | `ResLowRankHead` (`mlp_alignment_layer.py` ~51) | skip `P(z)` + gated low-rank residual `α·W₂(GELU(W₁z))`, `α=σ(logit)` learned |
| **PAL-CLS** | `PALAlignmentLayer` (`pal.py` ~57) | `normalize(normalize(z) @ normalize(anchors)ᵀ)` → `(B,K)` cosine profile |
| **PAL-Token ★** | `PALTokenAlignmentLayer` (`pal_token.py` ~116) | **CAP**, see below |
| FreezeAlign | `FreezeAlignAlignmentLayer` (`freeze_align.py` ~195) | `set_modality`; img = patch-mean + CLS proj; txt = masked-mean + MLP |
| SAIL | `SAILStarMLP` (`sail_star_mlp.py` ~81) | `set_modality`; GLU `g(ReLU6(f₁z)⊙f₂z)`, concat[cls,patch] |
| CSA | `cca_class.py` | closed-form CCA; NOT an nn layer / not factory-registered |

### ★ CAP — Cross-Attention Pooling (`pal_token.py` ~116-184)
K learnable anchors `(K,D)` **are** the alignment params (`projector_dim=0`, no
downstream MLP). For token input `z:(B,T,D)`:
```
ẑ = normalize(z, -1)                  # (B,T,D)
â = normalize(anchors, -1)            # (K,D)
sim = ẑ @ âᵀ                          # (B,T,K)  token×anchor cosine
logits = sim / pool_temperature       # τ=0.03 default
[ + betaₖ · log(cls_attn) ]           # optional cls_attn_prior, per-anchor bias
logits = logits.masked_fill(~mask, -inf)
attn = softmax(logits, dim=1)         # (B,T,K)  each anchor soft-selects tokens
profile = (attn * sim).sum(dim=1)     # (B,K)    per-anchor weighted sum
return normalize(profile, -1)
```
2D (CLS) input → falls back to the PAL-CLS path (~124-128). This is the layer being
refactored; the cls_attn-prior branch is inlined in forward.

---

## 3. Loss (`src/loss/clip_loss.py`, `siglip_loss.py`)
```
logits = aligned_img @ aligned_txtᵀ / temperature
loss = (CE(logits, arange) + CE(logitsᵀ, arange)) / 2          # bidirectional InfoNCE
     + structure_lambda · structure_reg(...)                   # optional
```
- **STRUCTURE regularizer** (`clip_loss.py` ~57-152): JS-divergence penalty that
  preserves the original embeddings' similarity structure after alignment (mean-
  pools 3D first). `structure_lambda` (~10), `structure_levels`, warmup.
- **SigLipLoss** alt: per-pair sigmoid with learnable `logit_scale`/`logit_bias`.

---

## 4. Inference / evaluation (all consume the trained alignment_image/text)

Common: load ckpt → `alignment_{image,text}.eval()` → `set_modality` if present.

**(a) Zero-shot classification** — `zero_shot_classifier.py:build_zero_shot_classifier`
(~34) + `AlignmentTrainer.evaluate_zero_shot_classification` (~2516)
- classnames × templates → text embeds → `alignment_text` → average templates →
  `(num_classes, K)` classifier; image → `alignment_image` → `(B,K)`;
  `100·img @ classifierᵀ` → argmax.
- **`token_level_zero_shot` is decisive**: true → build class templates via the
  **CAP/token path** (`pool_txt="none"` + mask); false → CLS path. Wrong path →
  distribution mismatch → random-looking scores. Token methods must set true.

**(b) Retrieval** — `retrieval.py:retrieval_metrics_df` (~29) +
`evaluate_retrieval` (~2854)
- features → alignment → `(N,K)` → normalize → `img @ txtᵀ` → topk →
  I2T/T2I R@1/5/10, MAP@k. Multi-caption GT grouped by `image_name`.

**(c) Segmentation** — `zero_shot_segmentation.py` (CLI `main` ~1329)
- `SegmentationMethod` subclasses: `direct_cosine`, **`anchor_codebook`
  (PALAnchorCodebookMethod)**, `freezealign`, `linear_perpatch`.
- **factorized decoding**: `S_pa (P,K) @ S_ac (K,C) = (P,C)` — anchors as explicit
  bridge; vs **direct**: normalized patches @ text.
- patch sims → `√P×√P` grid → bilinear upsample → argmax → **mIoU-fg** (foreground-
  factorized). `auto_filter_methods` runs only ckpt-compatible methods.

**Standalone entry points:**
- `rerun_eval.py` (~46): re-eval a ckpt; auto-detects layers from ckpt path
  `(\d+, \d+)`, `--token_level_zs` override (merged from both servers).
- `zero_shot_segmentation.py:main`: segmentation CLI.

---

## 5. The CLS ↔ Token axis (cross-cutting)

| stage | CLS (`token_level=false`) | Token (`token_level=true`) |
|---|---|---|
| extraction pool | cls/avg → `(N,D)` | none → `(N,T,D)` + mask |
| alignment forward | `layer(z)` 2D | `layer(z, mask)` CAP |
| ZS templates | CLS path | CAP path (`token_level_zero_shot`) |
| methods | Linear / MLP / CSA / PAL-CLS | PAL-Token / FreezeAlign |

The same "if token: mask path; else: 2D path" branch is re-implemented in
extraction, fit, and each eval — a prime consolidation target.

---

## 6. Refactoring observations (starting points)

1. **`alignment_trainer.py` is huge** (~2900 lines; `fit()` alone ~800): extraction
   + layer selection + token/CLS branching + dedup + subsample + cls_attn + LR
   finder + train/validate + 3 eval types in one class. Split responsibilities.
2. **CLS/Token branching is duplicated** across extraction/fit/zs/retrieval/seg —
   factor into a shared abstraction.
3. **Feature I/O is embedded in the trainer** (cache load / mmap / dedup inside
   fit). Extracting a **FeatureStore** abstraction would both clean this up and be
   the natural home for the LAION memory reimplementation (see
   `docs/laion_reimplementation_TODO.md`: virtual-concat + mmap + buffer-shuffle +
   prefetch).
4. **Pool mode / layer slicing via in-place `config[...]=` overrides** in several
   places → side-effect risk; pass explicit params instead.
5. **CAP layer** (`pal_token.py`) is relatively clean; the cls_attn-prior
   branch is inlined in forward and could be separated.

---

## Quick file map

| area | file |
|---|---|
| entry | `src/train_alignment.py` |
| trainers | `src/trainers/{alignment_trainer,base_trainer,csa_trainer,clip_eval_trainer}.py` |
| alignment layers | `src/alignment/*.py` (+ `alignment_factory.py`, `base_alignment_layer.py`) |
| losses | `src/loss/{clip_loss,siglip_loss}.py` |
| eval | `src/evaluation/{zero_shot_classifier,retrieval,zero_shot_segmentation}.py` |
| re-eval CLI | `rerun_eval.py` |
| config | `configs/default.yaml` + `configs/<method>/<encoder>/*.yaml` |
