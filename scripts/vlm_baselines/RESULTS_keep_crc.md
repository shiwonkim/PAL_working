# KEEP (Astaxanthin/KEEP) — CRC-100K zero-shot classification

2026 pathology VLM. ViT-L/16 vision + BERT text, 768-dim. Trained on "143K pathology semantic
groups" reorganized from millions of noisy pathology image-text pairs via a disease knowledge
graph (11,454 diseases). Loaded via AutoModel(trust_remote_code=True); encode_image / encode_text.

Eval = SAME protocol as CONCH/PLIP/QuiltNet/PAL:
- CRC-VAL-HE-7K (test split, 7,180 imgs) via ImageFolder, ADI..TUM = class 0..8
- 22 CONCH official templates; text prototype = mean of L2-normed (variant×template) embeds
- macro (balanced) + micro accuracy

Note on dataset naming: CRC100K == NCT-CRC-HE. NCT-CRC-HE-100K is the 100k TRAIN split;
CRC-VAL-HE-7K is the 7,180 TEST split (different patients). All our zs numbers use the TEST split.

Compat fix: KEEP's remote code monkeypatches timm's LayerScale with RenameLayerScale whose
__init__ lacked **kwargs; newer timm (1.0.28) passes device/dtype → TypeError. Added **kwargs to
RenameLayerScale.__init__ in the cached modeling_keep.py. (Re-download would revert the patch.)

## Results (macro / micro accuracy, %)

| protocol | macro (balanced) | micro |
|---|---|---|
| 22tmpl (single name) | 80.97 | 85.68 |
| 22tmpl + synonym (headline) | **87.84** | 91.95 |

## CRC headline table (22tmpl + synonym, macro acc)

| model | CRC macro | train data |
|---|---|---|
| **KEEP** | **0.878** | 143K knowledge-graph semantic groups (2026) |
| PAL (PathCap 200K) | 0.819 | 200K PathCap pairs, frozen UNI+PubMedBERT |
| MUSK | 0.799 | 50M img + 1B text tokens (unpaired) |
| CONCH | 0.756 | 1.17M pathology figure-caption pairs |

KEEP > PAL on CRC zs. KEEP is a purpose-built, knowledge-graph-curated 2026 SOTA; PAL's axis is
data efficiency + working atop ANY frozen encoder. TODO: cross-check KEEP's paper CRC number vs
this reproduction, and confirm which split the paper reports.
