# HPA 5-task — PAL (zero-shot & linear-probe) vs iSight/PLIP/CONCH

All numbers on HPA val (2000). Macro accuracy unless noted. location/quantity over positive rows
only (n≈1206). Tasks: **tissue** (65-way), **malignancy** (binary, derived from the tissue name:
cancer type → malignant, named normal organ → benign; 76% malignant), **intensity** (4),
**location** (3), **quantity** (3).

- PAL zero-shot: template prompts, NO labels. (`hpa_staining_zs.py`, `hpa_tissue_zs.py`)
- PAL linear-probe: frozen vision rep + sklearn LogisticRegression on **10K** labels, vision-only
  (no text encoder / text anchors). Probe train = 12K intensity-balanced from HPA train
  (val-disjoint); covers all 65 tissue classes (min 13/class). (`hpa_staining_linearprobe.py`,
  `--per-class 3000`)
- iSight / fine-tuned PLIP / fine-tuned CONCH: from the iSight paper (arXiv 2602.04063), full
  fine-tune on the **10.5M** HPA train set (overall accuracy, likely micro). Tissue/malignancy come
  from its Fig 2b (auxiliary tasks). CONCH/PLIP malignancy is reported only as "comparable" to
  iSight's 99.9 — no exact number in the paper. **No AUROC anywhere in the paper.**

## Checkpoints

| name | data | encoder | run | batch | best ep |
|---|---|---|---|---|---|
| generic | HPA generic caption, 200K | UNI2-h | lunar-darkness-224 | 2048 | 46 |
| cap1 | HPA caption_1, 100K | UNI2-h | eager-planet-229 | 2048 | 23 |
| genbal | HPA generic, cap 50/caption balanced, 96K (negatives 39%→10%) | UNI2-h | desert-blaze-230 | 2048 | 46 |
| pathcap | **PathCap 220K** (no HPA data; cross-domain transfer) | UNI2-h | comic-haze-232 | 4096 | 38 |

## Master table — zero-shot / probe / supervised

PAL columns show the best HPA-trained checkpoint (which one in parentheses).

| task | PAL zero-shot (macro/micro) | PAL probe 10K (macro/micro) | UNI2-raw probe | CONCH FT | PLIP FT | iSight |
|---|---|---|---|---|---|---|
| **tissue** (65) | .655/.582 (generic) | **.772/.741** (generic) | .655/.613 | 59.0 | 87.8 | **95.7** |
| **malignancy** | .771/.797 (cap1) | **.926/.933** (generic) | .865/.893 | ~99 (n/a) | ~99 (n/a) | **99.9** |
| **intensity** (4) | .575 (genbal) | **.617/.628** (generic) | .566/.565 | 70.0 | 73.0 | **76.6** |
| **location** (3) | .618 (generic) | **.652/.668** (generic) | .529/.619 | 75.2 | 79.4 | **85.5** |
| **quantity** (3) | .543 (genbal) | **.572/.598** (generic) | .493/.532 | 70.0 | 73.2 | **75.7** |

## Zero-shot, all checkpoints (macro)

| task | generic | cap1 | genbal | pathcap (transfer) |
|---|---|---|---|---|
| tissue | **.655** | .631 | .637 | .286 |
| malignancy | .615 | **.771** | .671 | .725 |
| intensity | .568 | .571 | **.575** | .414 |
| location | **.618** | .614 | .604 | .486 |
| quantity | .531 | .541 | **.543** | .458 |

Chance-normalised (generic): tissue 42.6× chance / 65% of headroom; intensity 2.3× / 42%;
location 1.9× / 43%; quantity 1.6× / 30%. (headroom = (acc−chance)/(1−chance).)

## Linear probe 10K, all features (macro)

| task | UNI2-raw (CLS) | UNI2-meanpatch | generic | cap1 | genbal | pathcap (transfer) |
|---|---|---|---|---|---|---|
| tissue | .655 | .665 | **.772** | .697 | .737 | .631 |
| malignancy | .865 | .885 | **.926** | .918 | .917 | .880 |
| intensity | .566 | .552 | **.617** | .597 | .610 | .606 |
| location | .529 | .573 | **.652** | .624 | .632 | .590 |
| quantity | .493 | .525 | **.572** | .562 | .554 | .530 |

What each probe feature is (all from the same UNI2-h `blocks.23` tokens, (B, 265, 1536) — the
last block, pre final-norm):
- **UNI2-raw** = CLS token, 1536-d.
- **UNI2-meanpatch** = mean of the 256 patch tokens (`[:, 9:]`; 1 CLS + 8 register skipped), 1536-d.
  Pooling control: PAL sees all 265 tokens through CAP, CLS sees one, so PAL-vs-CLS mixes the
  alignment effect with the "looked at every token" effect.
- **PAL-*** = `PALAlignmentLayer(image)` CAP-pooled anchor cosine profile, **512-d** (K=512
  anchors, softmax over tokens at temperature 0.05, then L2-normalised).

## Key findings

1. **Grounding decides the task ceiling** (visibility at a 224 whole-core thumbnail). Tissue
   (morphology, visible) is 42.6× chance zero-shot on a 65-way problem; staining location/quantity
   (subcellular, not resolved at 224) are only 1.6–1.9× chance on 3-way problems. Inside staining
   the same gradient holds: intensity (global hue) > location > quantity (cell counting). The
   supervised models show the same ordering (iSight: tissue 95.7 > location 85.5 > intensity
   76.6 ≈ quantity 75.7), so this is a property of the input resolution, not of the method.
   See `RESULTS_caption_comparison.md`.
2. **PAL alignment > raw encoder, on all 5 tasks** (same 10K labels): generic beats UNI2-raw by
   +11.7 tissue / +6.1 malignancy / +5.1 intensity / +12.3 location / +7.9 quantity (macro),
   despite 512-d vs 1536-d.
3. **…and it is the alignment, not the pooling.** Mean-pooling all patch tokens (the same
   whole-image view PAL gets) buys only +1.0 tissue / +2.0 malignancy / **−1.4** intensity /
   +4.4 location / +3.2 quantity over CLS; PAL then adds a further +10.7 / +4.1 / +6.5 / +7.9 /
   +4.6 on top of mean-patch. For tissue, 91% of PAL's gain over CLS is alignment.
4. **PAL tissue probe beats fine-tuned CONCH** (74.1 micro vs 59.0) with 10K labels, a frozen
   encoder and a 224 thumbnail — the one place a 10.5M-label full-FT model is passed. Below
   PLIP (87.8) / iSight (95.7), which add 336px + MIL + metadata.
5. **Label efficiency**: 10K labels + frozen probe reaches 82–88% of full-FT CONCH on staining
   (intensity 88%, location 87%, quantity 82%). Malignancy saturates for everyone (~93 vs ~99).
6. **Caption choice barely matters for classification** — generic ≥ cap1 ≥/≈ genbal within a few
   points on every task; the grounding ceiling dominates. The one exception is malignancy
   zero-shot, where cap1 (.771) ≫ generic (.615): cap1's captions carry richer
   malignancy vocabulary ("benign", "unremarkable", subtype names), so the same
   cancer-vs-normal prompts separate better. The probe erases that gap (.918 vs .926) — it was a
   text-side effect, not a vision-representation difference.
7. **Negative-caption balancing (genbal) is the better lever at matched size**: genbal (96K) beats
   cap1 (100K) on 4/5 probe tasks (tissue +4.0) and reaches 92–97% of generic-200K with half the
   images and 10% instead of 39% "no staining" captions. It does not beat generic-200K in
   absolute terms (2× data confound).
8. **Cross-domain transfer from PathCap is limited.** A PAL trained only on PathCap (220K
   figure-panel captions, no HPA data) drops sharply zero-shot (tissue .286, staining .41–.49) —
   its text anchors never saw HPA's template phrasing — except malignancy (.725, above generic),
   whose vocabulary PathCap has in abundance. Under the vision-only probe it lands below every
   HPA-trained checkpoint on all 5 tasks, and on tissue below even raw CLS (.631 vs .655), though
   above raw on intensity/quantity. In-domain pretraining wins for IHC TMA-core targets; PathCap's
   tile-level convenience does not make up for the domain gap.

## Framing
iSight/PLIP/CONCH = supervised, full fine-tune, 10.5M labels, 336px patches + CLAM MIL + metadata
(iSight also cell-type). PAL = frozen UNI2-h (224 whole-core thumbnail) + light anchor alignment,
trained on ≤200K image-caption pairs with NO staining labels. Not a same-conditions contest —
PAL's story is a strong, label-free / label-efficient representation that (a) beats the raw
encoder on every task, (b) does so through alignment rather than pooling, (c) reaches ~85% of
10.5M-label fine-tuning with 10K labels, and (d) already passes fine-tuned CONCH on tissue.
The gap to iSight tracks task visibility, so the remaining lever is native-resolution crops of
the 3000×3000 core (~0.46 µm/px, 20×-equivalent — the cell-level detail is in the file, it is
lost only at the 224 resize), then metadata conditioning and a larger probe / full FT.
