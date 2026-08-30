# HPA staining — full comparison: PAL (zero-shot & linear-probe) vs iSight/PLIP/CONCH

All numbers macro accuracy on HPA val (2000). location/quantity over positive rows only.
- PAL zero-shot: template prompts, NO labels. (hpa_staining_zs.py)
- PAL linear-probe: frozen vision rep + sklearn LogisticRegression on **10K** labels, vision-only
  (no text encoder/anchors). Probe train = 12K intensity-balanced from HPA train (val-disjoint);
  location/quantity use the positive subset (~9K). (hpa_staining_linearprobe.py, --per-class 3000)
- iSight / fine-tuned PLIP / fine-tuned CONCH: from the iSight paper (arXiv 2602.04063), full
  fine-tune on the **10.5M** HPA train set (accuracy; likely micro).

| task | PAL zero-shot | UNI2-raw +probe(10K) | PAL-generic +probe(10K) | PAL-cap1 +probe(10K) | CONCH FT (10.5M) | PLIP FT (10.5M) | iSight (10.5M) |
|---|---|---|---|---|---|---|---|
| intensity | 0.568 | 0.566 | **0.617** | 0.597 | 0.700 | 0.730 | 0.766 |
| location  | 0.618 | 0.529 | **0.652** | 0.624 | 0.753 | 0.794 | 0.855 |
| quantity  | 0.531 | 0.493 | **0.572** | 0.562 | 0.701 | 0.732 | 0.757 |

(probe micro: generic 0.628/0.668/0.598 ; cap1 0.610/0.664/0.581 ; raw 0.565/0.619/0.532)

## Key findings
1. **PAL alignment > raw encoder** (linear probe, same 10K labels): PAL-generic beats UNI2-raw on
   every task — intensity +5.1, location +12.3, quantity +7.9 (macro) — despite being 512-d vs
   1536-d. Direct evidence that PAL's contrastive alignment makes staining info more linearly
   separable, i.e. the representation learning is genuinely useful.
2. **Extreme label/data efficiency**: 10K labels + frozen linear probe reaches 82-88% of
   full-fine-tuned CONCH (which used 10.5M labels): intensity 0.617/0.700=88%, location
   0.652/0.753=87%, quantity 0.572/0.701=82%. ~1000x fewer labels, encoder frozen.
3. **Labels help a bit over zero-shot**: generic zs→probe intensity 0.568→0.617, location
   0.618→0.652, quantity 0.531→0.572 (+3-5). Zero-shot alone is already strong.
4. **generic ≥ cap1** for classification/probe (all tasks, small margin) — consistent with generic
   naming labels 100% vs cap1 68-88%. (Retrieval was the opposite: generic-dedup > cap1.)

## Framing
iSight/PLIP/CONCH = supervised, full fine-tune, 10.5M labels, 336px patches + CLAM MIL + metadata
(iSight also cell-type). PAL = frozen UNI2-h (224 whole-core thumbnail) + light anchor alignment,
trained on 200K image-caption pairs with NO staining labels. So this is not a same-conditions
contest — PAL's story is: strong, label-free/label-efficient representation that (a) beats the raw
encoder and (b) reaches ~85% of 10.5M-label fine-tuning with 10K labels. Remaining levers to close
the gap (all discussed with-pathologist items): native-res crop (vs 224 thumbnail), metadata
conditioning, and full fine-tune / larger probe.
