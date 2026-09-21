# Image–caption relationship: COCO vs PathCap vs HPA10M

**Why this matters.** PAL works well on COCO and PathCap. Whether it transfers to HPA10M
depends entirely on the *image–caption relationship* — what the caption describes, and whether
that content is visible at the resolution we feed the encoder. PAL's anchor-CAP mechanism
(pool patches through K learnable anchors → cosine profile → match to text) only has signal to
learn when the caption's discriminative content is spatially grounded in the image.

All numbers from the actual training selection CSVs (train split):
COCO `captions_train2017.json`; PathCap `pathcap_train_le128_seed42.csv`;
HPA `hpa10m_train_whole256_seed42.csv`.

## 1. What the caption describes (real examples)

| | Real caption example | What it points to |
|---|---|---|
| **COCO** | "A cat is taking a nap on top of a table" / "a group of people playing with a frisbee" | **Objects + relations** — each noun (cat, table, frisbee) grounds to a spatial region |
| **PathCap** | "Lung: Focal lymphocytic infiltration in the terminal bronchiole" / "Positive membrane expression of DcR1 (×200)" | **Tissue/organ + pathological finding** — grounds to tissue architecture / morphology (texture-level, still localizable) |
| **HPA generic** | "Immunohistochemical staining of human renal cancer shows moderate cytoplasmic/membranous staining in ~25–75% of tumor cells" | **Fixed template + 4 slots**: {tissue, intensity, location, quantity} |
| **HPA caption_1** | "Renal cancer stained for a protein (brown) demonstrates moderate cytoplasmic/membranous positivity in about 25–75% of tumor cells" | **Same 4 slots**, paraphrased (+DAB technique, +cancer subtype) |

## 2. Caption information content

| dataset | n | unique% | caption entropy (bits) | mean words |
|---|---|---|---|---|
| COCO | 591,753 | 96% | **19.1** | 10 |
| PathCap | 220,318 | 97% | **17.7** | 23 |
| HPA **generic** | 199,924 | **3%** | **10.6** | 17 |
| HPA caption_1 | 199,924 | 88% | 17.3 | 21 |

The generic caption's real content is only **{tissue 65 × intensity 4 × location 4 × quantity 4}
= 1,801 distinct attribute combinations** (→ ~5,619 unique caption strings). ~17 words but almost
all fixed boilerplate; the discriminative payload is 4 slots. Half the entropy of COCO/PathCap.

## 3. Grounding: is the caption's content visible at 224 whole-core thumbnail?

This is the crux — PAL's CAP can only align content the image actually resolves.

| caption component | visible at 224 whole-core thumbnail? |
|---|---|
| **COCO objects** | ✅ all — caption's resolution = image's resolution |
| **PathCap tissue/morphology finding** | ✅ mostly — tissue architecture reads at low magnification |
| HPA **tissue type** | ✅ discernible from morphology |
| HPA **intensity** (how brown) | 🔶 roughly, from core-level hue |
| HPA **location** (nuclear vs cytoplasmic/membranous) | ❌ needs **subcellular** resolution — destroyed by 224 downsample of a ~3000px core |
| HPA **quantity** (%positive cells) | ❌ needs **cell-level counting** — impossible from a thumbnail |

COCO/PathCap: *what the caption describes* matches *the resolution the image is shown at*.
HPA generic: half the discriminative payload (location, quantity) is **cell-level**, but we view
the whole core as a 224 thumbnail, so that information is not physically present in the pixels.
This scale mismatch is **independent of caption choice** (generic or cap1) — only native-res crop
(option A/B) fixes it.

## 4. Extra structural handicap: false negatives (the reason for the genbal run)

- generic: 3% unique, and **39% are the identical "no staining / negative" caption**.
- In contrastive learning, images sharing a caption become each other's negatives → false-negative
  flood. COCO (96%) / PathCap (97%) don't have this.
- `hpa10m_genbal` (cap 50 images/caption, negatives 39%→~10%) targets exactly this handicap.

## Conclusion — will PAL do on HPA what it did on COCO/PathCap?

Two separable issues:

1. **Caption diversity / false-negatives** — *fixable*. cap1 (88% unique) and genbal address it.
   Note cap1's added entropy (10.6→17.3 bits) is mostly **lexical paraphrase**, not new
   visual-discriminative content — the underlying 4 slots are the same. This is exactly why cap1
   helped retrieval uniqueness but did **not** beat generic on staining classification/probe.

2. **Scale mismatch** (location/quantity not visible at 224 thumbnail) — *not fixable by caption
   choice*. This is the root reason PAL's staining classification has a lower ceiling than iSight
   (336px + CLAM MIL). The only real lever is **native-res crop (option A/B)**.

So PAL's COCO/PathCap success rested on *"caption content = what's visible in the image"*. HPA10M
holds that for **tissue type and intensity**, but **breaks it for location and quantity** at
whole-core-thumbnail resolution. Our observed results fit this precisely: staining zero-shot is
well above random and PAL-pooled probe beats raw UNI2 (alignment genuinely works), yet the gap to
iSight is dominated by the resolution limit — not by the alignment method.
