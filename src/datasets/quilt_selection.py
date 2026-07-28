"""Build reproducible Quilt-1M training *selections* (filtered + subsampled).

The lookup CSV is noisy and mixes four sources (YouTube ``quilt``, Twitter
``openpath``, ``pubmed`` = PMC-OA, ``laion``). For pathology PAL training we
want CONCH-leakage-free, high-quality image-text pairs, so this module applies
**per-source metadata filters** (no VLM — keeps the low-resource claim clean)
and writes *selection* CSVs that :class:`QuiltCaptionDataset` loads directly.

Design: filtering/subsampling is decoupled from the dataset. A selection file is
a small ``(image_path, caption, source)`` table; nested seeded subsamples
(500 ⊂ 1k ⊂ 5k ⊂ …) make the data-scaling curve reproducible and inspectable.
Running the dataset without a selection loads the raw pool (needed for the CONCH
overlap diagnostic and ablations).

Filter rationale (verified against the CSV, see rebuttal notes):
- ``pubmed`` excluded — PMC-OA overlaps CONCH's training data (leakage).
- ``laion`` excluded — web noise, non-pathology.
- text column = ``caption`` (Quilt's clean per-image text; ``corrected_text`` is
  the long raw ASR window — do NOT use it).
- openpath: drop quiz/questions, strip trailing de-hashed tags.
- quilt: drop discourse/meta-framing captions ("the speaker is discussing…"),
  require non-empty ``roi_text`` (image-grounded).
"""

import argparse
import os
import re

import pandas as pd
from loguru import logger

# Pathology vocabulary — a caption must mention at least one to be kept (removes
# off-topic / non-histology captions without needing the (absent) not_histology
# flag from the newer lookup CSV).
PATH_VOCAB = re.compile(
    r"(?i)\b("
    r"carcinoma|adenoma|sarcoma|lymphoma|melanoma|tumou?r|biopsy|H&E|IHC|"
    r"immunohisto|stain|positive|negative|grade|mitos|nucle|cytoplasm|mucosa|"
    r"epitheli|stroma|gland|lesion|malignan|benign|metasta|dysplasia|"
    r"hyperplasia|inflammat|necrosis|fibrosis|cell|tissue|FNA|cyst|polyp|node|"
    r"duct|lobul|papill|squamous|serous|mucin|histolog|neoplas|granuloma|"
    r"infiltrat|carcinom|adenocarc|histiocyt|lymphocyt|chondro|osteo"
    r")\b"
)

# Discourse / meta-framing markers: the caption describes the narration or a
# non-visible clinical fact, not the image itself (quilt ASR-summary artifact).
META_FRAME = re.compile(
    r"(?i)("
    r"the speaker|is discussing|discussion (of|about)|description of|"
    r"the patient is|notes that|mentions that|talks about|"
    r"in this (video|slide|section|case)"
    r")"
)

# Trailing de-hashed pathology tags to strip from openpath (Twitter) captions.
_TAG_WORDS = {
    "pathology", "path", "dermpath", "gipath", "gynpath", "hemepath", "pulmpath",
    "cytopath", "surgpath", "neuropath", "renalpath", "breastpath", "bonepath",
    "softtissuepath", "endopath", "pedipath", "eyepath", "oralpath", "uropath",
    "forensicpath", "molecularpath", "pathtwitter", "pathboard", "pathboards",
    "twitter", "gucytopath", "cardiacpath",
}

# Per-source filter defaults (see module docstring for rationale).
SOURCE_DEFAULTS = {
    "openpath": dict(drop_questions=True, strip_tags=True,
                     drop_meta_frame=False, require_roi=False),
    "quilt": dict(drop_questions=True, strip_tags=False,
                  drop_meta_frame=True, require_roi=True),
}


def _strip_trailing_tags(cap: str) -> str:
    toks = str(cap).split()
    while toks and toks[-1].strip("#.,:;!").lower() in _TAG_WORDS:
        toks.pop()
    return " ".join(toks)


def filter_source(
    df: pd.DataFrame,
    source: str,
    word_range=(8, 40),
    require_path_vocab=True,
    drop_questions=True,
    drop_meta_frame=False,
    require_roi=False,
    strip_tags=False,
    dedup=True,
):
    """Filter one subset by metadata heuristics. Returns (df, funnel_steps)."""
    d = df[df["subset"] == source].copy()
    d["text"] = d["caption"].astype(str)
    if strip_tags:
        d["text"] = d["text"].map(_strip_trailing_tags)
    steps = [("start", len(d))]

    if drop_questions:
        d = d[~d["text"].str.contains(r"\?")]
        steps.append(("drop_questions", len(d)))
    if drop_meta_frame:
        d = d[~d["text"].str.contains(META_FRAME)]
        steps.append(("drop_meta_frame", len(d)))
    if require_roi and "roi_text" in d.columns:
        roi = d["roi_text"].astype(str)
        d = d[(roi.str.len() > 2) & (roi != "[]")]
        steps.append(("require_roi", len(d)))
    wl = d["text"].str.split().str.len()
    d = d[(wl >= word_range[0]) & (wl <= word_range[1])]
    steps.append((f"len{word_range}", len(d)))
    if require_path_vocab:
        d = d[d["text"].str.contains(PATH_VOCAB)]
        steps.append(("path_vocab", len(d)))
    if dedup:
        d = d.drop_duplicates(subset="text")
        steps.append(("dedup_caption", len(d)))
        d = d.drop_duplicates(subset="image_path")
        steps.append(("dedup_image", len(d)))
    return d, steps


def build_selection(
    csv_file: str,
    out_dir: str,
    sources=("openpath", "quilt"),
    sizes=(500, 1000, 5000, 10000, 20000, 40000),
    seed=42,
    split="train",
    word_range=(8, 40),
    require_path_vocab=True,
):
    """Filter per source, merge, seeded-shuffle, write nested selection CSVs.

    Nested = ``head(N)`` of one shuffled pool, so 500 ⊂ 1000 ⊂ 5000 ⊂ … and the
    source mix stays consistent across scales. Also writes the full filtered pool
    (``…_full_…``). Returns (list_of_paths, pool_size).
    """
    df = pd.read_csv(
        csv_file, usecols=["caption", "image_path", "subset", "split", "roi_text"]
    )
    df = df[df["split"] == split]

    parts = []
    for src in sources:
        opts = dict(word_range=word_range, require_path_vocab=require_path_vocab)
        opts.update(SOURCE_DEFAULTS.get(src, {}))
        d, steps = filter_source(df, src, **opts)
        logger.info(f"[{src}] " + "  ".join(f"{k}={v}" for k, v in steps))
        d["source"] = src
        parts.append(d[["image_path", "text", "source"]].rename(columns={"text": "caption"}))

    pool = pd.concat(parts, ignore_index=True)
    pool = pool.drop_duplicates(subset="image_path").reset_index(drop=True)
    pool = pool.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(f"merged filtered pool: {len(pool)} pairs "
                + ", ".join(f"{s}={int((pool['source'] == s).sum())}" for s in sources))

    os.makedirs(out_dir, exist_ok=True)
    tag = "-".join(sources)
    written = []
    for N in list(sizes) + ["full"]:
        sub = pool if N == "full" else (pool.head(N) if N <= len(pool) else None)
        if sub is None:
            logger.warning(f"size {N} > pool {len(pool)} — skipped")
            continue
        path = os.path.join(out_dir, f"quilt_{tag}_{N}_seed{seed}.csv")
        sub.to_csv(path, index=False)
        mix = ", ".join(f"{s}={int((sub['source'] == s).sum())}" for s in sources)
        logger.info(f"wrote {path}  (n={len(sub)}: {mix})")
        written.append(path)
    return written, len(pool)


def main():
    ap = argparse.ArgumentParser(description="Build Quilt-1M training selections.")
    ap.add_argument("--csv", default="data/quilt1m/quilt_1M_lookup.csv")
    ap.add_argument("--out_dir", default="data/quilt1m/selections")
    ap.add_argument("--sources", nargs="+", default=["openpath", "quilt"],
                    help="subset sources to include (pubmed/laion excluded by default)")
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[500, 1000, 5000, 10000, 20000, 40000])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--word_min", type=int, default=8)
    ap.add_argument("--word_max", type=int, default=40)
    ap.add_argument("--no_path_vocab", action="store_true",
                    help="disable the pathology-vocabulary requirement")
    args = ap.parse_args()
    build_selection(
        csv_file=args.csv, out_dir=args.out_dir, sources=tuple(args.sources),
        sizes=tuple(args.sizes), seed=args.seed,
        word_range=(args.word_min, args.word_max),
        require_path_vocab=not args.no_path_vocab,
    )


if __name__ == "__main__":
    main()
