"""Exhaustive check: FeatureSpec.cache_suffix reproduces every on-disk cache name.

The suffix builders were scattered across ~6 inline sites; cache_suffix unifies
them. This parses every real cache filename under results/features, rebuilds the
suffix from a matching FeatureSpec, and asserts byte-identity — so the unified
builder is proven to cover all variants (cls/avg/token/mask/zs/-n/-r) before any
call site is switched over.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.utils.feature_spec import FeatureSpec  # noqa: E402

FEATURES = Path("results/features")


def modality_of(model: str) -> str:
    return "image" if "vit" in model.lower() else "text"


def parse(filename: str):
    """Filename -> (model, dataset, split, spec, subsample_n, zs, is_mask)."""
    name = filename[:-4] if filename.endswith(".npy") else filename
    # model and dataset are '-'-free (model is sanitised, datasets have none),
    # so the first two '-' separate model / dataset / suffix.
    model, dataset, suffix = name.split("-", 2)

    is_mask = suffix.endswith("_mask")
    if is_mask:
        suffix = suffix[: -len("_mask")]

    zs = suffix.endswith("-zs")
    if zs:
        suffix = suffix[: -len("-zs")]

    subsample_n = None
    m = re.search(r"-n(\d+)$", suffix)
    if m:
        subsample_n = int(m.group(1))
        suffix = suffix[: m.start()]

    img_size = None
    m = re.search(r"-r(\d+)$", suffix)
    if m:
        img_size = int(m.group(1))
        suffix = suffix[: m.start()]

    split, body = suffix.split("-", 1)

    if body.startswith("none_layer-"):
        pool = "none"
        layer_index = int(body[len("none_layer-"):])
    else:
        pool = body
        layer_index = None

    modality = modality_of(model)
    spec = FeatureSpec(
        modality=modality,
        token_level=(pool == "none"),
        pool=pool,
        layer_index=layer_index,
        img_size=img_size if modality == "image" else None,
        needs_mask=(pool == "none" and modality == "text"),
    )
    return model, dataset, split, spec, subsample_n, zs, is_mask


def main():
    files = sorted(p.name for p in FEATURES.glob("*.npy"))
    if not files:
        sys.exit("no cache files found under results/features")

    patterns = {}
    failures = []
    for f in files:
        try:
            model, dataset, split, spec, n, zs, is_mask = parse(f)
            got = spec.cache_suffix(split, subsample_n=n, zs=zs)
            # reconstruct the full base name and compare to the on-disk one
            from src.utils.feature_store import FeatureStore
            rebuilt = FeatureStore.cache_path(model, dataset, ".", got).name
            if is_mask:
                rebuilt = rebuilt.replace(".npy", "_mask.npy")
            ok = rebuilt == f
        except Exception as e:  # noqa: BLE001
            ok, rebuilt = False, f"PARSE-ERR {type(e).__name__}: {e}"
        # normalise to a pattern for the summary
        pat = re.sub(r"\d+", "N", f.split("-", 2)[2][:-4])
        patterns.setdefault(pat, [0, 0])
        patterns[pat][0] += 1
        patterns[pat][1] += int(ok)
        if not ok:
            failures.append((f, rebuilt))

    print(f"{'suffix pattern':40s} {'files':>6} {'ok':>6}")
    print("-" * 56)
    for pat in sorted(patterns):
        tot, okc = patterns[pat]
        flag = "OK" if tot == okc else "FAIL"
        print(f"{pat:40s} {tot:6d} {okc:6d}  {flag}")
    print("-" * 56)
    print(f"TOTAL files: {len(files)}, reproduced: {len(files) - len(failures)}")
    if failures:
        print("\nFAILURES (first 10):")
        for f, got in failures[:10]:
            print(f"  on-disk: {f}\n  rebuilt: {got}")
        sys.exit(1)
    print("ALL on-disk cache names reproduced by cache_suffix ✅")


if __name__ == "__main__":
    main()
