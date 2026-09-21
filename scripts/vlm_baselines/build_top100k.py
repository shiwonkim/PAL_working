"""After conch_score_alltrain.py: take top-100K by CONCH score over the WHOLE
train split, write a drop-in training selection (image_path,caption,source),
shuffled seed 42 (so first-N subsampling stays unbiased)."""
import pandas as pd
QP = "/home/shiwon/STRUCTURE/data/quilt1m/selections"
K = 100000
SEED = 42

df = pd.read_csv(f"{QP}/conch_scores_alltrain.csv")
print("scored rows:", len(df))
top = df.nlargest(K, "conch_score").copy()
print(f"top-{K} score range: {top['conch_score'].min():.4f} .. {top['conch_score'].max():.4f}")
print("source(subset) 분포:", top["subset"].value_counts().to_dict())

out = pd.DataFrame({
    "image_path": top["image_path"].values,
    "caption":    top["caption"].values,
    "source":     top["subset"].values,
})
out = out.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle for first-N
OUT = f"{QP}/quilt_openpath-quilt_train_conchtop100k_seed42.csv"
out.to_csv(OUT, index=False)
print("wrote", OUT, "rows", len(out))
print("BUILD_DONE")
