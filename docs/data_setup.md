# Dataset setup (`data/` layout)

How to populate `data/` so train / eval work. `data/` is gitignored (symlinked to
the real data in this working copy); a fresh checkout must place datasets under
`data/` in the layouts below. All paths are relative to `paths.data_path`
(default `./data`, set in `configs/default.yaml`).

Two categories:
- **Auto-download** (torchvision `download=True`) — created under `data/` on first
  run if absent. No manual step.
- **Manual** — you must download + unzip into the exact path. Code raises
  FileNotFoundError otherwise (or, for `download=False` torchvision sets, errors).

The dataset dispatch lives in `src/datasets/data_utils.py` (`get_datasets`) plus
`src/datasets/{coco,flickr30k}_dataset.py`. This doc mirrors that code — if they
diverge, the code wins.

---

## Paper benchmark datasets (train + zero-shot + retrieval)

### Training data (image–caption)

| Dataset | config name | Type | `data/` layout |
|---|---|---|---|
| **COCO** (train2014) | `coco` | manual | `data/COCO/annotations/captions_train2014.json`, `captions_val2014.json`; images in `data/COCO/train2014/`, `data/COCO/val2014/` |
| **Flickr30k** | `flickr30` | manual | `data/flickr30k/results.csv` (caption meta, `|`-delimited); split lists `train.txt` / `val.txt` / `test.txt`; images in `data/flickr30k/images/*.jpg` |

### Retrieval eval

| Dataset | config name | Type | `data/` layout |
|---|---|---|---|
| **COCO (Karpathy test)** | `coco_karpathy` | manual | Same as `coco` **plus** `data/COCO/karpathy_test_ids.json` (filters val2014 to the Karpathy test split; if the file is missing it silently falls back to full val2014) |
| **Flickr30k** | `flickr30` | manual | Same as training Flickr30k above (uses `test.txt`) |

### Zero-shot classification eval

| Dataset | config name | Type | `data/` layout / source |
|---|---|---|---|
| **CIFAR-100** | `cifar100` | auto | torchvision downloads to `data/cifar-100-python/` |
| **DTD** | `dtd` | auto | torchvision downloads to `data/dtd/` |
| **STL-10** | `stl10` | auto | torchvision downloads to `data/stl10_binary/` |
| **Caltech-101** | `caltech101` | **manual** | Download https://data.caltech.edu/records/mzrjq-6wc02 → unzip to `data/caltech-101/101_ObjectCategories/` (ImageFolder: one subdir per class) |
| **EuroSAT** | `eurosat` | **manual** | `wget https://madm.dfki.de/files/sentinel/EuroSAT.zip` → unzip to `data/eurosat/` (torchvision `EuroSAT(download=False)` reads `data/eurosat/2750/<class>/`) |

---

## Other datasets supported by `get_datasets` (not in the core paper benchmark)

`get_datasets` also dispatches these (available if you want them; same
`data/`-relative convention):

- **Auto-download** (torchvision `download=True`): `food101`, `cifar10`, `flowers`,
  `mnist`, `fer2013`, `country211`, `gtsrb`.
- **Manual — ImageFolder** (`data/<name>/{train,test}/<class>/*.jpg`): `birdsnap`
  (`birdsnap/{train,test}`), `aircraft` (`aircraft/{trainval,test}`), `pets`
  (`pets/{train,test}`), `resisc45`, `kitti`.
- **Manual — other**: `cars` (torchvision `StanfordCars(download=False)`),
  `sun397` (`data/sun397/test/` ImageFolder, or torchvision partition files),
  `coco2017` (hard-coded to `/home/data/2026_COCO` — edit the path in
  `data_utils.py` if used).

---

## Notes

- **Feature caches ≠ data.** Extracted features live under
  `paths.save_path/features/*.npy` (default `./results/features`). If a cache
  exists, encoders are skipped — but the **dataset object is still built from
  `data/`** every run (labels / image paths / captions come from the df, not the
  cache). So `data/` is required even when all feature caches are present.
- **Where the requirement comes from:** `CocoCaptionDataset.__init__` opens the
  annotation JSON; `Flickr30kDataset.__init__` reads `results.csv` + `{split}.txt`;
  torchvision sets read/download under `root=data_path`.
- Which datasets a run touches is set by `features.dataset` (train) and
  `evaluation.{zero_shot_datasets,retrieval_datasets}` (eval) in the config.
