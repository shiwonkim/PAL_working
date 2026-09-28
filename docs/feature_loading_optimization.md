# Feature loading: measured bottleneck & optimization plan

> **Status (2026-08-31):** design only — nothing implemented. Deliberately deferred until
> after the other server's `exp/natural` branch is integrated, so the change lands once on a
> merged trainer instead of twice. Written from measurements taken on the live
> UNI2-h × PathCap run (`exp/pathology`, tmux `pathcap_uni2`).

Related: `docs/laion_reimplementation_TODO.md` §5 has the same recipe for the *LAION concat*
case. This doc covers the **single-corpus** case we actually hit, whose diagnosis differs.

---

## 0. Background: where the data lives and why it matters

```
[ disk / SSD ]     ~1-2 GB/s     the 179 GB cache file lives here
      |  read
[ RAM  377 GB ]    ~10-50 GB/s   where the file gets held
      |  PCIe copy (~10-25 GB/s)
[ GPU VRAM 45 GB ] ~1000 GB/s    the only memory the GPU can compute on
      |
[ GPU compute ]
```

Two constraints follow. The GPU computes **only** on what is in its 45 GB of VRAM, so a 179 GB
cache can never sit there — each step must copy the slice it needs from RAM across PCIe. And the
throughput of the whole pipeline is set by its slowest active stage, so "which stage is actually
working" is the first thing to measure, not guess.

**Two kinds of RAM, and the distinction our bug turns on:**

| | file pages (`RssFile`) | anonymous memory (`RssAnon`) |
|---|---|---|
| what it is | a copy of bytes that exist in a file | data the process created |
| backed by | the file on disk | nothing — RAM is the only copy |
| under memory pressure | can be **evicted** (re-read later) | cannot be dropped → OOM / swap |

`mmap` maps a file into the address space so only the pages actually touched are resident, and
they stay evictable file pages. `FeatureStore` opens the cache that way, which is correct — but
what the process ends up holding is what `RssFile` vs `RssAnon` reports, and that is the
measurement in §1 below. Once a tensor is *materialised* (any boolean mask or fancy index makes
a new tensor), the result is anonymous: the tie to the file is gone and the OS can no longer
reclaim it.

---

## 1. What we measured (not assumed)

Live run: UNI2-h + PubMedBERT on PathCap-train, 220,318 rows, token cache
`(220318, 265, 1536)` fp16 = **179.4 GB** image + 43.4 GB text.

| probe | value | reading |
|---|---|---|
| `/proc/<pid>/io` read_bytes over 10 s | **0 MB/s** (62 GB cumulative = initial load only) | **not disk-bound** |
| `RssAnon` | **222.7 GB** | data is **anonymous** memory |
| `RssFile` | **0.2 GB** | the `mmap=True` mapping is **not** what's resident |
| GPU util, 40 × 1 s samples | mean **16 %**, median **5 %**, **48 % of samples at 0 %**, max 82 % | **GPU starved** |
| epoch time | 350 s (108 batches @ 2048) → **3.24 s/batch** | |

**Diagnosis: CPU-side gather starves the GPU.** Nothing is read from disk; everything already
sits in RAM. Each step gathers 2048 random rows out of 220 K, i.e. `2048 × 265 × 1536 × 2 B =
**1.66 GB copied per batch**, 179 GB per epoch — synchronously, on the main process, with no
overlap. The GPU waits through it (0 % half the time) and then bursts (82 %).

### Why the cache is anonymous rather than file-backed
`FeatureStore` loads with `torch.load(..., mmap=True)` (`feature_store.py:290`), which is
correct. But `AlignmentTrainer.prepare_features` then applies the dedup / subsample masks with
boolean or fancy indexing (`_apply_if_full`: `tensor[mask_bool]`, and the
`torch.randperm(...)[:n]` subsample path). **Any of those produces a new, materialized tensor**,
which is anonymous memory — the file mapping is dropped at that moment. So the memory-mapping
benefit is lost before training starts, even though the load call asked for it.

Note the per-batch shuffle is *already* index-based, not data-copying
(`alignment_trainer.py:1557-1563` — a deliberate earlier fix). That part is fine; the cost is the
gather itself plus the absence of prefetch.

---

## 2. Consequence for the obvious "just stream batches from disk" idea

Naively re-reading each batch from disk would be **slower, not faster**: disk traffic is
currently zero, and this would reintroduce it as *random* reads. Speed comes from overlap and
locality, not from streaming per se. Streaming's payoff is **RAM**; the speed payoff is
prefetch + sequentialization. They are two separate fixes — a speed one (make CPU gather and
GPU compute overlap) and a memory one (keep the mapping file-backed, §0) — and the plan below
does both.

---

## 3. Plan

Ordered by payoff / effort. (1) alone should recover most of the idle GPU.

1. **Async prefetch.** Wrap feature access in an `IterableDataset` + `DataLoader`
   (`num_workers>0`, `pin_memory=True`, `prefetch_factor`) so batch *k+1* is gathered while
   batch *k* is on the GPU. Directly targets the 48 %-idle measurement; expected roughly
   **1.5–2× epoch-time reduction** with no change to batch composition.
2. **Keep the mapping file-backed.** Don't materialize dedup/subsample. Carry the selected row
   indices alongside the mmap tensor and compose them into the per-batch gather
   (`rows = base_idx[perm[batch_slice]]`). Turns 222 GB of anon into page cache the OS can
   evict — this is what makes >200 GB corpora possible at all.
3. **Chunk-sequential reads + buffer shuffle.** Replace fully random per-batch rows with: read
   large contiguous chunks in sequence, shuffle inside a buffer of a few chunks. Preserves
   enough SGD randomness while making access patterns prefetcher- and page-cache-friendly.
   Matters most once (2) makes reads actually touch the file.
4. **Per-batch padding only.** Never `F.pad` a whole (N, T, D) tensor to a common `T`; pad
   within the batch. (Carried over from the LAION doc — same failure mode.)

**Honest limit:** if the working set exceeds physical RAM *and* access stays fully random, no
arrangement reaches RAM speed. (3) is what buys back most of that gap.

## 4. Cache compatibility — no re-extraction

The on-disk format stays valid through all of the above. Caches are a **single contiguous fp16
tensor** `(N, T, D)` pre-allocated and filled by sequential offset
(`feature_store.py` streaming-allocation path), written with `torch.save`, and extraction runs
with `shuffle=False` (`train.py:84,93`) — so **cache row _i_ == selection-CSV row _i_**.

Only the *access pattern* changes, so the 179 GB PathCap and 151 GB HPA10M caches are reused
as-is. Row-order stability is also what lets subset experiments (e.g. a PathCap IHC-only or
size-matched-random split) run by indexing alone, with no re-extraction.

Re-extraction would only be needed if the on-disk **layout** changed (sharding into multiple
chunk files, or per-sample files). Not required by this plan.

## 5. Checklist

- [ ] Land after `exp/natural` integration, on the merged trainer.
- [ ] (1) prefetch via IterableDataset/DataLoader; measure GPU util + epoch time before/after.
- [ ] (2) keep mmap file-backed through dedup/subsample; check `RssFile` ≫ `RssAnon` after.
- [ ] (3) chunk-sequential + buffer shuffle; confirm loss curve is unchanged vs full shuffle.
- [ ] (4) per-batch padding.
- [ ] Re-verify: identical checkpoint quality on a short run, then epoch-time comparison on
      the PathCap 220 K cache (baseline: **350 s/epoch**, GPU mean **16 %**).
