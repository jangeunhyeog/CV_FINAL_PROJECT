# Results

All numbers are Recall (%) at the 25 m UTM threshold. Raw per-run logs are in
[`../results/results.csv`](../results/results.csv); curated headline numbers are in
[`../results/results_summary.csv`](../results/results_summary.csv).

## Summary

**Among training-free post-processing options, the most valuable signal is the temporal
(sequence) prior.** Temporal filtering raises R@1 by +12 to +33 points; geometric
re-ranking is marginal (+0 to +1.8) on a strong foundation descriptor.

## Module A — temporal consistency filtering (Nordland, winter ↔ summer)

| Model | base R@1 | R@5 | R@10 | + temporal (SeqSLAM, offline) | + temporal (forward, online) |
|---|---|---|---|---|---|
| EigenPlaces | 63.65 | 77.77 | 83.34 | **97.09** (+33.44) | 91.76 |
| SALAD | 86.46 | 93.85 | 95.90 | **98.98** (+12.52) | — |

- The gain is **model-agnostic**: it holds for both a ResNet-50 descriptor (EigenPlaces)
  and a DINOv2 descriptor (SALAD).
- The **offline** MAP path (Viterbi / SeqSLAM) beats the **causal online** forward filter
  (91.76 vs 97.09 for EigenPlaces), as expected — offline can use future frames.
- Nordland is the right stage for this module: extreme seasonal change collapses
  single-frame retrieval (low baseline → large headroom), while the temporal structure is
  preserved.

## Module B — geometric verification re-ranking (SVOX, EigenPlaces)

| Condition | base R@1 | best fixed-α re-rank | adaptive (ours) | note |
|---|---|---|---|---|
| sun | 85.25 | 85.48 (+0.23) | **85.60** | mild appearance change |
| overcast | 92.55 | 94.38 (+1.83) | 93.81 | best single-condition gain |
| night | 59.78 | 59.54 ⚠️ | **59.90** | SIFT matching fails |

- Geometry helps only a little, and only when SIFT matching is feasible (`overcast`).
- A **single** fixed α applied across conditions still **regresses** at night
  (`fixed α = 0.9`: 59.54 < 59.78 baseline).
- The **adaptive** rule (single config `α_min = 0.7`, `C = 40`, no per-condition tuning)
  improves every condition with **no regression**.

### Why fixed-α fails and adaptive does not

`Δ R@1 vs baseline`: the fixed weight dips below zero at night; the adaptive weight stays
≥ 0 everywhere (see `results/figures/delta_adaptive.png`). The absolute gains are small —
the honest claim is **robustness (no regression)**, not raw points.

## Latency (per query, post-processing only)

| Stage | ms / query |
|---|---|
| baseline retrieval (cosine top-K) | ~0.01–0.15 |
| + temporal (SeqSLAM offline) | ~1.95 |
| + temporal (forward online) | ~0.56 |
| + geometric re-rank (SIFT + MAGSAC, K=20) | ~130 |

Temporal filtering is essentially free; geometric re-ranking is two orders of magnitude
more expensive (SIFT detection + robust fitting per candidate), which — given its marginal
accuracy benefit — reinforces using it only as a confidence-gated tie-breaker.

## Dataset decisions (driven by measurements)

| Dataset | Decision | Reason |
|---|---|---|
| Nordland | ✅ Module A stage | sequence + extreme appearance, low baseline → large headroom |
| SVOX | ✅ Module B stage | per-condition queries (sun/overcast/night), clean UTM ground truth |
| St Lucia | ❌ excluded | baseline R@1 = 99.52 (ceiling effect, no headroom) |
| Baidu | ❌ excluded | inconsistent coordinate frame in the released labels |

## Honest positioning and limitations

- This is a **controlled empirical study**, not a new SOTA method. The closest prior work
  (SeqSLAM, sequence-based VPR, geometric re-ranking such as Patch-NetVLAD) is cited; the
  difference here is that the added modules are 100% classical (zero learned parameters),
  applied on top of a frozen 2023/2024 foundation descriptor, with temporal and geometric
  cues toggled independently.
- **Module A assumes** the queries are an ordered sequence and the database is a single
  traverse; it does not apply to unordered query sets.
- **Module B assumes** SIFT matching is feasible; it is essentially powerless under extreme
  appearance change (winter↔summer, night).
- The adaptive fusion's contribution is **no-regression robustness**, not a large absolute
  gain.

## Reproduction

```bash
# baseline + Module A (Nordland)
python src/run_experiment.py --dataset vpr-datasets-downloader/datasets/nordland --model eigenplaces
python src/run_experiment.py --dataset vpr-datasets-downloader/datasets/nordland --model eigenplaces \
    --module-a --ta-method seqslam
# SALAD variant
python src/run_experiment.py --dataset vpr-datasets-downloader/datasets/nordland --model salad \
    --module-a --ta-method seqslam

# Module B sweep (SVOX, per condition)
python src/rerank_sweep.py --dataset vpr-datasets-downloader/datasets/svox \
    --db-subdir gallery --q-subdir queries_overcast --model eigenplaces

# adaptive fusion analysis (uses cached inliers; prints the table above)
python src/gated_analysis.py

# regenerate all figures
python src/make_figures.py && python src/make_qualitative.py && python src/make_slide_figures.py
```
