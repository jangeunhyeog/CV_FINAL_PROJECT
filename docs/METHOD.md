# Method

Notation: queries form a temporal sequence of `T` frames; the database traverse has `N`
images. The frozen descriptor produces L2-normalized vectors, so the cosine similarity
matrix is `S = Q @ D.T` with shape `(T, N)`, where `s_{tj}` is the similarity between
query `t` and DB image `j`. Single-frame retrieval returns `argmax_j s_{tj}`.

---

## Module A — Temporal consistency filtering

The queries are consecutive frames of a drive, so the correct DB indices evolve almost
monotonically with `t`. We model this with a hidden Markov model whose hidden state is the
DB index.

**Emission.** The observation likelihood is the temperature-scaled similarity:

```
p(z_t | x_t = j) ∝ exp(s_{tj} / τ)
```

**Transition.** The place index advances by a bounded speed; with a small teleport
probability `ε` to recover from path breaks:

```
p(x_t = j | x_{t-1} = i) = uniform over j ∈ [i + v_min, i + v_max],   else ε
```

Three variants are implemented in [`src/temporal_filter.py`](../src/temporal_filter.py):

1. **SeqSLAM-style rescoring** (`seqslam_rescore`). Column-wise local-contrast
   normalization of `S`, then for several constant-velocity hypotheses sum the scores
   along the corresponding diagonal window and keep the per-cell maximum. Fully vectorized
   (shifted-add), so it runs on the full 27k×27k Nordland matrix.

2. **Offline Viterbi** (`viterbi_path`). The log-space MAP path:

   ```
   δ_t(j) = max_i [ δ_{t-1}(i) + log a_{ij} ] + log b_t(j)
   ```

   with `log b_t(j) = s_{tj}/τ`. The bounded forward transition is computed by a windowed
   running maximum over `d ∈ [v_min, v_max]`; the teleport floor uses the global best
   state. Returns one MAP DB index per query.

3. **Online forward (Bayes) filter** (`forward_filter`). The causal version: propagate the
   log-belief through the same bounded transition (a sliding `logsumexp` implemented with a
   cumulative sum), add the emission, renormalize. Returns the full `(T, N)` belief for
   Recall@K. This is the variant that could run online on a robot.

**Hyperparameters** (defaults): `τ = 0.1`, `(v_min, v_max)` integer span for Viterbi/forward,
SeqSLAM window half-length `ds = 10`, velocity range `[0.8, 1.2]` over `n_vel = 5` samples,
`ε = 1e-3`.

A self-test in the module builds a toy matrix with a planted diagonal and a perceptual
aliasing distractor and asserts that Viterbi recovers the path.

---

## Module B — Geometric verification re-ranking

Implemented in [`src/rerank_geometric.py`](../src/rerank_geometric.py). For each query,
take its top-K candidates from `S` and rescore them by local-feature geometry:

1. Detect SIFT keypoints (image resized so the long side is 640 px).
2. Match query↔candidate descriptors with a brute-force matcher + **Lowe ratio test**
   (ratio = 0.8).
3. Fit a homography with **MAGSAC** (`cv2.findHomography(..., cv2.USAC_MAGSAC, 3.0)`);
   the **inlier count** is the geometric consistency score (a fundamental-matrix option is
   also provided). DB keypoints are cached because the same candidates recur.

**Fusion.** Within the top-K, min-max normalize both the global similarity and the inlier
counts and take a convex combination:

```
fused = α · norm(global_sim) + (1 − α) · norm(inliers)
```

`α = 1` keeps the original retrieval order; `α = 0` trusts geometry only. A low `α`
over-trusts geometry and *drops* R@1 when SIFT matching is unreliable, so naive fixed-α
fusion is only safe as a light tie-breaker (`α ≈ 0.7–0.9`).

---

## Adaptive confidence-weighted fusion (contribution)

Implemented in [`src/gated_analysis.py`](../src/gated_analysis.py). The failure mode of
fixed-α fusion is that there is **no single α** that helps every condition: the weight
that helps `overcast` regresses on `night`, where SIFT collapses. Instead of tuning α per
condition (which would require knowing the condition, and would not generalize), set the
weight **per query** from the geometric confidence:

```
conf_q  = clip( max_j inliers_{q,j} / C , 0, 1 )
α_q     = 1 − (1 − α_min) · conf_q
fused_q = α_q · norm(global_sim_q) + (1 − α_q) · norm(inliers_q)
```

- When the strongest candidate has many inliers (`conf_q → 1`), `α_q → α_min` and geometry
  is allowed to re-order.
- When geometry is weak (`conf_q → 0`, e.g. at night), `α_q → 1` and the method falls back
  to the retrieval order — so it cannot regress below baseline by much.

A **single** configuration (`α_min = 0.7`, `C = 40`) is applied to all conditions and
selected by a no-regression-first criterion (maximize the minimum per-condition gain, then
the mean gain). This yields a gain when geometry helps and avoids the night collapse of any
fixed α. The absolute gains are small (this is the honest framing: geometry is marginal on
a strong descriptor); the value of the rule is **robustness / no-regression**, not raw
points.

---

## Evaluation

Implemented in [`src/eval_recall.py`](../src/eval_recall.py) and
[`src/datasets.py`](../src/datasets.py). The evaluation function is deliberately unaware of
the correctness criterion — `datasets.py` builds the positive set for each query from UTM
coordinates (a KD-tree ball query at 25 m) so the protocol is fixed in exactly one place
and identical across all conditions. `recall@k` is the fraction of queries whose top-`k`
predictions intersect the positive set.
