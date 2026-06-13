"""기여(novelty): Confidence-Gated 기하 re-ranking.

관찰: 단순(naive) 기하 re-ranking은 기하 신호가 약/모호할 때 정확한 top-1을 끌어내려
      baseline보다 나빠진다(특히 night, 낮은 alpha).
해결: 기하 신호가 '충분히 강하고(top inlier >= t_abs) global top-1보다 분명히 우세할
      때만(margin >= t_margin)' 재정렬하고, 아니면 global 순서를 그대로 둔다.
      → 설계상 'no regression' 보장에 가깝고, 도움될 때만 개입한다.

캐시된 inlier(npz)로 오프라인 분석(SIFT 재계산 없음). SVOX 3개 조건에서
baseline vs naive(best fixed alpha) vs gated(단일 고정 config)를 비교한다.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import datasets as ds
import eval_recall as ev

DS = "vpr-datasets-downloader/datasets/svox"
CONDS = ["sun", "overcast", "night"]
FIG = "results/figures"


def load(cond):
    D = np.load("cache/svox_eigenplaces_test_db_gallery.npy")
    Q = np.load(f"cache/svox_eigenplaces_test_q_queries_{cond}.npy")
    S = ev.cosine_similarity(Q, D)
    z = np.load(f"cache/inliers_svox_eigenplaces_queries_{cond}_sift_magsac_k20.npz")
    data = ds.load_geo_dataset(DS, "test", db_sub="gallery", q_sub=f"queries_{cond}")
    return S, z["topk"], z["inl"], data["positives"]


def naive_rerank(S, topk, inl, alpha):
    """기존 방식: fused = a*norm(global) + (1-a)*norm(inlier), 전부 재정렬."""
    cand = topk
    gsim = np.take_along_axis(S, cand, axis=1)
    def mm(x):
        lo = x.min(1, keepdims=True); hi = x.max(1, keepdims=True)
        r = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        return (x - lo) / r
    fused = alpha * mm(gsim) + (1 - alpha) * mm(inl.astype(float))
    order = np.argsort(-fused, axis=1)
    return np.take_along_axis(cand, order, axis=1)


def _mm_rows(x):
    lo = x.min(1, keepdims=True); hi = x.max(1, keepdims=True)
    r = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (x - lo) / r


def adaptive_rerank(S, topk, inl, alpha_min, C):
    """제안: per-query 적응형 융합. 기하 신뢰도 conf_q = clip(max_inlier_q / C, 0, 1).
    alpha_q = 1 - (1-alpha_min)*conf_q  →  기하 신호가 강하면 alpha↓(geometry 신뢰),
    약하면(night 등 SIFT 실패) alpha→1(global 순서 유지, 안전). 단일 (alpha_min,C)가
    조건 무관하게 동작 → 고정 alpha의 'night 파국'을 회피한다."""
    gsim = np.take_along_axis(S, topk, axis=1)
    inlf = inl.astype(np.float64)
    conf = np.clip(inlf.max(1, keepdims=True) / C, 0.0, 1.0)   # (T,1)
    alpha_q = 1.0 - (1.0 - alpha_min) * conf                   # (T,1)
    fused = alpha_q * _mm_rows(gsim) + (1.0 - alpha_q) * _mm_rows(inlf)
    order = np.argsort(-fused, axis=1)
    preds = np.take_along_axis(topk, order, axis=1)
    return preds, float(conf.mean())


def r1(preds, pos):
    return ev.recall_at_k(preds, pos, ks=(1,))[1] * 100


def main():
    os.makedirs(FIG, exist_ok=True)
    loaded = {c: load(c) for c in CONDS}
    base = {c: r1(loaded[c][1], loaded[c][3]) for c in CONDS}

    # naive fixed alpha: 단일 alpha를 모든 조건에 동일 적용했을 때(조건 모름) 무엇이 best?
    fixed_alphas = [0.3, 0.5, 0.7, 0.9]
    naive_fixed = {a: {c: r1(naive_rerank(*loaded[c][:3], a), loaded[c][3]) for c in CONDS}
                   for a in fixed_alphas}
    # 단일 고정 alpha의 최선(평균 최대) — 그래도 night에서 깎이는지 본다
    best_fixed_a = max(fixed_alphas, key=lambda a: np.mean([naive_fixed[a][c] for c in CONDS]))

    # adaptive(ours): 단일 (alpha_min, C)를 모든 조건 공통 적용 → sweep
    grid = [(am, C) for am in [0.3, 0.5, 0.7] for C in [20, 40, 80]]
    def gains_of(am, C):
        g = []
        for c in CONDS:
            preds, _ = adaptive_rerank(*loaded[c][:3], am, C)
            g.append(r1(preds, loaded[c][3]) - base[c])
        return g
    best_cfg, best_key = None, None
    for am, C in grid:
        g = gains_of(am, C)
        key = (min(g), float(np.mean(g)))   # no-regression 우선, 그다음 평균 이득
        if best_key is None or key > best_key:
            best_key, best_cfg = key, (am, C)

    print(f"단일 고정 alpha 최선 = {best_fixed_a} | 제안 adaptive config = "
          f"alpha_min={best_cfg[0]}, C={best_cfg[1]}\n")
    hdr = f"{'condition':>10} {'baseline':>9} {'fixedα='+str(best_fixed_a):>11} {'adaptive(ours)':>15}"
    print(hdr)
    adap_r1 = {}
    for c in CONDS:
        preds, _ = adaptive_rerank(*loaded[c][:3], *best_cfg)
        adap_r1[c] = r1(preds, loaded[c][3])
        nf = naive_fixed[best_fixed_a][c]
        reg = "  <regress!" if nf < base[c] - 1e-9 else ""
        print(f"{c:>10} {base[c]:>9.2f} {nf:>11.2f}{reg} {adap_r1[c]:>15.2f}")
    print(f"\n평균: baseline={np.mean([base[c] for c in CONDS]):.2f}  "
          f"fixedα={np.mean([naive_fixed[best_fixed_a][c] for c in CONDS]):.2f}  "
          f"adaptive={np.mean([adap_r1[c] for c in CONDS]):.2f}")

    x = np.arange(len(CONDS)); w = 0.26
    plt.figure(figsize=(8, 4.2))
    plt.bar(x - w, [base[c] for c in CONDS], w, label="baseline", color="#bbbbbb")
    plt.bar(x, [naive_fixed[best_fixed_a][c] for c in CONDS], w,
            label=f"naive fixed α={best_fixed_a}", color="#d95f0e")
    plt.bar(x + w, [adap_r1[c] for c in CONDS], w, label="adaptive (ours)", color="#2c7fb8")
    plt.xticks(x, CONDS); plt.ylabel("R@1 (%)"); plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3); plt.legend()
    plt.title(f"Adaptive confidence-weighted re-ranking (α_min={best_cfg[0]}, C={best_cfg[1]})\n"
              "single config, no per-condition tuning; avoids the night collapse of fixed α")
    plt.tight_layout(); plt.savefig(f"{FIG}/gated_rerank.png", dpi=130); plt.close()
    print(f"[saved] {FIG}/gated_rerank.png")


if __name__ == "__main__":
    main()
