"""모듈 B 오프라인 분석 — SIFT inlier를 1회 계산·캐시한 뒤 alpha/K를 sweep.

SIFT 재계산 없이 ablation을 빠르게 돌리기 위함. inlier 행렬 (T, Kmax)를 캐시한다.

사용:
    python src/rerank_sweep.py --dataset .../svox --db-subdir gallery \
        --q-subdir queries_sun --model eigenplaces --kmax 20
"""
import argparse
import os

import numpy as np

import datasets as ds
import eval_recall as ev
import rerank_geometric as rg


def compute_inliers(S, q_paths, db_paths, kmax, detector, method):
    """top-Kmax 후보의 inlier 수 행렬 (T, kmax)와 후보 인덱스 (T, kmax) 반환."""
    import cv2
    from tqdm import tqdm
    norm = cv2.NORM_L2 if detector == "sift" else cv2.NORM_HAMMING
    feat = rg.LocalFeatureCache(detector, 2000, 640)
    topk = ev.topk_from_scores(S, kmax)
    T = S.shape[0]
    inl = np.zeros((T, kmax), dtype=np.int32)
    for t in tqdm(range(T), desc="inliers"):
        pts_q, des_q = feat.get(q_paths[t])
        for c, j in enumerate(topk[t]):
            inl[t, c] = rg.geometric_score(pts_q, des_q, *feat.get(db_paths[j]),
                                           norm=norm, method=method)
    return topk, inl


def sweep(S, topk, inl, positives, alphas, ks):
    """캐시된 inlier로 alpha/K sweep. 반환 list of dict."""
    rows = []
    Tcand = topk.shape[1]
    for k in ks:
        if k > Tcand:
            continue
        cand = topk[:, :k]
        gsim = np.take_along_axis(S, cand, axis=1)          # (T,k)
        inlk = inl[:, :k].astype(np.float64)
        # 행별 min-max
        def mm(x):
            lo = x.min(1, keepdims=True); hi = x.max(1, keepdims=True)
            r = np.where(hi - lo > 1e-12, hi - lo, 1.0)
            return (x - lo) / r
        gN, iN = mm(gsim), mm(inlk)
        for a in alphas:
            fused = a * gN + (1 - a) * iN
            order = np.argsort(-fused, axis=1)
            preds = np.take_along_axis(cand, order, axis=1)
            r = ev.recall_at_k(preds, positives, ks=(1, 5, 10))
            rows.append({"k": k, "alpha": a, "R@1": round(r[1]*100, 2),
                         "R@5": round(r[5]*100, 2), "R@10": round(r[10]*100, 2)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--db-subdir", default="database")
    ap.add_argument("--q-subdir", default="queries")
    ap.add_argument("--model", default="eigenplaces")
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--detector", default="sift")
    ap.add_argument("--method", default="magsac")
    ap.add_argument("--thresh-m", type=float, default=ds.DEFAULT_POS_THRESH_M)
    args = ap.parse_args()

    data = ds.load_geo_dataset(args.dataset, "test", args.thresh_m,
                               db_sub=args.db_subdir, q_sub=args.q_subdir)
    tag = f"{os.path.basename(args.dataset.rstrip('/'))}_{args.model}_{args.q_subdir}"
    D = np.load(f"cache/{os.path.basename(args.dataset.rstrip('/'))}_{args.model}_test_db_{args.db_subdir}.npy")
    Q = np.load(f"cache/{os.path.basename(args.dataset.rstrip('/'))}_{args.model}_test_q_{args.q_subdir}.npy")
    S = ev.cosine_similarity(Q, D)

    inl_cache = f"cache/inliers_{tag}_{args.detector}_{args.method}_k{args.kmax}.npz"
    if os.path.exists(inl_cache):
        z = np.load(inl_cache); topk, inl = z["topk"], z["inl"]
        print(f"[cache] {inl_cache}")
    else:
        topk, inl = compute_inliers(S, data["q_paths"], data["db_paths"],
                                    args.kmax, args.detector, args.method)
        np.savez(inl_cache, topk=topk, inl=inl)
        print(f"[saved] {inl_cache}")

    base = ev.recall_at_k(topk, data["positives"], ks=(1, 5, 10))
    print(f"\nbaseline: R@1={base[1]*100:.2f} R@5={base[5]*100:.2f} R@10={base[10]*100:.2f}")
    rows = sweep(S, topk, inl, data["positives"],
                 alphas=[0.0, 0.3, 0.5, 0.7, 0.9, 1.0], ks=[5, 10, 20])
    print(f"\n{'k':>3} {'alpha':>6} {'R@1':>7} {'R@5':>7} {'R@10':>7}")
    for r in rows:
        mark = "  <-- best" if r["R@1"] == max(x["R@1"] for x in rows) else ""
        print(f"{r['k']:>3} {r['alpha']:>6} {r['R@1']:>7} {r['R@5']:>7} {r['R@10']:>7}{mark}")


if __name__ == "__main__":
    main()
