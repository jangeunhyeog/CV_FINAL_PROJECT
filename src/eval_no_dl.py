"""딥러닝 없이 — 고전 기하 매칭만으로 검색했을 때를 frozen descriptor와 비교한다.

learned descriptor의 기여를 분리하기 위해, 동일한 부분집합에서:
  - 기하만(딥러닝 X): SIFT inlier 수로 쿼리-대-전체 DB를 직접 순위
  - 기하만 + 시간
  - 딥러닝(frozen) base / +시간 / +기하 re-rank / +기하+시간
를 같은 채점기(R@k @ 25m)로 비교한다.

기하만 검색은 쿼리를 DB 전체와 일일이 매칭하므로 O(M^2)이라 부분집합에서만 가능하다
(그 비용 자체가 '실시간 불가'의 근거). 연속 프레임은 공간적으로 너무 촘촘해 정답이
과다해지므로, STRIDE로 띄엄띄엄 뽑아 쿼리당 정답을 ~1개로 맞춘다.

사용:
    SUBSET_M=400 STRIDE=20 python src/eval_no_dl.py
결과: results/no_dl_subset.csv
"""
import csv
import os
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

import datasets as ds
import eval_recall as ev
import temporal_filter as tf
import rerank_geometric as rg

DATASET = "vpr-datasets-downloader/datasets/nordland"
DB_CACHE = "cache/nordland_eigenplaces_test_db.npy"
Q_CACHE = "cache/nordland_eigenplaces_test_q.npy"
M = int(os.environ.get("SUBSET_M", "400"))
STRIDE = int(os.environ.get("STRIDE", "20"))
K = 20
THRESH = 25.0
OUT_CSV = "results/no_dl_subset.csv"


def main():
    data = ds.load_geo_dataset(DATASET)
    idx = np.arange(0, M * STRIDE, STRIDE)
    idx = idx[idx < len(data["q_paths"])]
    qp = [data["q_paths"][i] for i in idx]
    dp = [data["db_paths"][i] for i in idx]
    pos = [set(ix) for ix in cKDTree(data["db_utm"][idx]).query_ball_point(data["q_utm"][idx], r=THRESH)]
    Sdeep = ev.cosine_similarity(np.load(Q_CACHE)[idx], np.load(DB_CACHE)[idx])
    m = len(idx)
    print(f"M={m} stride={STRIDE} avg_positives={np.mean([len(p) for p in pos]):.2f}")

    # SIFT 특징 캐시 후 inlier 행렬 G (m x m)
    feat = rg.LocalFeatureCache("sift", 2000, 640)
    qf = [feat.get(p) for p in qp]
    df = [feat.get(p) for p in dp]
    G = np.zeros((m, m), dtype=np.int32)
    t0 = time.time()
    for i in range(m):
        pq, dq = qf[i]
        for j in range(m):
            pd, dd = df[j]
            G[i, j] = rg.geometric_score(pq, dq, pd, dd, norm=cv2.NORM_L2, method="magsac")
    geo_ms = (time.time() - t0) * 1000.0 / m   # 쿼리당 (m개 DB와 매칭) 지연
    Gf = G.astype(np.float64)

    def r1(S_):
        return round(ev.recall_at_k(ev.topk_from_scores(S_, 1), pos, (1,))[1] * 100, 2)

    def deep_geo(base_S, alpha=0.7):
        topk = ev.topk_from_scores(base_S, K)
        out = topk.copy()
        for t in range(m):
            cand = topk[t]
            g, s = Gf[t, cand], base_S[t, cand]
            mm = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() - x.min() > 1e-12 else np.zeros_like(x)
            out[t] = cand[np.argsort(-(alpha * mm(s) + (1 - alpha) * mm(g)))]
        return round(ev.recall_at_k(out, pos, (1,))[1] * 100, 2)

    rows = [
        ("geometric only (no deep learning)", r1(Gf), round(geo_ms, 1)),
        ("geometric only + temporal", r1(tf.seqslam_rescore(Gf)), ""),
        ("deep descriptor (frozen)", r1(Sdeep), ""),
        ("deep + temporal", r1(tf.seqslam_rescore(Sdeep)), ""),
        ("deep + geometric re-rank", deep_geo(Sdeep), ""),
        ("deep + geometric + temporal", deep_geo(tf.seqslam_rescore(Sdeep)), ""),
    ]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "R@1", "geo_ms_per_query"])
        w.writerows(rows)
    for name, v, ms in rows:
        print(f"{name:<38} R@1={v}" + (f"   {ms} ms/q" if ms != "" else ""))
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
