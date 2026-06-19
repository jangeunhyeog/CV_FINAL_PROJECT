"""정답 허용거리(UTM threshold) 민감도 — 2.5/5/10/25m 에서 R@1.

localization 정밀도를 본다: 임계값을 좁혀도(2.5m) 시퀀스 방식이 높게 유지되면
'느슨한 매칭'이 아니라 '정밀 매칭'이라는 근거가 된다. 예측(top-k)은 한 번만 계산하고
positives만 임계값별로 다시 만들어 채점(전체 Nordland, 캐시 재사용).

사용:  python src/eval_threshold_sensitivity.py
결과:  results/threshold_sensitivity.csv
"""
import csv
import os

import numpy as np
from scipy.spatial import cKDTree

import datasets as ds
import eval_recall as ev
import temporal_filter as tf

DATASET = "vpr-datasets-downloader/datasets/nordland"
THS = [2.5, 5, 10, 25]
OUT_CSV = "results/threshold_sensitivity.csv"


def rowmm(S):
    lo = S.min(1, keepdims=True); hi = S.max(1, keepdims=True)
    return ((S - lo) / np.maximum(hi - lo, 1e-9)).astype(np.float32)


def main():
    data = ds.load_geo_dataset(DATASET)
    tree = cKDTree(data["db_utm"])
    posT = {t: [set(ix) for ix in tree.query_ball_point(data["q_utm"], r=t)] for t in THS}

    Sd = ev.cosine_similarity(np.load("cache/nordland_eigenplaces_test_q.npy"),
                              np.load("cache/nordland_eigenplaces_test_db.npy"))
    methods = {
        "deep single-frame": ev.topk_from_scores(Sd, 10),
        "deep + online (temporal)": ev.topk_from_scores(tf.forward_filter(Sd, tau=0.15, v_min=1, v_max=1), 10),
    }
    if os.path.exists("cache/hog_nordland_q.npy"):
        Sh = (np.load("cache/hog_nordland_q.npy") @ np.load("cache/hog_nordland_db.npy").T).astype(np.float32)
        methods["HOG + online (temporal)"] = ev.topk_from_scores(tf.forward_filter(Sh, tau=0.15, v_min=1, v_max=1), 10)
        methods["deep + HOG fusion (single)"] = ev.topk_from_scores(0.5 * rowmm(Sd) + 0.5 * rowmm(Sh), 10)

    os.makedirs("results", exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"R1_{t}m" for t in THS])
        for name, preds in methods.items():
            row = [name] + [round(ev.recall_at_k(preds, posT[t], (1,))[1] * 100, 2) for t in THS]
            w.writerow(row); print(name, row[1:])
        w.writerow(["avg_positives_per_query"] + [round(float(np.mean([len(p) for p in posT[t]])), 1) for t in THS])
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
