"""DTW 강건성 — 변속(variable-speed) 시나리오에서 정렬기 비교 (왜 DTW).

쿼리를 불규칙 간격(스텝 1~3 랜덤)으로 샘플해 '등속 가정'을 깨뜨린 뒤,
fixed-v(+1) forward / SeqSLAM(고정 속도≈1) / wide-v forward / DTW 의 R@1을 비교한다.
고정 속도를 가정하는 방식은 변속에서 붕괴(심지어 single-frame 이하)하고, 속도 가정이
없는 DTW(시간왜곡 정렬)는 강건하다. deep base 고정(정렬기 효과만). DTW는 O(K*N)이라
부분구간에서 수행.

사용:  python src/eval_dtw_robustness.py
결과:  results/dtw_robustness.csv
"""
import csv
import os

import numpy as np
from scipy.spatial import cKDTree

import datasets as ds
import eval_recall as ev
import temporal_filter as tf

DATASET = "vpr-datasets-downloader/datasets/nordland"
N = 5000
THS = [2.5, 25.0]
OUT_CSV = "results/dtw_robustness.csv"


def dtw_predict(S):
    """비용=-S를 최소화하는 단조 시간왜곡 경로 → 쿼리별 매칭 db (속도 가정 없음)."""
    T, M = S.shape
    C = (-S).astype(np.float64)
    Dp = np.full((T, M), np.inf); Dp[0, 0] = C[0, 0]
    for j in range(1, M):
        Dp[0, j] = Dp[0, j - 1] + C[0, j]
    for i in range(1, T):
        Dp[i, 0] = Dp[i - 1, 0] + C[i, 0]
        for j in range(1, M):
            Dp[i, j] = C[i, j] + min(Dp[i - 1, j], Dp[i, j - 1], Dp[i - 1, j - 1])
    i, j, rows = T - 1, M - 1, {}
    while True:
        rows.setdefault(i, []).append(j)
        if i == 0 and j == 0:
            break
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            s = int(np.argmin([Dp[i - 1, j - 1], Dp[i - 1, j], Dp[i, j - 1]]))
            i, j = (i - 1, j - 1) if s == 0 else (i - 1, j) if s == 1 else (i, j - 1)
    pred = np.zeros(T, int)
    for r, js in rows.items():
        pred[r] = js[int(np.argmax(S[r, js]))]
    return pred


def main():
    rng = np.random.default_rng(0)
    data = ds.load_geo_dataset(DATASET)
    du = data["db_utm"][:N]
    Dd = np.load("cache/nordland_eigenplaces_test_db.npy")[:N]
    Dq = np.load("cache/nordland_eigenplaces_test_q.npy")

    path = [0]
    while path[-1] < N - 3:
        path.append(path[-1] + int(rng.integers(1, 4)))   # 변속: 1~3칸 전진
    path = np.array(path)
    S = ev.cosine_similarity(Dq[path], Dd).astype(np.float32)
    K = len(path)
    tree = cKDTree(du)
    posT = {t: [set(tree.query_ball_point(du[p], r=t)) for p in path] for t in THS}
    print(f"variable-speed queries K={K} over N={N} (avg step {np.mean(np.diff(path)):.2f})")

    def r1S(S_):
        pr = ev.topk_from_scores(S_, 1)
        return [round(ev.recall_at_k(pr, posT[t], (1,))[1] * 100, 2) for t in THS]

    def r1P(pred):
        return [round(np.mean([1.0 if (posT[t][i] and pred[i] in posT[t][i]) else 0.0
                               for i in range(K)]) * 100, 2) for t in THS]

    rows = [
        ("single-frame (no temporal)", r1S(S)),
        ("forward fixed v=1 (constant-velocity)", r1S(tf.forward_filter(S, tau=0.15, v_min=1, v_max=1))),
        ("SeqSLAM (fixed velocity prior ~1)", r1S(tf.seqslam_rescore(S))),
        ("DTW (time-warp; no speed assumption)", r1P(dtw_predict(S))),
        ("forward wide v=1..3", r1S(tf.forward_filter(S, tau=0.1, v_min=1, v_max=3))),
    ]
    os.makedirs("results", exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aligner"] + [f"R1_{t}m" for t in THS])
        for name, vals in rows:
            w.writerow([name] + vals); print(f"{name:<42} {vals}")
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
