"""고전 base 사다리 — SeqSLAM의 원본픽셀(SAD) base를 더 나은 고전 descriptor로 교체.

원본픽셀 SAD vs (CLAHE + HOG) vs 딥(frozen)을, 각각 단독 / +시간(SeqSLAM) / +DTW 정렬로
같은 부분집합·같은 채점(R@1@25m)에서 비교한다. 모두 학습 요소 0(딥은 frozen 추론).

DTW는 표준 동적시간왜곡(변속 허용 단조 정렬)을 직접 구현 — 비용=-유사도 최소 경로.
주의: 작은 1:1 정렬 부분집합에선 순서 제약이 너무 강해 +시간/+DTW가 포화(artifact)하므로,
단독(single-frame) 열이 'descriptor 품질'의 신뢰 가능한 비교다. 전체 규모 HOG vs 딥 수치는
docs/REALTIME.md 참고(HOG는 L2정규화라 matmul로 전체 계산 가능; SAD는 L1이라 전체 불가).

사용:
    SUBSET_M=400 STRIDE=20 python src/eval_classical_base.py
결과: results/classical_base.csv
"""
import csv
import os

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import hog

import datasets as ds
import eval_recall as ev
import temporal_filter as tf

DATASET = "vpr-datasets-downloader/datasets/nordland"
M = int(os.environ.get("SUBSET_M", "400"))
STRIDE = int(os.environ.get("STRIDE", "20"))
THRESH = 25.0
OUT_CSV = "results/classical_base.csv"
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def sad_vecs(paths, size=(32, 64)):
    """SeqSLAM식 원본픽셀 base: 다운샘플 + 패치 정규화."""
    out = []
    for p in paths:
        g = cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (size[1], size[0])).astype(np.float32).flatten()
        out.append((g - g.mean()) / (g.std() + 1e-6))
    return np.asarray(out, np.float32)


def hog_vecs(paths):
    """CLAHE 대비보정 후 HOG(구조/gradient) 특징, L2 정규화."""
    out = []
    for p in paths:
        g = _clahe.apply(cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (256, 256)))
        h = hog(g, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2), feature_vector=True)
        out.append((h / (np.linalg.norm(h) + 1e-9)).astype(np.float32))
    return np.asarray(out, np.float32)


def sim_sad(Q, D):
    S = np.empty((len(Q), len(D)), np.float32)
    for i in range(len(Q)):
        S[i] = -np.abs(D - Q[i]).mean(1)      # -평균절대차 (높을수록 닮음)
    return S


def dtw_predict(S):
    """비용=-S를 최소화하는 단조 정렬 경로 → 쿼리별 매칭 db (변속 허용)."""
    T, N = S.shape
    C = (-S).astype(np.float64)
    Dp = np.full((T, N), np.inf)
    Dp[0, 0] = C[0, 0]
    for j in range(1, N):
        Dp[0, j] = Dp[0, j - 1] + C[0, j]
    for i in range(1, T):
        Dp[i, 0] = Dp[i - 1, 0] + C[i, 0]
        for j in range(1, N):
            Dp[i, j] = C[i, j] + min(Dp[i - 1, j], Dp[i, j - 1], Dp[i - 1, j - 1])
    i, j, rows = T - 1, N - 1, {}
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
    data = ds.load_geo_dataset(DATASET)
    idx = np.arange(0, M * STRIDE, STRIDE)
    idx = idx[idx < len(data["q_paths"])]
    qp = [data["q_paths"][i] for i in idx]
    dp = [data["db_paths"][i] for i in idx]
    pos = [set(x) for x in cKDTree(data["db_utm"][idx]).query_ball_point(data["q_utm"][idx], r=THRESH)]
    m = len(idx)

    def r1S(S):
        return round(ev.recall_at_k(ev.topk_from_scores(S, 1), pos, (1,))[1] * 100, 2)

    def r1P(pred):
        return round(np.mean([1.0 if (pos[i] and pred[i] in pos[i]) else 0.0 for i in range(m)]) * 100, 2)

    Qs, Ds = sad_vecs(qp), sad_vecs(dp)
    Qh, Dh = hog_vecs(qp), hog_vecs(dp)
    bases = {
        "raw-pixel SAD": sim_sad(Qs, Ds),
        "CLAHE+HOG": Qh @ Dh.T,
        "deep (frozen)": ev.cosine_similarity(np.load("cache/nordland_eigenplaces_test_q.npy")[idx],
                                              np.load("cache/nordland_eigenplaces_test_db.npy")[idx]),
    }
    os.makedirs("results", exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["base", "single_R1", "seqslam_R1", "dtw_R1"])
        for name, S in bases.items():
            row = [name, r1S(S), r1S(tf.seqslam_rescore(S.astype(np.float32))), r1P(dtw_predict(S))]
            w.writerow(row)
            print(f"{name:<16} single={row[1]:>6}  +SeqSLAM={row[2]:>6}  +DTW={row[3]:>6}")
    print(f"\nsaved {OUT_CSV}  (subset M={m}; single-frame column is the reliable descriptor-quality comparison)")


if __name__ == "__main__":
    main()
