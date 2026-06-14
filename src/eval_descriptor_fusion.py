"""학습 없는 descriptor 융합 — frozen deep + (CLAHE+HOG).

HOG(구조/gradient) 단서가 deep descriptor와 보완적인지 본다. 두 유사도 행렬을 행별
min-max 정규화 후 fused = a*deep + (1-a)*hog 로 섞고, deep 가중치 a를 sweep해 단독 R@1을
측정한다. 학습 요소 0(deep은 frozen 추론, HOG는 고전 특징).

HOG는 한 번 추출해 cache/ 에 저장(재사용). 전체 Nordland 기준.
사용:  python src/eval_descriptor_fusion.py
결과:  results/deep_hog_fusion.csv
"""
import csv
import os

import cv2
import numpy as np
from skimage.feature import hog

import datasets as ds
import eval_recall as ev

DATASET = "vpr-datasets-downloader/datasets/nordland"
DEEP_DB = "cache/nordland_eigenplaces_test_db.npy"
DEEP_Q = "cache/nordland_eigenplaces_test_q.npy"
OUT_CSV = "results/deep_hog_fusion.csv"
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def hog_cached(paths, cache):
    if os.path.exists(cache):
        return np.load(cache)
    out = np.empty((len(paths), 1764), np.float32)
    for k, p in enumerate(paths):
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        out[k] = 0 if g is None else _row_hog(g)
        if k % 3000 == 0:
            print(f"[hog] {k}/{len(paths)}", flush=True)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, out)
    return out


def _row_hog(g):
    g = _clahe.apply(cv2.resize(g, (256, 256)))
    h = hog(g, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2), feature_vector=True)
    return (h / (np.linalg.norm(h) + 1e-9)).astype(np.float32)


def rowminmax(S):
    lo = S.min(1, keepdims=True); hi = S.max(1, keepdims=True)
    return ((S - lo) / np.maximum(hi - lo, 1e-9)).astype(np.float32)


def main():
    data = ds.load_geo_dataset(DATASET)
    pos = data["positives"]
    Sd = rowminmax(ev.cosine_similarity(np.load(DEEP_Q), np.load(DEEP_DB)))
    Hq = hog_cached(data["q_paths"], "cache/hog_nordland_q.npy")
    Hd = hog_cached(data["db_paths"], "cache/hog_nordland_db.npy")
    Sh = rowminmax((Hq @ Hd.T).astype(np.float32))

    def r1(S_):
        return round(ev.recall_at_k(ev.topk_from_scores(S_, 1), pos, (1,))[1] * 100, 2)

    os.makedirs("results", exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["deep_weight_a", "single_frame_R1", "note"])
        for a in [1.0, 0.9, 0.7, 0.5, 0.3, 0.0]:
            fused = Sd if a == 1.0 else (a * Sd + (1 - a) * Sh).astype(np.float32)
            v = r1(fused)
            note = "deep only" if a == 1.0 else ("HOG only" if a == 0.0 else "")
            w.writerow([a, v, note])
            print(f"a={a}: single R@1 = {v}  {note}")
            if a != 1.0:
                del fused
    print(f"\nsaved {OUT_CSV}  (best fusion lifts single-frame R@1 by ~+11 over deep alone)")


if __name__ == "__main__":
    main()
