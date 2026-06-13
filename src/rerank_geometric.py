"""모듈 B — 기하 검증 re-ranking (M3).

쿼리별 top-K 후보에 대해 SIFT 대응점을 RANSAC/MAGSAC로 검증하고, inlier 수를
기하 일관성 점수로 사용해 재정렬한다. 학습 요소 0(고전 SIFT + robust estimator).

주의: 주행/실내 장면은 비평면이라 homography 가정이 엄밀하진 않지만, inlier 수를
'기하 일관성 점수'로만 쓰는 검증 용도로는 표준 관행(Patch-NetVLAD 등). fundamental
matrix 옵션도 제공한다.
"""
import cv2
import numpy as np
from tqdm import tqdm

import eval_recall as ev


def _detector(name: str, nfeatures: int):
    if name == "sift":
        return cv2.SIFT_create(nfeatures=nfeatures)
    if name == "orb":
        return cv2.ORB_create(nfeatures=nfeatures)
    raise ValueError(name)


def _load_gray(path: str, long_side: int):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    s = long_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


class LocalFeatureCache:
    """경로별 (keypoints, descriptors) 캐시 — 같은 DB 후보가 반복 등장하므로."""
    def __init__(self, detector="sift", nfeatures=2000, long_side=640):
        self.det = _detector(detector, nfeatures)
        self.long_side = long_side
        self._cache = {}

    def get(self, path):
        if path not in self._cache:
            img = _load_gray(path, self.long_side)
            if img is None:
                self._cache[path] = (None, None)
            else:
                kp, des = self.det.detectAndCompute(img, None)
                pts = np.float32([k.pt for k in kp]) if kp else None
                self._cache[path] = (pts, des)
        return self._cache[path]


def _ratio_match(des_q, des_d, norm, ratio=0.8):
    if des_q is None or des_d is None or len(des_q) < 2 or len(des_d) < 2:
        return None
    bf = cv2.BFMatcher(norm)
    knn = bf.knnMatch(des_q, des_d, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    return good


def geometric_score(pts_q, des_q, pts_d, des_d, norm, method="magsac", ratio=0.8):
    """SIFT 대응 → robust 모델 적합 → inlier 수 반환. 매칭 실패 시 0."""
    good = _ratio_match(des_q, des_d, norm, ratio)
    if not good or len(good) < 8:
        return 0
    src = pts_q[[m.queryIdx for m in good]]
    dst = pts_d[[m.trainIdx for m in good]]
    if method == "fundamental":
        _, mask = cv2.findFundamentalMat(src, dst, cv2.USAC_MAGSAC, 3.0, 0.999)
    else:  # homography + MAGSAC
        _, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0)
    return int(mask.sum()) if mask is not None else 0


def rerank(S, q_paths, db_paths, k=20, alpha=0.5,
           detector="sift", method="magsac", nfeatures=2000,
           long_side=640, ratio=0.8):
    """top-K 기하 re-ranking. 반환: (T,K) 재정렬된 DB 인덱스(나머지는 원순서 유지).

    fused = alpha * norm(global_sim) + (1-alpha) * norm(inliers)  (top-K 내 min-max).
    """
    norm = cv2.NORM_L2 if detector == "sift" else cv2.NORM_HAMMING
    feat = LocalFeatureCache(detector, nfeatures, long_side)
    T = S.shape[0]
    base_topk = ev.topk_from_scores(S, k)            # (T,k) 원 순서
    out = base_topk.copy()
    for t in tqdm(range(T), desc="rerank"):
        cands = base_topk[t]
        pts_q, des_q = feat.get(q_paths[t])
        inliers = np.array([
            geometric_score(pts_q, des_q, *feat.get(db_paths[j]),
                            norm=norm, method=method, ratio=ratio)
            for j in cands
        ], dtype=np.float64)
        gsim = S[t, cands]
        def mm(x):
            r = x.max() - x.min()
            return (x - x.min()) / r if r > 1e-12 else np.zeros_like(x)
        fused = alpha * mm(gsim) + (1 - alpha) * mm(inliers)
        out[t] = cands[np.argsort(-fused)]
    return out


if __name__ == "__main__":
    # SIFT/MAGSAC 가용성 sanity: 한 이미지를 약간 변형해 자기 자신과 매칭 → inlier 다수
    import sys
    p = sys.argv[1]
    feat = LocalFeatureCache("sift", 2000, 640)
    pts, des = feat.get(p)
    img = _load_gray(p, 640)
    M = np.float32([[1, 0.05, 5], [0.02, 1, 3]])
    warp = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cv2.imwrite("/tmp/_warp.png", warp)
    pts2, des2 = LocalFeatureCache("sift", 2000, 640).get("/tmp/_warp.png")
    norm = cv2.NORM_L2
    n = geometric_score(pts, des, pts2, des2, norm)
    print(f"self-warp inliers={n} (다수여야 정상)")
    assert n > 20, "SIFT/MAGSAC 기하 검증 비정상"
    print("OK")