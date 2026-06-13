"""M1+: 유사도 행렬 → Recall@K. 모든 후처리 모듈의 공통 평가 입력 형태를 통일한다.

핵심 인터페이스:
    S = cosine_similarity(Q, D)            # (T, N), 행=쿼리, 열=DB
    preds = topk_from_scores(S, k)         # (T, k) DB 인덱스
    r = recall_at_k(preds, positives, ks)  # {1: .., 5: .., 10: ..}

positives[i] = i번째 쿼리의 정답 DB 인덱스 집합(set). 정답 판정(25m / frame tol)은
데이터셋 로더가 만들어 넘긴다 — 평가 함수는 판정 기준을 모른다(프로토콜 고정).
"""
import numpy as np


def cosine_similarity(Q: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Q,D는 L2 정규화 가정. (T,d)·(N,d) → (T,N)."""
    return Q @ D.T


def topk_from_scores(S: np.ndarray, k: int) -> np.ndarray:
    """행별 상위 k DB 인덱스(점수 내림차순). S: (T,N) → (T,k)."""
    k = min(k, S.shape[1])
    idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(S.shape[0])[:, None]
    order = np.argsort(-S[rows, idx], axis=1)
    return idx[rows, order]


def recall_at_k(preds: np.ndarray, positives, ks=(1, 5, 10)) -> dict:
    """preds: (T, K) DB 인덱스. positives: list[set]. K >= max(ks) 이어야 함."""
    T = preds.shape[0]
    assert len(positives) == T, "preds와 positives 길이 불일치"
    out = {}
    for k in ks:
        hit = 0
        for i in range(T):
            if positives[i] and len(set(preds[i, :k]) & positives[i]) > 0:
                hit += 1
        out[k] = hit / T
    return out


def evaluate(S: np.ndarray, positives, ks=(1, 5, 10)) -> dict:
    """유사도 행렬 → Recall@K 한 번에."""
    preds = topk_from_scores(S, max(ks))
    return recall_at_k(preds, positives, ks)


if __name__ == "__main__":
    # 자체 sanity: 대각선이 정답인 toy 행렬에서 R@1 == 1.0 이어야 함
    T = N = 20
    S = np.eye(T) + 0.01 * np.random.randn(T, N)
    pos = [{i} for i in range(T)]
    print("toy recall:", evaluate(S, pos))
    assert evaluate(S, pos)[1] == 1.0
    print("OK")
