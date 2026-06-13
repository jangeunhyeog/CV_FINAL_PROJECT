"""모듈 A — 시간 일관성 필터링 (M2의 핵심). 벡터화 버전(대규모 행렬 대응).

쿼리가 '주행 시퀀스(연속 프레임, 순서 보존)'라는 사실을 활용해, 단일 프레임
유사도 행렬 S(T×N)를 시간 구조로 재점수화한다. 세 변형:

  1. seqslam_rescore : 국소 대비 정규화 + 등속 가정 대각선 윈도 합산 → (T,N)
  2. viterbi_path    : HMM MAP 경로(오프라인). 쿼리별 단일 예측 → R@1·분석용
  3. forward_filter  : 온라인 Bayes 필터(forward-only). belief (T,N) → R@K

전이 모델: DB 인덱스가 [v_min, v_max]만큼 전진. 경로 이탈 대비 작은 teleport 확률 eps.
모든 핵심 연산은 numpy/scipy로 벡터화되어 27k×27k 규모에서도 동작한다.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d


# ----------------------------------------------------------------------------
# 1. SeqSLAM 방식 (점수 기반, 학습 없음) — 벡터화
# ----------------------------------------------------------------------------
def local_contrast_normalize(S: np.ndarray, win: int = 10) -> np.ndarray:
    """열 방향(DB축) 국소 평균/표준편차로 정규화 — SeqSLAM 전처리. 벡터화."""
    S = S.astype(np.float32, copy=False)
    size = 2 * win + 1
    mu = uniform_filter1d(S, size=size, axis=1, mode="nearest")
    musq = uniform_filter1d(S * S, size=size, axis=1, mode="nearest")
    sd = np.sqrt(np.maximum(musq - mu * mu, 1e-12))
    return (S - mu) / sd


def _diag_window_mean(X: np.ndarray, qo: np.ndarray, do: np.ndarray) -> np.ndarray:
    """score[t,n] = mean_k X[t+qo[k], n+do[k]] (유효 범위만). 벡터화된 shifted-add."""
    T, N = X.shape
    acc = np.zeros((T, N), dtype=np.float32)
    cnt = np.zeros((T, N), dtype=np.float32)
    for a, b in zip(qo, do):
        t0, t1 = max(0, -a), min(T, T - a)
        n0, n1 = max(0, -b), min(N, N - b)
        if t0 >= t1 or n0 >= n1:
            continue
        acc[t0:t1, n0:n1] += X[t0 + a:t1 + a, n0 + b:n1 + b]
        cnt[t0:t1, n0:n1] += 1.0
    cnt[cnt == 0] = 1.0
    return acc / cnt


def seqslam_rescore(S: np.ndarray, ds: int = 10,
                    v_min: float = 0.8, v_max: float = 1.2, n_vel: int = 5,
                    normalize: bool = True) -> np.ndarray:
    """등속 가정 대각선 윈도 합산. 여러 속도 후보 중 최대 점수. 반환 (T,N)."""
    X = local_contrast_normalize(S) if normalize else S.astype(np.float32, copy=False)
    offs = np.arange(-ds, ds + 1)
    best = None
    for v in np.linspace(v_min, v_max, n_vel):
        do = np.round(v * offs).astype(int)
        score = _diag_window_mean(X, offs, do)
        best = score if best is None else np.maximum(best, score)
    return best


# ----------------------------------------------------------------------------
# 공통: 전진 전이 윈도 (Viterbi/forward 둘 다 사용) — span만큼만 shift (벡터화)
# ----------------------------------------------------------------------------
def _windowed_prev_max(prev: np.ndarray, v_min: int, v_max: int):
    """best[j]=max_{d in [v_min,v_max]} prev[j-d], arg[j]=해당 i=j-d. 벡터화."""
    N = prev.shape[0]
    best = np.full(N, -np.inf, dtype=np.float64)
    arg = np.zeros(N, dtype=np.int32)
    idx = np.arange(N)
    for d in range(v_min, v_max + 1):
        cand = np.full(N, -np.inf)
        if d < N:
            cand[d:] = prev[:N - d]          # prev[j-d]
        better = cand > best
        best = np.where(better, cand, best)
        arg = np.where(better, idx - d, arg)
    return best, arg


# ----------------------------------------------------------------------------
# 2. Viterbi (오프라인 MAP 경로)
# ----------------------------------------------------------------------------
def viterbi_path(S: np.ndarray, tau: float = 0.1,
                 v_min: int = 1, v_max: int = 3, eps: float = 1e-3) -> np.ndarray:
    """로그 공간 Viterbi. 반환 (T,) 쿼리별 MAP DB 인덱스."""
    logB = (S / tau).astype(np.float64)
    T, N = logB.shape
    log_eps = np.log(eps)
    delta = logB[0].copy()
    back = np.zeros((T, N), dtype=np.int32)
    for t in range(1, T):
        best, arg = _windowed_prev_max(delta, v_min, v_max)
        gmax_i = int(np.argmax(delta))
        tele = delta[gmax_i] + log_eps
        use = tele > best
        best = np.where(use, tele, best)
        arg = np.where(use, gmax_i, arg)
        delta = best + logB[t]
        back[t] = arg
    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = int(np.argmax(delta))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


# ----------------------------------------------------------------------------
# 3. 온라인 Bayes 필터 (forward-only, belief 반환 → R@K) — 벡터화
# ----------------------------------------------------------------------------
def forward_filter(S: np.ndarray, tau: float = 0.1,
                   v_min: int = 1, v_max: int = 3, eps: float = 1e-3) -> np.ndarray:
    """forward filtering. 반환 (T,N) 로그 belief(행별, R@K용). 슬라이딩 logsumexp."""
    logB = (S / tau).astype(np.float64)
    T, N = logB.shape
    span = v_max - v_min + 1
    log_uniform = -np.log(span)
    log_eps = np.log(eps)
    bel = logB[0].copy()
    out = np.empty_like(logB)
    out[0] = bel
    for t in range(1, T):
        M = bel.max()
        e = np.exp(bel - M)
        csum = np.concatenate([[0.0], np.cumsum(e)])   # csum[k]=sum e[:k]
        # j는 i in [j-v_max, j-v_min] 합산: sum e[max(0,j-v_max) .. j-v_min]
        j = np.arange(N)
        hi = j - v_min + 1          # exclusive 상한
        lo = np.maximum(0, j - v_max)
        hi = np.clip(hi, 0, N)
        wsum = csum[hi] - csum[lo]
        with np.errstate(divide="ignore"):
            pred = M + np.log(wsum * np.exp(log_uniform))
        pred = np.logaddexp(pred, log_eps + M)          # teleport 바닥
        bel = pred + logB[t]
        bel -= bel.max()
        out[t] = bel
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, N = 50, 100
    S = 0.2 * rng.standard_normal((T, N)).astype(np.float32)
    gt = np.arange(T) + 25
    for t in range(T):
        S[t, gt[t]] += 1.0
    base_r1 = (S.argmax(1) == gt).mean()
    vit_r1 = (viterbi_path(S, 0.1, 1, 2, 1e-3) == gt).mean()
    ff_r1 = (forward_filter(S, 0.1, 1, 2, 1e-3).argmax(1) == gt).mean()
    ss_r1 = (seqslam_rescore(S, ds=5, v_min=0.8, v_max=1.2, n_vel=5).argmax(1) == gt).mean()
    print(f"[easy]  baseline={base_r1:.3f} viterbi={vit_r1:.3f} forward={ff_r1:.3f} seqslam={ss_r1:.3f}")
    assert vit_r1 >= base_r1 - 1e-9 and ff_r1 >= base_r1 - 1e-9

    # hard: perceptual aliasing (distractor가 정답보다 강함)
    S2 = 0.2 * rng.standard_normal((T, N)).astype(np.float32)
    for t in range(T):
        S2[t, gt[t]] += 0.6
    for t in rng.choice(T, T // 2, replace=False):
        S2[t, (gt[t] + 40) % N] += 0.9
    base2 = (S2.argmax(1) == gt).mean()
    vit2 = (viterbi_path(S2, 0.1, 1, 2, 1e-4) == gt).mean()
    ff2 = (forward_filter(S2, 0.1, 1, 2, 1e-4).argmax(1) == gt).mean()
    ss2 = (seqslam_rescore(S2, ds=5, v_min=0.8, v_max=1.2, n_vel=5).argmax(1) == gt).mean()
    print(f"[hard]  baseline={base2:.3f} viterbi={vit2:.3f} forward={ff2:.3f} seqslam={ss2:.3f}")
    assert vit2 > base2 + 0.2, f"viterbi 복원 실패 {vit2:.2f} vs {base2:.2f}"
    print("OK — 시간 모듈이 perceptual aliasing을 복원함")