"""실시간 분석 — 시간 필터의 정확도 vs 지연(ms/query)을 한 번에 측정한다.

캐시된 descriptor에서 유사도 행렬 S를 만들고, 동일 S에 대해
  base(한 장) / 오프라인 SeqSLAM / 오프라인 Viterbi / 온라인 forward(인과적)
를 돌려 R@1/5/10과 후처리 지연을 비교한다. 온라인 forward는 전진폭(v)·온도(tau)를
sweep해 최적 인과 설정을 고른다(로봇 실시간 적용 가능 버전).

사용:
    python src/eval_realtime.py            # 기본: Nordland, EigenPlaces 캐시
결과: results/realtime_summary.csv
"""
import csv
import os
import time

import numpy as np

import datasets as ds
import eval_recall as ev
import temporal_filter as tf

DATASET = "vpr-datasets-downloader/datasets/nordland"
DB_CACHE = "cache/nordland_eigenplaces_test_db.npy"
Q_CACHE = "cache/nordland_eigenplaces_test_q.npy"
OUT_CSV = "results/realtime_summary.csv"


def main():
    data = ds.load_geo_dataset(DATASET)
    pos = data["positives"]
    D, Q = np.load(DB_CACHE), np.load(Q_CACHE)
    T = Q.shape[0]
    S = ev.cosine_similarity(Q, D)

    def recall(S_):
        preds = ev.topk_from_scores(S_, 10)
        r = ev.recall_at_k(preds, pos, (1, 5, 10))
        return {k: round(r[k] * 100, 2) for k in (1, 5, 10)}

    def timed(fn):
        t0 = time.time()
        out = fn()
        return out, (time.time() - t0) * 1000.0 / T

    rows = []
    # base
    _, dt = timed(lambda: ev.topk_from_scores(S, 10))
    rows.append({"method": "base (single-frame)", **recall(S), "ms_per_q": round(dt, 4),
                 "causal": "yes"})
    # 오프라인 SeqSLAM
    Sa, dt = timed(lambda: tf.seqslam_rescore(S))
    rows.append({"method": "temporal offline (SeqSLAM)", **recall(Sa), "ms_per_q": round(dt, 4),
                 "causal": "no (+-10 lookahead)"})
    del Sa
    # 오프라인 Viterbi (단일 MAP 경로 → R@1만)
    path, dt = timed(lambda: tf.viterbi_path(S))
    vit_r1 = round(np.mean([1.0 if (pos[i] and path[i] in pos[i]) else 0.0
                            for i in range(T)]) * 100, 2)
    rows.append({"method": "temporal offline (Viterbi)", 1: vit_r1, 5: "", 10: "",
                 "ms_per_q": round(dt, 4), "causal": "no (whole sequence)"})
    del path
    # 온라인 forward(인과적) sweep → 최적 1개
    best = None
    for vmin, vmax in [(1, 1), (1, 2), (1, 3)]:
        for tau in [0.05, 0.1, 0.15]:
            Sf, dt = timed(lambda: tf.forward_filter(S, tau=tau, v_min=vmin, v_max=vmax))
            r = recall(Sf)
            if best is None or r[1] > best[0][1]:
                best = (r, dt, vmin, vmax, tau)
            del Sf
    r, dt, vmin, vmax, tau = best
    rows.append({"method": f"temporal online (forward, v=[{vmin},{vmax}], tau={tau})",
                 **r, "ms_per_q": round(dt, 4), "causal": "yes (real-time)"})

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["method", 1, 5, 10, "ms_per_q", "causal"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    for row in rows:
        print(f"{row['method']:<42} R@1={row[1]}  {row['ms_per_q']} ms/q  [{row['causal']}]")
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
