"""실험 러너 — config 하나로 전체 파이프라인 실행, results/results.csv에 append.

M1(baseline) 사용:
    python src/run_experiment.py --dataset vpr-datasets-downloader/datasets/st_lucia \
        --model eigenplaces

모듈 A/B는 추후 --module-a / --module-b 로 켠다(현재는 baseline 골격).
descriptor는 cache/에 (dataset,model,split)별로 1회만 추출·재사용한다.
"""
import argparse
import csv
import hashlib
import os
import time

import numpy as np
import torch

import datasets as ds
import eval_recall as ev
import temporal_filter as tf
import rerank_geometric as rg
from extract_features import build_transform, extract, get_model

CACHE_DIR = "cache"
RESULTS_CSV = "results/results.csv"


def cached_descriptors(model_name, dataset_dir, split, kind, paths, img_size, batch_size, device):
    """(dataset,model,split,kind)별 descriptor 캐시. 있으면 로드, 없으면 추출."""
    tag = f"{os.path.basename(dataset_dir.rstrip('/'))}_{model_name}_{split}_{kind}"
    cache_path = os.path.join(CACHE_DIR, tag + ".npy")
    if os.path.exists(cache_path):
        print(f"[cache] load {cache_path}")
        return np.load(cache_path)
    print(f"[extract] {kind}: {len(paths)} images")
    model = get_model(model_name, device)
    tf = build_transform(img_size)
    descs, _ = extract(model, paths, tf, device, batch_size)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_path, descs)
    del model
    torch.cuda.empty_cache()
    return descs


def append_result(row: dict):
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="eigenplaces")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--thresh-m", type=float, default=ds.DEFAULT_POS_THRESH_M)
    ap.add_argument("--db-subdir", default="database", help="DB 폴더명(SVOX=gallery)")
    ap.add_argument("--q-subdir", default="queries", help="쿼리 폴더명(SVOX=queries_night 등)")
    ap.add_argument("--module-a", action="store_true", help="시간 일관성 필터링")
    ap.add_argument("--ta-method", default="seqslam", choices=["seqslam", "forward"])
    ap.add_argument("--ta-ds", type=int, default=10, help="시퀀스 윈도 절반 길이")
    ap.add_argument("--ta-vmin", type=float, default=0.8)
    ap.add_argument("--ta-vmax", type=float, default=1.2)
    ap.add_argument("--ta-nvel", type=int, default=5)
    ap.add_argument("--ta-tau", type=float, default=0.1, help="forward filter 온도")
    ap.add_argument("--module-b", action="store_true", help="기하 re-ranking")
    ap.add_argument("--tb-k", type=int, default=20, help="re-ranking 후보 수")
    ap.add_argument("--tb-alpha", type=float, default=0.5, help="global vs inlier 융합 가중")
    ap.add_argument("--tb-detector", default="sift", choices=["sift", "orb"])
    ap.add_argument("--tb-method", default="magsac", choices=["magsac", "fundamental"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data = ds.load_geo_dataset(args.dataset, args.split, args.thresh_m,
                               db_sub=args.db_subdir, q_sub=args.q_subdir)
    D = cached_descriptors(args.model, args.dataset, args.split, f"db_{args.db_subdir}",
                           data["db_paths"], args.img_size, args.batch_size, args.device)
    Q = cached_descriptors(args.model, args.dataset, args.split, f"q_{args.q_subdir}",
                           data["q_paths"], args.img_size, args.batch_size, args.device)
    print(f"D={D.shape} Q={Q.shape}")

    S = ev.cosine_similarity(Q, D)

    t0 = time.time()
    if args.module_a:
        if args.ta_method == "seqslam":
            S = tf.seqslam_rescore(S, ds=args.ta_ds, v_min=args.ta_vmin,
                                   v_max=args.ta_vmax, n_vel=args.ta_nvel)
        else:  # forward filter (online Bayes)
            S = tf.forward_filter(S, tau=args.ta_tau,
                                  v_min=max(1, int(args.ta_vmin)),
                                  v_max=max(1, int(round(args.ta_vmax))))
    if args.module_b:
        preds = rg.rerank(S, data["q_paths"], data["db_paths"],
                          k=args.tb_k, alpha=args.tb_alpha,
                          detector=args.tb_detector, method=args.tb_method)
        res = ev.recall_at_k(preds, data["positives"], ks=(1, 5, 10))
    else:
        res = ev.evaluate(S, data["positives"], ks=(1, 5, 10))
    dt = (time.time() - t0) * 1000 / max(1, len(Q))

    cfg_str = (f"{args.dataset}|{args.q_subdir}|{args.model}|A={args.module_a}"
               f"|B={args.module_b}|{args.thresh_m}")
    cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
    print(f"\n=== {args.model} on {os.path.basename(args.dataset)} ({args.q_subdir}) ===")
    print(f"R@1={res[1]*100:.2f}  R@5={res[5]*100:.2f}  R@10={res[10]*100:.2f}  "
          f"({dt:.3f} ms/query post-proc)")

    append_result({
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cfg_hash": cfg_hash,
        "dataset": os.path.basename(args.dataset.rstrip("/")),
        "condition": args.q_subdir,
        "model": args.model,
        "module_a": args.module_a,
        "module_b": args.module_b,
        "thresh_m": args.thresh_m,
        "n_db": len(data["db_paths"]),
        "n_q": len(data["q_paths"]),
        "R@1": round(res[1] * 100, 2),
        "R@5": round(res[5] * 100, 2),
        "R@10": round(res[10] * 100, 2),
        "ms_per_query": round(dt, 4),
    })


if __name__ == "__main__":
    main()
