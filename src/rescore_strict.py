"""Strict-threshold re-scoring for Nordland (dense ~2.4m frame spacing).

Reuses the project's OWN modules (datasets, temporal_filter, eval_recall) so the
retrieval + temporal-filter algorithms are identical to run_experiment.py — only
the positives threshold changes. Uses the cached descriptors, so no re-extraction.

Reports base / +SeqSLAM(offline) / +forward(online) at 25m (standard) and 2.5m
(=+-1 frame, strict) for EigenPlaces and SALAD.
"""
import os, sys, glob
import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import datasets as ds
import eval_recall as ev
import temporal_filter as tf

CACHE = "/home/jeh/CV_project/vpr-postproc/cache"
NORD = "/home/jeh/CV_project/vpr-postproc/vpr-datasets-downloader/datasets/nordland/images/test"

db_paths = sorted(glob.glob(f"{NORD}/database/*"))
q_paths = sorted(glob.glob(f"{NORD}/queries/*"))
db_utm, q_utm = ds.utm_array(db_paths), ds.utm_array(q_paths)
print(f"db={len(db_paths)} q={len(q_paths)}")

# positives per threshold (reuses project's protocol code)
POS = {th: ds.build_positives(q_utm, db_utm, thresh_m=th) for th in (25.0, 2.5, 2.0)}
for th, p in POS.items():
    print(f"  thresh={th}m  avg_positives/query={np.mean([len(s) for s in p]):.2f}")

MODELS = {
    "EigenPlaces": (f"{CACHE}/nordland_eigenplaces_test_db.npy",
                    f"{CACHE}/nordland_eigenplaces_test_q.npy"),
    "SALAD": (f"{CACHE}/nordland_salad_test_db_database.npy",
              f"{CACHE}/nordland_salad_test_q_queries.npy"),
}


def rec_line(name, S):
    out = {}
    preds = ev.topk_from_scores(S, 10)
    for th in (25.0, 2.5, 2.0):
        out[th] = ev.recall_at_k(preds, POS[th], ks=(1, 5, 10))
    print(f"  {name:<26} "
          + "  ".join(f"@{th}m R@1={out[th][1]*100:5.2f} R@5={out[th][5]*100:5.2f}"
                      for th in (25.0, 2.5, 2.0)))
    return out


for mname, (dbf, qf) in MODELS.items():
    if not (os.path.exists(dbf) and os.path.exists(qf)):
        print(f"[skip] {mname}: cache missing"); continue
    D = np.load(dbf).astype(np.float32)
    Q = np.load(qf).astype(np.float32)
    print(f"\n=== {mname}  D={D.shape} Q={Q.shape} ===")
    S = ev.cosine_similarity(Q, D)             # base (same as run_experiment)
    del D, Q                                   # free descriptors before the temporal filter
    rec_line("base", S)
    if mname == "EigenPlaces":                 # online variant only needs S; do it before seqslam frees S
        rec_line("+temporal forward(online)", tf.forward_filter(S, tau=0.1, v_min=1, v_max=1))
    S_seq = tf.seqslam_rescore(S, ds=10, v_min=0.8, v_max=1.2, n_vel=5)
    del S
    rec_line("+temporal SeqSLAM(off)", S_seq)
    del S_seq
