"""M4 정성 결과 — Nordland에서 baseline top-1은 틀리고 +모듈A top-1은 맞은 사례 montage.
각 행: [query(winter) | baseline top-1(wrong, summer) | +A top-1(correct, summer)].
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import datasets as ds
import eval_recall as ev
import temporal_filter as tf

FIG = "results/figures"
os.makedirs(FIG, exist_ok=True)


def main(n_examples=4):
    data = ds.load_geo_dataset("vpr-datasets-downloader/datasets/nordland")
    D = np.load("cache/nordland_eigenplaces_test_db.npy")
    Q = np.load("cache/nordland_eigenplaces_test_q.npy")
    S = ev.cosine_similarity(Q, D)
    base = S.argmax(1)
    Sa = tf.seqslam_rescore(S, ds=10, v_min=0.8, v_max=1.2, n_vel=5)
    aft = Sa.argmax(1)
    pos = data["positives"]

    picks = [i for i in range(len(pos))
             if pos[i] and base[i] not in pos[i] and aft[i] in pos[i]]
    # 시퀀스 전반에 퍼지게 균등 샘플
    if len(picks) > n_examples:
        picks = [picks[k] for k in np.linspace(0, len(picks)-1, n_examples).astype(int)]
    print(f"baseline-wrong & A-correct 사례: {len([i for i in range(len(pos)) if pos[i] and base[i] not in pos[i] and aft[i] in pos[i]])}개 중 {len(picks)}개 표시")

    fig, axes = plt.subplots(len(picks), 3, figsize=(8.4, 2.55 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    titles = ["query (winter)", "baseline top-1  (WRONG)", "+ module A top-1  (correct)"]
    tcolors = ["#222222", "#c0271a", "#1a8a2e"]
    framecolors = ["#888888", "#c0271a", "#1a8a2e"]
    for r, i in enumerate(picks):
        imgs = [data["q_paths"][i], data["db_paths"][base[i]], data["db_paths"][aft[i]]]
        for c, p in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(Image.open(p).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(framecolors[c]); s.set_linewidth(3.0)
            if r == 0:
                ax.set_title(titles[c], fontsize=13, fontweight="bold", color=tcolors[c], pad=8)
    fig.suptitle("Single-frame retrieval picks a look-alike track;\n"
                 "the sequence prior recovers the correct place",
                 fontsize=12, y=0.997)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(f"{FIG}/qualitative_nordland.png", dpi=170)
    plt.close()
    print(f"[saved] {FIG}/qualitative_nordland.png")


if __name__ == "__main__":
    main()
