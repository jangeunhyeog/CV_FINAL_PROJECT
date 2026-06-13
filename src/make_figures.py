"""M4 — 보고서용 figure 생성. results/figures/에 저장.

1) 유사도 행렬 heatmap (Nordland, 서브샘플) — 대각선 구조 확인용
2) baseline vs +A vs +B R@1 막대그래프 (results.csv 기반)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eval_recall as ev

FIG = "results/figures"
os.makedirs(FIG, exist_ok=True)


def _find_cache(*candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def heatmap_nordland(sub=400):
    db = _find_cache("cache/nordland_eigenplaces_test_db.npy",
                     "cache/nordland_eigenplaces_test_db_database.npy")
    q = _find_cache("cache/nordland_eigenplaces_test_q.npy",
                    "cache/nordland_eigenplaces_test_q_queries.npy")
    if not (db and q):
        print("[skip] nordland cache 없음"); return
    D, Q = np.load(db), np.load(q)
    # 앞쪽 sub개 구간만(대각선 구조 보기 좋게)
    S = ev.cosine_similarity(Q[:sub], D[:sub])
    plt.figure(figsize=(5, 5))
    plt.imshow(S, cmap="viridis", aspect="auto")
    plt.colorbar(label="cosine similarity")
    plt.xlabel("DB index (summer)"); plt.ylabel("query index (winter)")
    plt.title(f"Nordland similarity matrix (first {sub})\nbright diagonal = correct sequence")
    plt.tight_layout(); plt.savefig(f"{FIG}/nordland_heatmap.png", dpi=130)
    plt.close()
    print(f"[saved] {FIG}/nordland_heatmap.png")


def two_panel_bars():
    """좌: 모듈 A(Nordland, base vs +A, 두 모델). 우: 모듈 B(SVOX, 조건별 base vs +B)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # 모듈 A
    groups = ["EigenPlaces", "SALAD"]
    base_a = [63.65, 86.46]
    plus_a = [97.09, 98.98]
    x = np.arange(len(groups)); w = 0.35
    ax1.bar(x - w/2, base_a, w, label="base", color="#bbbbbb")
    ax1.bar(x + w/2, plus_a, w, label="+ module A", color="#2c7fb8")
    for i in range(len(groups)):
        ax1.text(x[i] + w/2, plus_a[i] + 1, f"+{plus_a[i]-base_a[i]:.1f}",
                 ha="center", fontsize=9, color="#2c7fb8")
    ax1.set_xticks(x); ax1.set_xticklabels(groups)
    ax1.set_ylim(0, 105); ax1.set_ylabel("R@1 (%)"); ax1.legend()
    ax1.set_title("Module A (temporal) — Nordland")
    ax1.grid(axis="y", alpha=0.3)

    # 모듈 B
    conds = ["sun", "overcast", "night"]
    base_b = [85.25, 92.55, 59.78]
    plus_b = [85.48, 94.38, 59.90]
    x2 = np.arange(len(conds))
    ax2.bar(x2 - w/2, base_b, w, label="base", color="#bbbbbb")
    ax2.bar(x2 + w/2, plus_b, w, label="+ module B", color="#d95f0e")
    for i in range(len(conds)):
        ax2.text(x2[i] + w/2, plus_b[i] + 1, f"+{plus_b[i]-base_b[i]:.1f}",
                 ha="center", fontsize=9, color="#d95f0e")
    ax2.set_xticks(x2); ax2.set_xticklabels(conds)
    ax2.set_ylim(0, 105); ax2.set_ylabel("R@1 (%)"); ax2.legend()
    ax2.set_title("Module B (geometric) — SVOX (EigenPlaces)")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout(); plt.savefig(f"{FIG}/recall_bars.png", dpi=130)
    plt.close()
    print(f"[saved] {FIG}/recall_bars.png")


def module_a_bars():
    """슬라이드용 — 모듈 A 단독(Nordland) 막대."""
    groups = ["EigenPlaces", "SALAD"]
    base_a = [63.65, 86.46]; plus_a = [97.09, 98.98]
    x = np.arange(len(groups)); w = 0.36
    plt.figure(figsize=(6.2, 4.2))
    plt.bar(x - w/2, base_a, w, label="base (single-frame)", color="#bbbbbb")
    plt.bar(x + w/2, plus_a, w, label="+ temporal filtering", color="#2c7fb8")
    for i in range(len(groups)):
        plt.text(x[i]+w/2, plus_a[i]+1.2, f"+{plus_a[i]-base_a[i]:.1f}p",
                 ha="center", fontsize=13, color="#2c7fb8", fontweight="bold")
        plt.text(x[i]-w/2, base_a[i]+1.2, f"{base_a[i]:.1f}", ha="center", fontsize=10, color="#555")
    plt.xticks(x, groups, fontsize=12); plt.ylim(0, 108); plt.ylabel("R@1 (%)")
    plt.legend(fontsize=11, loc="lower right"); plt.grid(axis="y", alpha=0.3)
    plt.title("Module A: Temporal Filtering (Nordland)", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{FIG}/bars_module_a.png", dpi=140); plt.close()
    print(f"[saved] {FIG}/bars_module_a.png")


def module_b_bars():
    """슬라이드용 — 모듈 B 단독(SVOX 조건별, base vs adaptive)."""
    conds = ["sun", "overcast", "night"]
    base_b = [85.25, 92.55, 59.78]; adap = [85.60, 93.81, 59.90]
    x = np.arange(len(conds)); w = 0.36
    plt.figure(figsize=(6.2, 4.2))
    plt.bar(x - w/2, base_b, w, label="base", color="#bbbbbb")
    plt.bar(x + w/2, adap, w, label="+ adaptive geo. re-rank", color="#d95f0e")
    for i in range(len(conds)):
        d = adap[i]-base_b[i]
        plt.text(x[i]+w/2, adap[i]+1.2, f"+{d:.1f}", ha="center", fontsize=12,
                 color="#d95f0e", fontweight="bold")
    plt.xticks(x, conds, fontsize=12); plt.ylim(0, 108); plt.ylabel("R@1 (%)")
    plt.legend(fontsize=11, loc="lower left"); plt.grid(axis="y", alpha=0.3)
    plt.title("Module B: Geometric verification is marginal (SVOX)", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{FIG}/bars_module_b.png", dpi=140); plt.close()
    print(f"[saved] {FIG}/bars_module_b.png")


if __name__ == "__main__":
    heatmap_nordland()
    two_panel_bars()
    module_a_bars()
    module_b_bars()
