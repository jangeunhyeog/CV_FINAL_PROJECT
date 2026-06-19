"""보고서용 그림 2종 → results/figures/.
  1) fig_threshold_sensitivity.png — 정답 허용거리(2.5~25m)별 R@1 (정밀 localization)
  2) fig_dtw_robustness.png        — 변속 시나리오에서 정렬기 강건성 (왜 DTW)
수치는 eval_threshold_sensitivity.py / eval_dtw_robustness.py 가 재현한 검증값.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "results/figures"
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, GREY, GREEN, RED, DRED = "#2c7fb8", "#d95f0e", "#999999", "#1a8a2e", "#c0271a", "#7a0010"


def fig_threshold():
    th = [2.5, 5, 10, 25]
    series = [
        ("deep + online (temporal)", [95.86, 97.35, 98.76, 98.76], BLUE, "-o"),
        ("HOG + online (temporal)", [94.46, 95.40, 96.09, 96.09], ORANGE, "-o"),
        ("deep + HOG fusion (single)", [68.20, 71.60, 73.81, 74.96], GREEN, "-s"),
        ("deep single-frame", [55.98, 59.66, 62.05, 63.65], GREY, "--^"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for name, ys, c, st in series:
        ax.plot(th, ys, st, color=c, lw=2, ms=7, label=name)
    ax.set_xticks(th); ax.set_xlabel("ground-truth tolerance (m)", fontsize=12)
    ax.set_ylabel("R@1 (%)", fontsize=12); ax.set_ylim(50, 102)
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="center right")
    ax.set_title("Localization precision (Nordland): temporal stays high even at strict 2.5 m\n"
                 "the sequence prior gives precise matches, not just loose ones", fontsize=11.5)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_threshold_sensitivity.png", dpi=150); plt.close()
    print("saved fig_threshold_sensitivity.png")


def fig_dtw_robustness():
    names = ["single-frame", "forward\nfixed v=1", "SeqSLAM\n(fixed v~1)", "DTW\n(no speed assump.)", "forward\nwide v=1..3"]
    r25 = [86.40, 23.27, 2.27, 92.51, 96.28]   # @ 2.5 m (strict)
    cols = [GREY, RED, DRED, BLUE, GREEN]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(range(len(names)), r25, color=cols, width=0.62)
    for i, v in enumerate(r25):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(86.40, color=GREY, ls="--", lw=1.2, alpha=0.8)
    ax.text(4.4, 88, "single-frame", fontsize=8.5, color=GREY, ha="right")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("R@1 @ 2.5 m (%)", fontsize=12); ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Robustness to variable speed (avg step ~2x, Nordland subset)\n"
                 "fixed-velocity temporal collapses below single-frame; DTW stays robust", fontsize=11.5)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_dtw_robustness.png", dpi=150); plt.close()
    print("saved fig_dtw_robustness.png")


if __name__ == "__main__":
    fig_threshold()
    fig_dtw_robustness()
