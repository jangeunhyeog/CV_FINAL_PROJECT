"""실시간/왜-딥러닝 분석 그림 3종 생성 → results/figures/.

수치는 eval_realtime.py / eval_no_dl.py 가 캐시에서 재현한 검증값(하드코딩).
  1) fig_realtime_accuracy_vs_latency.png — 정확도 vs 지연(온라인 인과 필터가 최고+실시간)
  2) fig_why_deeplearning.png            — 같은 과제에서 기하만 vs 딥러닝
  3) fig_latency_budget.png              — 지연 로그 막대 + 카메라 프레임 예산
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "results/figures"
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, GREY, GREEN, RED = "#2c7fb8", "#d95f0e", "#999999", "#1a8a2e", "#c0271a"

# --- 검증값 (전체 Nordland, EigenPlaces frozen): (이름, R@1, ms/q, 색, 마커, 라벨오프셋, 화살표) ---
RT = [
    ("single-frame (base)", 63.65, 0.08, GREY, "o", (12, -2), False),
    ("temporal OFFLINE (SeqSLAM)", 97.09, 1.78, ORANGE, "s", (-12, -26), True),
    ("temporal OFFLINE (Viterbi)", 98.69, 0.41, ORANGE, "^", (-150, 22), True),
    ("temporal ONLINE (causal = real-time)\n98.76% @ 0.46 ms", 98.76, 0.46, BLUE, "*", (20, -52), True),
]
# --- 검증값 (동일 400장 부분집합): 기하만 vs 딥러닝 ---
GEO_ONLY, GEO_TEMPORAL, DEEP, DEEP_TEMPORAL = 30.75, 89.25, 96.0, 100.0
GEO_MS_400 = 382.6   # 기하만: 쿼리 1장 vs 400 DB


def fig_accuracy_vs_latency():
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for name, r1, ms, c, m, off, arrow in RT:
        ax.scatter(ms, r1, s=420 if m == "*" else 150, c=c, marker=m, zorder=4,
                   edgecolor="k", linewidth=0.7)
        ax.annotate(name, (ms, r1), textcoords="offset points", xytext=off, fontsize=10,
                    fontweight="bold" if m == "*" else "normal",
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.3) if arrow else None)
    ax.axhline(63.65, color=GREY, ls="--", lw=1, alpha=0.7)
    ax.set_xscale("log"); ax.set_xlim(0.05, 4); ax.set_ylim(56, 106)
    ax.set_xlabel("latency  (ms / query, log scale)", fontsize=12)
    ax.set_ylabel("R@1 (%)", fontsize=12)
    ax.set_title("Real-time temporal filtering on Nordland (EigenPlaces, frozen)\n"
                 "the causal online filter is the most accurate AND real-time", fontsize=12, pad=12)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_realtime_accuracy_vs_latency.png", dpi=150); plt.close()
    print("saved fig_realtime_accuracy_vs_latency.png")


def fig_why_deeplearning():
    labels = ["geometric only\n(no deep learning)", "geometric only\n+ temporal",
              "deep descriptor\n(frozen)", "deep\n+ temporal"]
    vals = [GEO_ONLY, GEO_TEMPORAL, DEEP, DEEP_TEMPORAL]
    cols = [RED, ORANGE, BLUE, GREEN]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(range(4), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("R@1 (%)", fontsize=12); ax.set_ylim(0, 108)
    ax.set_title("Why the learned descriptor matters (same 400-image retrieval task)\n"
                 "classical geometric matching alone is weak; the deep descriptor is essential", fontsize=11.5)
    ax.grid(axis="y", alpha=0.3)
    ax.annotate("+65 %p", xy=(1.5, 63), fontsize=12, color=BLUE, fontweight="bold", ha="center")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_why_deeplearning.png", dpi=150); plt.close()
    print("saved fig_why_deeplearning.png")


def fig_latency_budget():
    names = ["base", "temporal\nonline", "temporal\noffline",
             "geometric-only\n(400 DB)", "geometric-only\n(full 27k DB,\nextrapolated)"]
    lat = [0.08, 0.46, 1.78, GEO_MS_400, GEO_MS_400 / 400 * 27592]
    cols = [GREY, BLUE, ORANGE, RED, RED]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(range(len(names)), lat, color=cols, width=0.6)
    ax.set_yscale("log")
    for i, v in enumerate(lat):
        ax.text(i, v * 1.3, f"{v:.2f} ms" if v < 1000 else f"{v/1000:.1f} s",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhspan(33, 100, color="#cfe8cf", alpha=0.7, zorder=0)
    ax.text(0.1, 55, "camera frame budget (10-30 Hz)", fontsize=9, color=GREEN, style="italic")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("latency per query (ms, log)", fontsize=12)
    ax.set_title("Real-time feasibility: temporal filtering is ~1000x faster than geometric matching\n"
                 "geometric-only retrieval scales with DB size and blows the frame budget", fontsize=11)
    ax.grid(axis="y", alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_latency_budget.png", dpi=150); plt.close()
    print("saved fig_latency_budget.png")


def fig_classical_base():
    """고전 base 사다리: (좌) 단독 descriptor 품질, (우) 전체 Nordland HOG vs 딥 +시간."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    # 좌: 단독 single-frame 품질 (400장 부분집합)
    names = ["raw-pixel\nSAD", "CLAHE+HOG", "deep\n(frozen)"]
    vals = [15.0, 76.0, 96.0]; cols = [RED, ORANGE, BLUE]
    ax1.bar(range(3), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax1.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(3)); ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylim(0, 108); ax1.set_ylabel("R@1 (%)", fontsize=11)
    ax1.set_title("Single-frame descriptor quality\n(400-img subset): HOG >> raw pixels", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)
    # 우: 전체 Nordland, HOG vs deep, 단계별
    stages = ["single", "+SeqSLAM", "+online\n(real-time)"]
    hog = [25.32, 93.95, 96.09]; deep = [63.65, 97.09, 98.76]
    x = np.arange(3); w = 0.36
    ax2.bar(x - w/2, hog, w, label="CLAHE+HOG (no deep learning)", color=ORANGE)
    ax2.bar(x + w/2, deep, w, label="deep (frozen)", color=BLUE)
    for i in range(3):
        ax2.text(x[i]-w/2, hog[i]+1.5, f"{hog[i]:.1f}", ha="center", fontsize=9, color=ORANGE, fontweight="bold")
        ax2.text(x[i]+w/2, deep[i]+1.5, f"{deep[i]:.1f}", ha="center", fontsize=9, color=BLUE, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(stages, fontsize=10)
    ax2.set_ylim(0, 112); ax2.set_ylabel("R@1 (%)", fontsize=11); ax2.legend(fontsize=9, loc="lower right")
    ax2.set_title("Full Nordland: classical base + sequence is good,\nbut deep wins and is far more robust single-frame", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_classical_base.png", dpi=150); plt.close()
    print("saved fig_classical_base.png")


def fig_descriptor_fusion():
    """deep + (CLAHE+HOG) 점수 융합 — 단독 R@1 vs deep 가중치 a (전체 Nordland)."""
    a = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    r1 = [25.32, 71.15, 74.96, 71.67, 66.33, 63.65]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(a, r1, "-o", color=BLUE, lw=2, ms=7)
    ax.scatter([1.0], [63.65], s=120, color=GREY, zorder=5, label="deep only (a=1)")
    ax.scatter([0.0], [25.32], s=120, color=RED, zorder=5, label="HOG only (a=0)")
    ax.scatter([0.5], [74.96], s=200, color=GREEN, marker="*", zorder=6, label="best fusion (a=0.5)")
    ax.annotate("74.96  (+11.3 over deep)", (0.5, 74.96), textcoords="offset points",
                xytext=(6, 10), fontsize=11, fontweight="bold", color=GREEN)
    ax.set_xlabel("fusion weight on deep descriptor  (a)", fontsize=12)
    ax.set_ylabel("single-frame R@1 (%)", fontsize=12)
    ax.set_ylim(20, 82); ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="lower center")
    ax.set_title("Training-free descriptor fusion: deep + CLAHE/HOG\n"
                 "a hand-crafted cue is complementary, lifting single-frame R@1 by +11", fontsize=11.5)
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_descriptor_fusion.png", dpi=150); plt.close()
    print("saved fig_descriptor_fusion.png")


if __name__ == "__main__":
    fig_accuracy_vs_latency()
    fig_why_deeplearning()
    fig_latency_budget()
    fig_classical_base()
    fig_descriptor_fusion()
