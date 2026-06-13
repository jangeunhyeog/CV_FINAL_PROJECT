"""발표/보고서용 보강 figure 생성 (web 핸드오프 deck 용).

추가 그림:
1) delta_adaptive.png   — ★ 핵심 기여 시각화. baseline 대비 ΔR@1(%p)을 그려
                          고정-α가 night에서 음수(하락)로 내려가고 적응형은 전 조건 ≥0(무하락)임을
                          0선 기준으로 한눈에 보이게 한다. (0~100 막대에선 안 보이던 메시지)
2) recall_bars.png      — 2패널 overview 재생성(라벨/범례 겹침 수정).
3) story_overview.png   — 한 장 요약: 시간 prior(큰 이득) vs 기하(한계적)의 비대칭.

수치는 src/gated_analysis.py가 cache에서 재현한 값(검증됨).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "results/figures"
os.makedirs(FIG, exist_ok=True)

GREY, BLUE, ORANGE = "#bbbbbb", "#2c7fb8", "#d95f0e"

# --- 검증된 수치 (gated_analysis.py가 cache로 재현) ---
CONDS = ["sun", "overcast", "night"]
BASE_B = {"sun": 85.25, "overcast": 92.55, "night": 59.78}
FIXED_B = {"sun": 85.25, "overcast": 93.58, "night": 59.54}   # fixed alpha=0.9 (단일 최선)
ADAPT_B = {"sun": 85.60, "overcast": 93.81, "night": 59.90}   # adaptive (ours)


def delta_adaptive():
    """★ baseline 대비 ΔR@1(%p): night에서 고정-α는 음수(하락), 적응형은 모두 ≥0."""
    d_fixed = [FIXED_B[c] - BASE_B[c] for c in CONDS]
    d_adapt = [ADAPT_B[c] - BASE_B[c] for c in CONDS]
    x = np.arange(len(CONDS)); w = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    # baseline 아래 영역(하락) 음영
    ax.axhspan(-1.0, 0.0, color="#f2c9c0", alpha=0.5, zorder=0)
    ax.text(2.42, -0.55, "worse than\nbaseline", ha="center", va="center",
            fontsize=9, color="#a02000", style="italic")

    b1 = ax.bar(x - w/2, d_fixed, w, label="naive fixed α=0.9", color=ORANGE, zorder=3)
    b2 = ax.bar(x + w/2, d_adapt, w, label="adaptive (ours)", color=BLUE, zorder=3)
    ax.axhline(0, color="#333", lw=1.2, zorder=4)

    for rects in (b1, b2):
        for r in rects:
            h = r.get_height()
            va = "bottom" if h >= 0 else "top"
            off = 0.04 if h >= 0 else -0.04
            ax.text(r.get_x() + r.get_width()/2, h + off, f"{h:+.2f}",
                    ha="center", va=va, fontsize=10, fontweight="bold",
                    color=("#a02000" if h < 0 else "#222"))

    # night 하락 강조 화살표/주석
    ax.annotate("fixed α regresses\n(−0.24 %p)", xy=(2 - w/2, -0.24),
                xytext=(1.15, -0.78), fontsize=9.5, color="#a02000",
                arrowprops=dict(arrowstyle="->", color="#a02000", lw=1.4))

    ax.set_xticks(x); ax.set_xticklabels(CONDS, fontsize=12)
    ax.set_ylabel("Δ R@1 vs baseline (%p)", fontsize=12)
    ax.set_ylim(-1.0, 1.7)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(fontsize=11, loc="upper left")
    ax.set_title("Adaptive fusion: gain when geometry helps, no regression when it fails\n"
                 "single config (α_min=0.7, C=40), no per-condition tuning", fontsize=12)
    plt.tight_layout(); plt.savefig(f"{FIG}/delta_adaptive.png", dpi=150); plt.close()
    print(f"[saved] {FIG}/delta_adaptive.png")


def recall_bars_fixed():
    """2패널 overview — 라벨/범례 겹침 수정판."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    # 모듈 A
    groups = ["EigenPlaces", "SALAD"]; base_a = [63.65, 86.46]; plus_a = [97.09, 98.98]
    x = np.arange(len(groups)); w = 0.34
    ax1.bar(x - w/2, base_a, w, label="base", color=GREY)
    ax1.bar(x + w/2, plus_a, w, label="+ module A (temporal)", color=BLUE)
    for i in range(len(groups)):
        ax1.text(x[i] - w/2, base_a[i] + 1.5, f"{base_a[i]:.1f}", ha="center", fontsize=9, color="#555")
        ax1.text(x[i] + w/2, plus_a[i] + 1.5, f"+{plus_a[i]-base_a[i]:.1f}", ha="center",
                 fontsize=11, color=BLUE, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(groups, fontsize=11)
    ax1.set_ylim(0, 116); ax1.set_ylabel("R@1 (%)")
    ax1.legend(loc="lower center", fontsize=10)
    ax1.set_title("Module A (temporal) — Nordland\nlarge, model-agnostic gain", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # 모듈 B (adaptive)
    base_b = [BASE_B[c] for c in CONDS]; adap = [ADAPT_B[c] for c in CONDS]
    x2 = np.arange(len(CONDS))
    ax2.bar(x2 - w/2, base_b, w, label="base", color=GREY)
    ax2.bar(x2 + w/2, adap, w, label="+ module B (adaptive geo.)", color=ORANGE)
    for i in range(len(CONDS)):
        ax2.text(x2[i] + w/2, adap[i] + 1.5, f"+{adap[i]-base_b[i]:.1f}", ha="center",
                 fontsize=10, color=ORANGE, fontweight="bold")
    ax2.set_xticks(x2); ax2.set_xticklabels(CONDS, fontsize=11)
    ax2.set_ylim(0, 116); ax2.set_ylabel("R@1 (%)")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.set_title("Module B (geometric) — SVOX\nmarginal on a strong descriptor", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout(); plt.savefig(f"{FIG}/recall_bars.png", dpi=140); plt.close()
    print(f"[saved] {FIG}/recall_bars.png")


def story_overview():
    """한 장 요약: 후처리별 최대 ΔR@1 — 시간 prior >> 기하."""
    labels = ["Temporal\n(EigenPlaces)", "Temporal\n(SALAD)",
              "Geometric\n(overcast)", "Geometric\n(night)"]
    gains = [33.44, 12.52, 1.26, 0.12]
    colors = [BLUE, BLUE, ORANGE, ORANGE]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    bars = ax.bar(x, gains, 0.6, color=colors)
    for r, g in zip(bars, gains):
        ax.text(r.get_x() + r.get_width()/2, g + 0.6, f"+{g:.1f}", ha="center",
                fontsize=12, fontweight="bold", color=r.get_facecolor())
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("R@1 gain over baseline (%p)", fontsize=12)
    ax.set_ylim(0, 38)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Training-free post-processing: the temporal (sequence) prior dominates",
                 fontsize=12.5)
    # 구분 안내
    ax.text(0.5, 35.5, "sequence cue", ha="center", color=BLUE, fontsize=10, fontweight="bold")
    ax.text(2.5, 35.5, "geometric cue", ha="center", color=ORANGE, fontsize=10, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{FIG}/story_overview.png", dpi=150); plt.close()
    print(f"[saved] {FIG}/story_overview.png")


if __name__ == "__main__":
    delta_adaptive()
    recall_bars_fixed()
    story_overview()
