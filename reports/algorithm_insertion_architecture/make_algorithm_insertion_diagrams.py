from __future__ import annotations

from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent

INK = "#263238"
MUTED = "#66737A"
GREY = "#AEB7BC"
GREY_LIGHT = "#F3F5F6"
BLUE = "#2F6B9A"
BLUE_LIGHT = "#EAF2F8"
GOLD = "#D39A12"
GOLD_LIGHT = "#FBF3DB"
OLIVE = "#778A43"
OLIVE_LIGHT = "#EFF3E5"


def box(ax, x, y, width, height, *, face, edge, linewidth=1.5, linestyle="-", radius=0.18):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle=f"round,pad=0.02,rounding_size={radius}",
            facecolor=face, edgecolor=edge, linewidth=linewidth, linestyle=linestyle,
        )
    )


def label(ax, x, y, value, *, size=10, color=INK, weight="normal", ha="center", va="center"):
    ax.text(x, y, value, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, linespacing=1.35)


def arrow(ax, start, end, *, color=GREY, width=1.6, connection="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14, linewidth=width,
            color=color, connectionstyle=connection, shrinkA=4, shrinkB=4,
        )
    )


def setup(title, subtitle, accent):
    fig, ax = plt.subplots(figsize=(16, 7.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    label(ax, 8, 7.18, title, size=21, weight="bold")
    label(ax, 8, 6.76, subtitle, size=11, color=MUTED)
    box(ax, 0.55, 1.70, 14.90, 4.45, face="#FCFDFD", edge=GREY, linewidth=1.3, linestyle="--", radius=0.24)
    label(ax, 0.86, 5.90, "MATD3的Actor", size=10.2, color=accent, weight="bold", ha="left")
    return fig, ax


def standard_input(ax):
    box(ax, 0.95, 3.20, 2.05, 1.38, face=GREY_LIGHT, edge=GREY)
    label(ax, 1.98, 4.13, "微网观测", size=11.2, weight="bold")
    label(ax, 1.98, 3.64, "本地状态 · 未来信息\n邻居状态 · 市场信息", size=8.8, color=MUTED)


def standard_output(ax, module_end, accent):
    arrow(ax, (module_end, 3.89), (11.65, 3.89), color=accent)
    box(ax, 11.65, 3.20, 1.60, 1.38, face=GREY_LIGHT, edge=GREY)
    label(ax, 12.45, 4.12, "Actor动作", size=11.0, weight="bold")
    label(ax, 12.45, 3.64, "储能与P2P决策", size=8.8, color=MUTED)
    arrow(ax, (13.25, 3.89), (13.75, 3.89), color=GREY)
    box(ax, 13.75, 3.20, 1.15, 1.38, face=GREY_LIGHT, edge=GREY)
    label(ax, 14.33, 4.12, "环境", size=11.0, weight="bold")
    label(ax, 14.33, 3.64, "执行调度", size=8.8, color=MUTED)


def footer(ax):
    box(ax, 1.00, 0.45, 14.00, 0.72, face=GREY_LIGHT, edge=GREY, linewidth=1.1)
    label(ax, 2.02, 0.81, "MATD3训练机制保持不变", size=9.5, weight="bold")
    label(ax, 5.20, 0.81, "经验回放", size=9.1, color=MUTED)
    label(ax, 8.00, 0.81, "双集中Critic", size=9.1, color=MUTED)
    label(ax, 10.85, 0.81, "TD3更新", size=9.1, color=MUTED)
    label(ax, 13.45, 0.81, "目标网络", size=9.1, color=MUTED)


def save(fig, filename):
    path = OUTPUT_DIR / filename
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(path)


def module_steps(ax, color, light, title, steps):
    box(ax, 4.05, 2.72, 6.35, 2.34, face=light, edge=color, linewidth=1.8, radius=0.24)
    label(ax, 7.23, 4.68, title, size=14, color=color, weight="bold")
    positions = [(4.55, 1.55), (6.55, 1.80), (8.80, 1.10)]
    for index, ((x, width), step) in enumerate(zip(positions, steps)):
        box(ax, x, 3.25, width, 0.88, face="white", edge=color, linewidth=1.1)
        label(ax, x + width / 2, 3.69, step, size=10.2, weight="bold")
        if index < 2:
            next_x = positions[index + 1][0]
            arrow(ax, (x + width, 3.69), (next_x, 3.69), color=color)


def draw_pnn():
    fig, ax = setup("MATD3 + PNN", "PNN加入Actor的特征处理位置，学习不同信息之间的交互关系", GOLD)
    standard_input(ax)
    arrow(ax, (3.00, 3.89), (4.05, 3.89), color=GOLD)
    module_steps(ax, GOLD, GOLD_LIGHT, "PNN特征交互模块", ["特征分组", "成对交互", "融合"])
    standard_output(ax, 10.40, GOLD)
    footer(ax)
    save(fig, "01_matd3_plus_pnn.png")


def draw_transformer():
    fig, ax = setup("MATD3 + Transformer", "Transformer加入Actor的特征处理位置，关注重要的时间与空间信息", BLUE)
    standard_input(ax)
    arrow(ax, (3.00, 3.89), (4.05, 3.89), color=BLUE)
    module_steps(ax, BLUE, BLUE_LIGHT, "Transformer注意力模块", ["信息序列", "注意力提取", "汇总"])
    standard_output(ax, 10.40, BLUE)
    footer(ax)
    save(fig, "02_matd3_plus_transformer.png")


def draw_hybrid():
    fig, ax = setup("MATD3 + PNN + Transformer", "两个模块在Actor中并行提取互补信息，再共同生成调度动作", OLIVE)
    standard_input(ax)
    arrow(ax, (3.00, 3.89), (4.10, 4.63), color=BLUE, connection="arc3,rad=-0.08")
    arrow(ax, (3.00, 3.89), (4.10, 3.12), color=GOLD, connection="arc3,rad=0.08")

    box(ax, 4.10, 4.15, 3.55, 1.02, face=BLUE_LIGHT, edge=BLUE, linewidth=1.6)
    label(ax, 5.88, 4.79, "Transformer", size=12.2, color=BLUE, weight="bold")
    label(ax, 5.88, 4.45, "学习整体时空关系", size=8.8, color=MUTED)

    box(ax, 4.10, 2.60, 3.55, 1.02, face=GOLD_LIGHT, edge=GOLD, linewidth=1.6)
    label(ax, 5.88, 3.24, "PNN", size=12.2, color=GOLD, weight="bold")
    label(ax, 5.88, 2.90, "学习局部特征交互", size=8.8, color=MUTED)

    arrow(ax, (7.65, 4.66), (8.45, 4.10), color=BLUE, connection="arc3,rad=0.08")
    arrow(ax, (7.65, 3.11), (8.45, 3.66), color=GOLD, connection="arc3,rad=-0.08")
    box(ax, 8.45, 3.20, 1.95, 1.38, face=OLIVE_LIGHT, edge=OLIVE, linewidth=1.8)
    label(ax, 9.43, 4.12, "特征融合", size=11.4, color=OLIVE, weight="bold")
    label(ax, 9.43, 3.64, "互补信息汇合", size=8.8, color=MUTED)

    standard_output(ax, 10.40, OLIVE)
    footer(ax)
    save(fig, "03_matd3_plus_pnn_transformer.png")


def main():
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "figure.facecolor": "white", "savefig.facecolor": "white"})
    draw_pnn()
    draw_transformer()
    draw_hybrid()


if __name__ == "__main__":
    main()
