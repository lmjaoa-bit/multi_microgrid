from __future__ import annotations

from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


OUTPUT = Path(__file__).resolve().parent / "current_environment_architecture.png"

INK = "#263238"
MUTED = "#5F6B72"
BLUE = "#2F6B9A"
BLUE_LIGHT = "#EAF2F8"
GOLD = "#D5A021"
GOLD_LIGHT = "#FBF4DF"
ORANGE = "#D97841"
ORANGE_LIGHT = "#FBEDE5"
OLIVE = "#7A8B45"
OLIVE_LIGHT = "#EFF3E5"
PINK = "#B75D75"
PINK_LIGHT = "#F8EAF0"
GREY = "#AAB2B7"
GREY_LIGHT = "#F3F5F6"
LINE = "#C9D0D4"


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str,
    edge: str,
    radius: float = 0.16,
    linewidth: float = 1.5,
    linestyle: str = "-",
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    width: float = 1.5,
    style: str = "-|>",
    connection: str = "arc3",
    zorder: int = 1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=12,
            linewidth=width,
            color=color,
            connectionstyle=connection,
            zorder=zorder,
            shrinkA=3,
            shrinkB=3,
        )
    )


def label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 10,
    color: str = INK,
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
    zorder: int = 5,
    linespacing: float = 1.35,
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        color=color,
        fontweight=weight,
        zorder=zorder,
        linespacing=linespacing,
    )


def microgrid_card(
    ax: plt.Axes,
    x: float,
    title: str,
    role: str,
    renewable: str,
    battery: str,
    tariff: str,
    color: str,
    light: str,
) -> None:
    y, width, height = 4.95, 3.95, 2.35
    box(ax, x, y, width, height, face="white", edge=color, radius=0.18, linewidth=1.8, zorder=3)
    box(ax, x, y + height - 0.52, width, 0.52, face=light, edge=color, radius=0.16, linewidth=0.0, zorder=3)
    label(ax, x + 0.20, y + height - 0.26, title, size=12, color=color, weight="bold", ha="left")
    label(ax, x + 0.20, y + 1.52, role, size=9.4, color=INK, weight="bold", ha="left")
    label(
        ax,
        x + 0.20,
        y + 0.83,
        f"能源：{renewable}\n储能：{battery}\n电价：{tariff}",
        size=8.9,
        color=MUTED,
        ha="left",
    )
    label(ax, x + 1.98, y + 0.20, "本地负荷优先 → 储能调度 → 形成余缺", size=8.4, color=color)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Microsoft YaHei",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 220,
        }
    )

    fig, ax = plt.subplots(figsize=(18, 10.5))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    label(ax, 9, 10.08, "多微网P2P强化学习环境组成与职责", size=22, weight="bold")
    label(
        ax,
        9,
        9.66,
        "协同运行 · 24小时自然日回合 · 1小时步长 · 储能时序调度与跨微网空间互济",
        size=11.5,
        color=MUTED,
    )

    # External controller and information preparation layer.
    box(ax, 0.55, 8.05, 3.65, 1.15, face=BLUE_LIGHT, edge=BLUE)
    label(ax, 0.82, 8.82, "数据与场景", size=12, color=BLUE, weight="bold", ha="left")
    label(ax, 0.82, 8.40, "训练/验证/测试日；负荷、光照、风速、电价", size=9.2, color=INK, ha="left")

    box(ax, 5.00, 8.05, 4.15, 1.15, face=GOLD_LIGHT, edge=GOLD)
    label(ax, 5.28, 8.82, "预测与状态构建", size=12, color="#9A7115", weight="bold", ha="left")
    label(ax, 5.28, 8.40, "本地状态 + 跨微网状态 + 未来24小时带误差点预测", size=9.2, color=INK, ha="left")

    box(ax, 10.00, 8.05, 4.55, 1.15, face=ORANGE_LIGHT, edge=ORANGE)
    label(ax, 10.28, 8.82, "多智能体控制器（环境接口）", size=12, color=ORANGE, weight="bold", ha="left")
    label(ax, 10.28, 8.40, "各智能体输出单一带符号储能调度意图", size=9.2, color=INK, ha="left")

    box(ax, 15.25, 8.05, 2.20, 1.15, face=GREY_LIGHT, edge=GREY)
    label(ax, 16.35, 8.79, "CTDE训练", size=11.5, color=INK, weight="bold")
    label(ax, 16.35, 8.38, "集中Critic\n分散Actor执行", size=8.8, color=MUTED)

    arrow(ax, (4.20, 8.63), (5.00, 8.63), color=BLUE)
    arrow(ax, (9.15, 8.63), (10.00, 8.63), color=GOLD)
    arrow(ax, (14.55, 8.63), (15.25, 8.63), color=ORANGE, style="<->")

    # Main environment boundary.
    box(
        ax,
        0.38,
        3.05,
        17.24,
        4.48,
        face="#FCFDFD",
        edge=GREY,
        radius=0.22,
        linewidth=1.4,
        linestyle="--",
        zorder=0,
    )
    label(ax, 0.68, 7.60, "MultiMicrogridEnv：负责物理约束、功率平衡、交易清算与共享奖励", size=10.5, color=MUTED, weight="bold", ha="left")

    microgrid_card(
        ax,
        0.78,
        "MG1｜日间光伏工业型",
        "日间负荷较高，光伏富余显著",
        "光伏 520 kW",
        "210 kWh，±65 kW",
        "固定购电价",
        BLUE,
        BLUE_LIGHT,
    )
    microgrid_card(
        ax,
        5.05,
        "MG2｜连续负荷风电型",
        "连续生产负荷，承担主要风电供给",
        "风电 900 kW",
        "290 kWh，±80 kW",
        "指数联动电价",
        GOLD,
        GOLD_LIGHT,
    )
    microgrid_card(
        ax,
        9.32,
        "MG3｜晚间高负荷光伏型",
        "晚间需求集中，本地光伏规模较小",
        "光伏 100 kW",
        "95 kWh，±34 kW",
        "峰谷分时电价",
        PINK,
        PINK_LIGHT,
    )

    box(ax, 13.66, 4.95, 3.48, 2.35, face=GREY_LIGHT, edge=GREY, linewidth=1.8)
    label(ax, 13.94, 6.94, "外部电网", size=12, color=INK, weight="bold", ha="left")
    label(
        ax,
        13.94,
        6.20,
        "负责P2P后剩余功率平衡\n各微网独立购电与上网\n进/出功率上限：200 kW",
        size=9.1,
        color=MUTED,
        ha="left",
    )
    label(ax, 15.40, 5.25, "禁止同小时购电后转售P2P", size=8.4, color=INK)

    # Actions from the policy and state feedback.
    for center in (2.76, 7.03, 11.30):
        arrow(ax, (12.15, 8.05), (center, 7.30), color=ORANGE, width=1.15, connection="arc3,rad=0.04")
    label(ax, 11.92, 7.73, "储能动作", size=8.6, color=ORANGE)
    arrow(ax, (6.65, 7.53), (7.05, 8.05), color=GOLD, width=1.2, connection="arc3,rad=-0.15")
    label(ax, 6.05, 7.76, "SOC/余缺/交易反馈", size=8.2, color="#9A7115")

    # P2P pool and grid exchange.
    box(ax, 3.72, 3.42, 8.18, 1.02, face=OLIVE_LIGHT, edge=OLIVE, linewidth=1.8)
    label(ax, 4.02, 4.10, "P2P合作能源池", size=12, color=OLIVE, weight="bold", ha="left")
    label(
        ax,
        4.02,
        3.70,
        "匹配可再生能源余量、已接受的储能共享供给与负荷/充电需求；双边容量50 kW，零网损、零交易费",
        size=8.8,
        color=INK,
        ha="left",
    )
    for center in (2.76, 7.03, 11.30):
        arrow(ax, (center, 4.95), (center, 4.44), color=OLIVE, width=1.35, style="<->")
    arrow(ax, (13.27, 6.10), (13.66, 6.10), color=GREY, width=1.5, style="<->")

    box(ax, 12.42, 3.42, 4.72, 1.02, face=BLUE_LIGHT, edge=BLUE, linewidth=1.8)
    label(ax, 12.72, 4.10, "成本核算与共享奖励", size=11.5, color=BLUE, weight="bold", ha="left")
    label(ax, 12.72, 3.70, "购电−售电 + 退化/弃能/失负荷成本 → 团队利润 → 共享奖励", size=8.7, color=INK, ha="left")
    arrow(ax, (11.90, 3.93), (12.42, 3.93), color=BLUE)

    # Evaluation-only layer.
    box(ax, 0.70, 0.62, 16.60, 1.83, face="white", edge=LINE, radius=0.20, linewidth=1.3, linestyle="--", zorder=1)
    label(ax, 0.98, 2.17, "测试后处理（不改变动作、物理调度、经验回放或训练）", size=10, color=MUTED, weight="bold", ha="left")

    box(ax, 1.05, 0.91, 4.45, 0.93, face=GREY_LIGHT, edge=GREY)
    label(ax, 1.30, 1.55, "24小时确定性测试", size=10.5, weight="bold", ha="left")
    label(ax, 1.30, 1.18, "保存逐时轨迹、成本、P2P与储能指标", size=8.7, color=MUTED, ha="left")

    box(ax, 6.75, 0.91, 4.45, 0.93, face=GOLD_LIGHT, edge=GOLD)
    label(ax, 7.00, 1.55, "七个非空联盟MILP", size=10.5, color="#9A7115", weight="bold", ha="left")
    label(ax, 7.00, 1.18, "计算单体、两两联盟和大联盟特征值", size=8.7, color=MUTED, ha="left")

    box(ax, 12.45, 0.91, 4.45, 0.93, face=PINK_LIGHT, edge=PINK)
    label(ax, 12.70, 1.55, "Equal与Shapley结算", size=10.5, color=PINK, weight="bold", ha="left")
    label(ax, 12.70, 1.18, "分配实际联盟收益，检查预算平衡与个体理性", size=8.7, color=MUTED, ha="left")

    arrow(ax, (5.50, 1.37), (6.75, 1.37), color=GREY)
    arrow(ax, (11.20, 1.37), (12.45, 1.37), color=GREY)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
