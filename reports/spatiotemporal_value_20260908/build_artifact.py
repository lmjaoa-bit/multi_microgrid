from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    summary = json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
    sensitivity = json.loads(
        (OUTPUT_DIR / "sensitivity_summary.json").read_text(encoding="utf-8")
    )
    daily = pd.read_csv(OUTPUT_DIR / "daily_counterfactuals.csv")
    daily["month"] = pd.to_datetime(daily["day"]).dt.month
    monthly = (
        daily.groupby("month", as_index=False)[
            ["temporal_shapley_gain", "spatial_shapley_gain", "interaction_gain"]
        ]
        .mean()
        .round(6)
    )
    monthly_rows = []
    for row in monthly.to_dict(orient="records"):
        monthly_rows.extend(
            [
                {
                    "month": int(row["month"]),
                    "factor": "时间调度",
                    "gain": row["temporal_shapley_gain"],
                    "interaction_gain": row["interaction_gain"],
                },
                {
                    "month": int(row["month"]),
                    "factor": "空间互济",
                    "gain": row["spatial_shapley_gain"],
                    "interaction_gain": row["interaction_gain"],
                },
            ]
        )

    mean = summary["mean"]
    current = summary["current_policy"]
    ci = summary["bootstrap_95_ci"]
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    factor_rows = [
        {
            "factor": "时间调度",
            "mean_gain": mean["temporal_shapley_gain"],
            "share": summary["factor_shares"]["temporal"],
            "ci_low": ci["temporal_shapley_gain"][0],
            "ci_high": ci["temporal_shapley_gain"][1],
            "days": summary["days"],
        },
        {
            "factor": "空间互济",
            "mean_gain": mean["spatial_shapley_gain"],
            "share": summary["factor_shares"]["spatial"],
            "ci_low": ci["spatial_shapley_gain"][0],
            "ci_high": ci["spatial_shapley_gain"][1],
            "days": summary["days"],
        },
    ]
    scenario_rows = [
        {"order": 1, "scenario": "无储能、无P2P", "mean_cost": mean["cost_no_storage_no_p2p"]},
        {"order": 2, "scenario": "仅时间调度", "mean_cost": mean["cost_temporal_only"]},
        {"order": 3, "scenario": "仅空间互济", "mean_cost": mean["cost_spatial_only"]},
        {"order": 4, "scenario": "当前MATD3", "mean_cost": current["mean_cost"]},
        {"order": 5, "scenario": "时空联合MILP", "mean_cost": mean["cost_spatiotemporal"]},
    ]
    headroom_rows = [
        {
            "metric": "总灵活性价值捕获率",
            "current": current["total_flexibility_value_capture"],
            "oracle": 1.0,
            "unit": "比例",
        },
        {
            "metric": "储能吞吐量",
            "current": current["battery_throughput_kwh"],
            "oracle": mean["full_battery_throughput_kwh"],
            "unit": "kWh/日",
        },
        {
            "metric": "P2P交易量",
            "current": current["p2p_energy_kwh"],
            "oracle": mean["full_p2p_energy_kwh"],
            "unit": "kWh/日",
        },
        {
            "metric": "弃电量",
            "current": current["curtailment_kwh"],
            "oracle": mean["full_curtailment_kwh"],
            "unit": "kWh/日",
        },
    ]

    database_path = OUTPUT_DIR / "report_sources.sqlite"
    connection = sqlite3.connect(database_path)
    pd.DataFrame(factor_rows).to_sql("factor_values", connection, if_exists="replace", index=False)
    pd.DataFrame(monthly_rows).to_sql("monthly_values", connection, if_exists="replace", index=False)
    pd.DataFrame(scenario_rows).to_sql("scenario_costs", connection, if_exists="replace", index=False)
    pd.DataFrame(headroom_rows).to_sql("headroom", connection, if_exists="replace", index=False)
    daily.to_sql("daily_counterfactuals", connection, if_exists="replace", index=False)

    factor_sql = "SELECT factor, mean_gain, share, ci_low, ci_high, days FROM factor_values ORDER BY mean_gain"
    monthly_sql = "SELECT month, factor, gain, interaction_gain FROM monthly_values ORDER BY month, factor"
    scenario_sql = 'SELECT "order", scenario, mean_cost FROM scenario_costs ORDER BY "order"'
    headroom_sql = "SELECT metric, current, oracle, unit FROM headroom ORDER BY metric"

    def query_rows(sql: str) -> list[dict]:
        return pd.read_sql_query(sql, connection).to_dict(orient="records")

    factor_rows = query_rows(factor_sql)
    monthly_rows = query_rows(monthly_sql)
    scenario_rows = query_rows(scenario_sql)
    headroom_rows = query_rows(headroom_sql)
    connection.close()

    counterfactual_source = {
        "id": "counterfactual",
        "label": "当前配置的55日MILP时空反事实",
        "path": "reports/spatiotemporal_value_20260908/summary.json",
        "query": {
            "engine": "Python / HiGHS / pandas",
            "language": "python",
            "description": "在同一批测试日上求解无灵活性、仅本地储能、仅同小时P2P和时空联合四种调度，并用二因素Shapley分解时间与空间价值。",
            "tables_used": [
                "configs/cooperative.yaml",
                "data/spatial_market_2018_test.csv",
                "reports/spatiotemporal_value_20260908/daily_counterfactuals.csv",
                "reports/spatiotemporal_value_20260908/stored_energy_peer_export_sensitivity.csv",
            ],
            "filters": [
                "55个固定测试自然日；每日电池从soc_min开始；terminal_soc.mode=none。",
                "空间互济受50 kW两两链路容量限制；P2P手续费和损耗均为0。",
            ],
            "metric_definitions": [
                "时间独立价值=C00-C10；空间独立价值=C00-C01；C00为无储能无P2P成本。",
                "时间Shapley=0.5[(C00-C10)+(C01-C11)]；空间Shapley=0.5[(C00-C01)+(C10-C11)]。",
                "储能跨空间通道价值=限制P2P只能来自当时可再生能源时的联合成本-不限制来源时的联合成本。",
                "置信区间为逐日独立重采样20000次、随机种子42；不衡量训练随机性。",
            ],
        },
    }
    current_source = {
        "id": "current_run",
        "label": "当前MATD3正式测试保存结果",
        "path": "outputs/matd3_dispatch/latest_run.json",
        "query": {
            "engine": "Saved experiment outputs",
            "language": "json/csv",
            "description": "读取latest_run指向的正式测试摘要；使用验证集选出的best_model.pt在相同55日确定性测试集上的结果。",
            "tables_used": [
                "outputs/matd3_dispatch/latest_run.json",
                "outputs/matd3_dispatch/runs/MATD3_2026-09-08_15-46-18_seed42/test_summary.json",
                "outputs/matd3_dispatch/runs/MATD3_2026-09-08_15-46-18_seed42/test_daily_metrics.csv",
            ],
            "filters": [
                "run_id=MATD3_2026-09-08_15-46-18_seed42；55个测试日；deterministic policy/environment。"
            ],
            "metric_definitions": [
                "成本=-team_profit；总灵活性价值捕获率=(C00-MATD3成本)/(C00-C11)。",
                "储能吞吐量=所有微网充电量与放电量之和；P2P交易量=所有方向成交电量之和。",
            ],
        },
    }
    combined_source = {
        "id": "combined",
        "label": "反事实MILP与当前MATD3对照",
        "path": "reports/spatiotemporal_value_20260908/summary.json",
        "query": {
            "engine": "Python / saved experiment outputs",
            "language": "python/json/csv",
            "description": "将同一物理配置和测试日期上的MILP反事实与当前MATD3正式测试结果按日均口径对照。",
            "tables_used": counterfactual_source["query"]["tables_used"]
            + current_source["query"]["tables_used"],
            "filters": ["相同55个测试日期；同一当前物理配置口径。"],
            "metric_definitions": ["算法缺口=MATD3日均成本-时空联合MILP日均成本。"],
        },
    }

    def sqlite_source(source_id: str, label: str, sql: str, description: str) -> dict:
        return {
            "id": source_id,
            "label": label,
            "path": "reports/spatiotemporal_value_20260908/report_sources.sqlite",
            "query": {
                "engine": "SQLite",
                "language": "sql",
                "sql": sql,
                "description": description,
                "tables_used": [
                    f"reports.spatiotemporal_value_20260908.{source_id.removesuffix('_query')}"
                ],
                "filters": ["55个固定测试自然日；当前configs/cooperative.yaml。"],
                "metric_definitions": counterfactual_source["query"]["metric_definitions"],
            },
        }

    factor_query_source = sqlite_source(
        "factor_query", "时间与空间Shapley贡献查询", factor_sql,
        "读取二因素Shapley日均贡献、占比和逐日重采样区间。",
    )
    monthly_query_source = sqlite_source(
        "monthly_query", "月度时间与空间贡献查询", monthly_sql,
        "读取按月份聚合的时间与空间Shapley贡献。",
    )
    scenario_query_source = sqlite_source(
        "scenario_query", "调度场景日均成本查询", scenario_sql,
        "读取四个MILP反事实和当前MATD3的日均成本。",
    )
    headroom_query_source = sqlite_source(
        "headroom_query", "当前策略与联合MILP资源利用查询", headroom_sql,
        "读取当前MATD3与联合MILP的储能、P2P、弃电和价值捕获对照。",
    )
    all_sources = [
        counterfactual_source,
        current_source,
        combined_source,
        factor_query_source,
        monthly_query_source,
        scenario_query_source,
        headroom_query_source,
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "多微网储能时空调度价值诊断",
        "generatedAt": generated_at,
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# 多微网储能时空调度价值诊断"},
            {
                "id": "summary",
                "type": "markdown",
                "body": (
                    "## 技术摘要：当前系统应优先挖掘空间协调，但论文不能把全部空间收益称为储能时空收益\n\n"
                    "在当前55个测试日和完整MILP信息下，空间互济的二因素Shapley价值为 **486.12/日**，时间调度为 **240.29/日**，空间约为时间的 **2.02倍**，分别占总灵活性价值的 **66.9% / 33.1%**。空间贡献在46/55天更大；空间减时间的日均差为245.82，逐日重采样95%区间为 **[187.31, 307.18]**。\n\n"
                    "但把P2P限制为只能发送当时的可再生能源后，允许“储存后再送给其他微网”的额外价值只有 **91.33/日**，占全部灵活性价值 **12.6%**。所以最大的成本池是同小时跨微网互济，储能跨时空通道有价值但不是主导项。\n\n"
                    "当前MATD3成本为 **990.80/日**，联合MILP为 **757.41/日**，还差 **233.40/日**；已捕获理论总灵活性价值的 **67.9%**。"
                ),
            },
            {
                "id": "factor_text",
                "type": "markdown",
                "body": (
                    "## 空间互济贡献约为时间调度的两倍\n\n"
                    "无储能无P2P时成本为1483.82/日；只启用各微网本地储能可节省240.12，只启用同小时P2P可节省485.95。用二因素Shapley平均两种加入顺序后，时间和空间贡献分别为240.29与486.12，合计严格等于时空联合节省726.41。\n\n"
                    "这说明在当前资源配置下，异构微网之间的同时性互补——MG1日间光伏、MG2风电、MG3晚间负荷及差异化电价——比单个微网内部的跨时段套利更值钱。"
                ),
                "sourceId": "factor_query",
            },
            {"id": "factor_chart_block", "type": "chart", "chartId": "factor_chart"},
            {
                "id": "monthly_text",
                "type": "markdown",
                "body": (
                    "## 空间优势覆盖大多数日期，但存在时间主导的低空间互补日\n\n"
                    "空间贡献在83.6%的测试日高于时间贡献。按月份聚合，只有11月时间贡献略高；在1月部分低可再生能源日，时间调度也占优。因此不宜将策略永久偏向P2P，应该让策略根据同期余缺和未来价格动态切换。"
                ),
                "sourceId": "monthly_query",
            },
            {"id": "monthly_chart_block", "type": "chart", "chartId": "monthly_chart"},
            {
                "id": "cost_text",
                "type": "markdown",
                "body": (
                    "## 当前MATD3接近“仅空间最优”，但离时空联合最优仍有明显距离\n\n"
                    "仅空间MILP成本为997.87/日，当前MATD3为990.80，二者数值接近；这不代表MATD3已经学会空间最优，因为MATD3还使用了储能，而两种策略的可行域不同。它只说明当前策略的综合效果大致停留在“把主要空间价值拿到、但没有充分叠加时间价值”的水平。\n\n"
                    "当前策略到联合MILP仍有233.40/日缺口。单靠新增储能跨微网送电通道，其理论总价值只有91.33/日，最多覆盖该缺口的39.1%；其余改进必须来自更好的本地储能时序、同小时P2P匹配和二者的联合控制。"
                ),
            },
            {"id": "cost_chart_block", "type": "chart", "chartId": "cost_chart"},
            {
                "id": "channel_text",
                "type": "markdown",
                "body": (
                    "## 储能跨微网通道有稳定价值，但不是最大价值来源\n\n"
                    "限制储能电量向同行输出后，联合最优成本平均增加91.33/日，95%重采样区间为[80.18, 102.59]，55天全部为正。这个量更接近你所说的“储能在时间和空间上的联合调度价值”。\n\n"
                    "另一方面，时间×空间的二因素交互项均值只有0.34/日、95%区间跨零；它在不同日期正负抵消。这表示总体价值近似可加，但不表示每一天都没有时空耦合。更合适的算法方向是识别哪些日期和时段值得保留电量给同行，而不是无条件增加ESS报量。"
                ),
                "sourceId": "counterfactual",
            },
            {
                "id": "current_text",
                "type": "markdown",
                "body": (
                    "## 当前策略的空间利用率低于时间利用率，但成交量不是价值归因\n\n"
                    "当前MATD3的储能吞吐量为980.72 kWh/日，相当于联合MILP的63.0%；P2P交易量为861.20 kWh/日，相当于联合MILP的53.5%。弃电仍为144.53 kWh/日，比联合MILP多122.26；ESS报量能量成交率只有41.85%。\n\n"
                    "这些指标共同指向空间匹配和储能预留仍有明显不足，但交易量本身不是收益。要把233.40/日的算法缺口严格分成时间和空间，必须对同一已训练策略做可行域消融，或在完全相同训练预算下分别训练local-storage-only、renewable-P2P-only和ESS-P2P三个对照。"
                ),
                "sourceId": "current_run",
            },
            {"id": "headroom_table_block", "type": "table", "tableId": "headroom_table"},
            {
                "id": "definitions",
                "type": "markdown",
                "body": (
                    "## 四个反事实把时间、空间与联合价值分开\n\n"
                    "设C00为无储能无P2P成本，C10为仅本地储能，C01为仅同小时P2P，C11为时空联合。时间Shapley为0.5[(C00−C10)+(C01−C11)]，空间Shapley为0.5[(C00−C01)+(C10−C11)]。该分解避免先后顺序偏差，并保证两项之和等于C00−C11。\n\n"
                    "每个场景都使用相同负荷、可再生能源、电价、SOC初值和55个测试日期；所有MILP均达到最优状态。储能关闭通过把充放电功率上限设为0实现。"
                ),
                "sourceId": "counterfactual",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## 结论受当前实验设定约束\n\n"
                    "当前空间价值包含同小时能量互济和不同微网电价带来的系统成本差，不等同于地理距离价值。模型未加入配电潮流、距离相关线损或P2P手续费，链路只受50 kW容量限制；这些因素通常会压低空间价值。\n\n"
                    "MILP具有24小时完整未来信息，而MATD3只有有限预测；两者差距同时包含信息优势与策略优化差距。每天SOC从下限开始、日末无目标和残值，也会影响时间价值。55天是固定测试日期，重采样区间描述日期敏感性，不描述训练种子不确定性。"
                ),
                "sourceId": "counterfactual",
            },
            {
                "id": "recommendations",
                "type": "markdown",
                "body": (
                    "## 下一步应把“空间协调”作为主线，把“储能跨时空”作为受控增量\n\n"
                    "1. **成本优先级：先改空间匹配和储能预留。** 让策略显式决定本地使用、留待未来、送给同行三种去向，并重点处理MG2较低的ESS成交率。\n"
                    "2. **论文贡献要分层。** 主结果报告同小时空间互济的486.12/日价值；另以91.33/日作为储能跨时空通道的严格增量，避免把全部P2P收益归功于储能。\n"
                    "3. **做同架构消融。** 固定数据、网络规模、训练步数和随机种子，比较无协同、local-storage-only、renewable-P2P-only、ESS-P2P及full五组。\n"
                    "4. **如果要强化时空论文主张，应另设跨算法物理场景。** 加入节点线损/拥塞、异构储能位置、多日SOC、交易费和特定的跨时段异地余缺窗口；不要只改某个算法的环境。"
                ),
            },
            {
                "id": "questions",
                "type": "markdown",
                "body": (
                    "## 仍需回答的问题\n\n"
                    "储能跨微网送电的91.33/日价值主要来自哪一对微网、哪些小时和哪类天气？在加入距离线损与交易费后，空间仍是否约为时间的两倍？同一策略在禁用ESS-P2P后损失多少实际收益？这三项会决定最终论文应强调空间市场设计，还是强调储能跨时空控制。"
                ),
            },
        ],
        "charts": [
            {
                "id": "factor_chart",
                "title": "时间与空间灵活性的日均成本贡献",
                "subtitle": "55个测试日；二因素Shapley分解，数值越大表示成本下降越多",
                "type": "bar",
                "dataset": "factor_values",
                "sourceId": "factor_query",
                "source": factor_query_source,
                "encodings": {"x": {"field": "factor"}, "y": {"field": "mean_gain"}},
                "layout": "full",
                "valueFormat": "number",
                "palette": {"kind": "sequential", "name": "blue"},
                "labels": {"values": "all"},
            },
            {
                "id": "monthly_chart",
                "title": "各月时间与空间贡献",
                "subtitle": "每月测试日的平均Shapley成本贡献；项目计价单位/日",
                "type": "line",
                "dataset": "monthly_values",
                "sourceId": "monthly_query",
                "source": monthly_query_source,
                "encodings": {
                    "x": {"field": "month"},
                    "y": {"field": "gain"},
                    "color": {"field": "factor"},
                },
                "layout": "full",
                "valueFormat": "number",
                "palette": {"kind": "categorical", "colors": ["#3b6fb6", "#c69235"]},
            },
            {
                "id": "cost_chart",
                "title": "五种调度口径的日均能源成本",
                "subtitle": "相同55个测试日期；成本越低越好",
                "type": "bar",
                "dataset": "scenario_costs",
                "sourceId": "scenario_query",
                "source": scenario_query_source,
                "encodings": {"x": {"field": "scenario"}, "y": {"field": "mean_cost"}},
                "layout": "full",
                "valueFormat": "number",
                "palette": {"kind": "sequential", "name": "blue"},
                "labels": {"values": "all"},
            },
        ],
        "tables": [
            {
                "id": "headroom_table",
                "title": "当前MATD3与联合MILP的资源利用对照",
                "subtitle": "交易量和吞吐量用于定位，不直接代表货币价值",
                "dataset": "headroom",
                "sourceId": "headroom_query",
                "source": headroom_query_source,
                "defaultSort": {"field": "metric", "direction": "asc"},
                "columns": [
                    {"field": "metric", "label": "指标", "type": "text"},
                    {"field": "current", "label": "当前MATD3", "format": "number"},
                    {"field": "oracle", "label": "联合MILP", "format": "number"},
                    {"field": "unit", "label": "单位", "type": "text"},
                ],
            }
        ],
        "sources": all_sources,
    }
    snapshot = {
        "version": 1,
        "status": "ready",
        "generatedAt": generated_at,
        "datasets": {
            "factor_values": factor_rows,
            "monthly_values": monthly_rows,
            "scenario_costs": scenario_rows,
            "headroom": headroom_rows,
            "daily_counterfactuals": daily.round(8).to_dict(orient="records"),
            "stored_energy_peer_export": [
                {
                    "mean_value_per_day": sensitivity["mean_value_per_day"],
                    "median_value_per_day": sensitivity["median_value_per_day"],
                    "ci_low": sensitivity["bootstrap_mean_95_ci"][0],
                    "ci_high": sensitivity["bootstrap_mean_95_ci"][1],
                    "positive_day_rate": sensitivity["positive_day_rate"],
                    "share_of_total_gain": sensitivity[
                        "share_of_total_spatiotemporal_gain"
                    ],
                }
            ],
        },
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": all_sources,
    }
    (OUTPUT_DIR / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
