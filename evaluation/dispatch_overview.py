"""Seasonal dispatch evidence drawn only from persisted hourly trajectories."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

MICROGRIDS = ("mg1", "mg2", "mg3")
SEASONS = {"spring": (3, 4, 5), "summer": (6, 7, 8),
           "autumn": (9, 10, 11), "winter": (12, 1, 2)}
COLORS = {"res": "#e1ab3a", "discharge": "#d46745", "buy": "#5e9e9b",
          "grid": "#789ab8", "charge": "#7568a1", "sell": "#55a9c7",
          "ink": "#253448", "grey": "#9aa4ad", "green": "#629c73", "red": "#c85654"}


def seasonal_selection(hourly: pd.DataFrame) -> dict[str, Any]:
    """Closest real 24-hour profile to each seasonal centroid; no policy fields.

    Each channel is scaled by its population standard deviation over ALL saved
    test hours, preserving intraday timing and applying the same scale to every
    season. Constant channels contribute zero. Sorted dates break ties.
    """
    fields = [f"{mg}_{suffix}" for mg in MICROGRIDS
              for suffix in ("base_load_kw", "renewable_available_kw")]
    fields.append("mg2_t_out_c")
    if "p2p_price" in hourly:
        # The saved clearing price is exogenous (the midpoint of grid tariffs).
        fields.append("p2p_price")
    missing = set(["date", "hour", *fields]) - set(hourly.columns)
    if missing:
        raise ValueError(f"Seasonal selection requires saved fields: {sorted(missing)}")
    if hourly.empty or hourly.duplicated(["date", "hour"]).any():
        raise ValueError("Seasonal selection requires unique saved date/hour pairs")
    days, profiles = [], []
    for date, frame in hourly.groupby("date", sort=True):
        if sorted(frame.hour.tolist()) != list(range(24)):
            raise ValueError(f"Saved test day {date} must contain hours 0..23 exactly once")
        profile = frame.sort_values("hour")[fields].to_numpy(dtype=float)
        if not np.isfinite(profile).all():
            raise ValueError(f"Non-finite exogenous conditions in saved day {date}")
        days.append(str(date))
        profiles.append(profile)
    values = np.stack(profiles)
    scale = values.std(axis=(0, 1))
    scale[scale < 1e-12] = 1.0
    months = np.array([pd.Timestamp(date).month for date in days])
    selected, evidence = {}, {}
    for season, season_months in SEASONS.items():
        indices = np.flatnonzero(np.isin(months, season_months))
        if not len(indices):
            continue
        subset = values[indices]
        distance = np.square((subset - subset.mean(axis=0)) / scale).mean(axis=(1, 2))
        winner = int(np.argmin(distance))
        selected[season] = days[indices[winner]]
        evidence[season] = {"candidate_count": len(indices),
                            "mean_squared_standardized_distance": float(distance[winner])}
    return {"seasonal_days": selected, "selection_fields": fields,
            "selection_method": "real day nearest seasonal 24-hour centroid; channel scaling over all test hours; equal channel weights; earliest date breaks ties",
            "channel_scales": dict(zip(fields, scale.tolist())), "selection_evidence": evidence,
            "missing_seasons": [season for season in SEASONS if season not in selected]}


def select_seasonal_days(hourly: pd.DataFrame) -> dict[str, str]:
    return seasonal_selection(hourly)["seasonal_days"]


def _axis(ax: Any, title: str, ylabel: str) -> None:
    ax.set(xlim=(0, 24), xticks=np.arange(0, 25, 3), ylabel=ylabel)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.65)
    ax.set_axisbelow(True)


def _available(ax: Any, day: pd.DataFrame, fields: list[str]) -> bool:
    missing = [field for field in fields if field not in day]
    if not missing:
        return True
    ax.text(0.5, 0.5, "Saved fields unavailable:\n" + "\n".join(missing),
            transform=ax.transAxes, ha="center", va="center", fontsize=8)
    return False


def _stairs(ax: Any, values: Any, **kwargs: Any) -> None:
    ax.stairs(np.asarray(values, dtype=float), np.arange(25), **kwargs)


def _dispatch(subfig: Any, day: pd.DataFrame) -> None:
    subfig.suptitle("1 | Three-microgrid power balance", fontsize=13, fontweight="bold")
    axes = subfig.subplots(3, 1, sharex=True)
    components = [("renewable_used_kw", "Renewable used", "res", 1),
                  ("battery_discharge_kw", "ESS discharge", "discharge", 1),
                  ("p2p_buy_kw", "P2P import", "buy", 1),
                  ("grid_import_kw", "Grid import", "grid", 1),
                  ("battery_charge_kw", "ESS charge", "charge", -1),
                  ("p2p_sell_kw", "P2P export", "sell", -1),
                  ("grid_export_kw", "Grid export", "grey", -1)]
    for mg, ax in zip(MICROGRIDS, axes):
        _axis(ax, f"{mg.upper()} dispatch", "Power (kW)")
        fields = [f"{mg}_{item[0]}" for item in components] + [f"{mg}_actual_load_kw"]
        if not _available(ax, day, fields):
            continue
        positive, negative = np.zeros(24), np.zeros(24)
        for field, label, color, sign in components:
            values = day[f"{mg}_{field}"].to_numpy(dtype=float) * sign
            bottom = positive if sign > 0 else negative
            ax.bar(day.hour + 0.5, values, bottom=bottom, width=0.86,
                   color=COLORS[color], label=label)
            bottom += values
        _stairs(ax, day[f"{mg}_actual_load_kw"], color=COLORS["ink"], linewidth=1.4, label="Actual load")
        _stairs(ax, day[f"{mg}_renewable_available_kw"], color="#b88020", linestyle="--", label="Renewable available")
        ax.axhline(0, color=COLORS["grey"], linewidth=0.7)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(ncols=3, fontsize=7, loc="upper left")
    axes[-1].set_xlabel("Hour interval (bars centered at h + 0.5)", fontsize=9)


def _storage(subfig: Any, day: pd.DataFrame, microgrids: Mapping[str, Any], env: Mapping[str, Any]) -> None:
    subfig.suptitle("2 | Storage, tariff and net-load context", fontsize=13, fontweight="bold")
    gs = subfig.add_gridspec(4, 2, height_ratios=(0.6, 1, 1, 1))
    tariff = subfig.add_subplot(gs[0, :])
    _axis(tariff, "Shared grid purchase tariff", "Price / kWh")
    if "p2p_price" in day and "sell_ratio" in env.get("price", {}):
        inherited = all(m.get("tariff", {}).get("mode", "inherited") == "inherited"
                        for m in microgrids.values())
        if inherited:
            price = day.p2p_price * 2 / (1 + float(env["price"]["sell_ratio"]))
            _stairs(tariff, price, color=COLORS["ink"])
        else:
            tariff.text(0.5, 0.5, "Individual tariffs; no common grid price", transform=tariff.transAxes, ha="center")
    else:
        tariff.text(0.5, 0.5, "Saved common tariff unavailable", transform=tariff.transAxes, ha="center")
    for index, mg in enumerate(MICROGRIDS):
        ax = subfig.add_subplot(gs[index + 1, 0])
        _axis(ax, f"{mg.upper()} ESS power and SOC", "ESS (kW)")
        fields = [f"{mg}_battery_charge_kw", f"{mg}_battery_discharge_kw", f"{mg}_soc"]
        if _available(ax, day, fields):
            ax.bar(day.hour + 0.5, day[fields[1]], color=COLORS["discharge"], label="Discharge (+)")
            ax.bar(day.hour + 0.5, -day[fields[0]], color=COLORS["charge"], label="Charge (-)")
            battery = microgrids.get(mg, {}).get("battery", {})
            if "initial_soc" in battery:
                twin = ax.twinx()
                twin.plot(np.arange(25), np.r_[battery["initial_soc"], day[fields[2]]] * 100,
                          color=COLORS["green"], linewidth=1.5, marker=".", label="SOC")
                twin.set(ylim=(0, 100), ylabel="SOC (%)")
                twin.tick_params(labelsize=8, colors=COLORS["green"])
                for bound in ("soc_min", "soc_max"):
                    if bound in battery:
                        twin.axhline(battery[bound] * 100, color=COLORS["green"], linestyle=":", linewidth=0.7)
            if index == 0:
                ax.legend(ncols=2, fontsize=7)
        context = subfig.add_subplot(gs[index + 1, 1])
        _axis(context, f"{mg.upper()} local and peer net load", "Net load (kW)")
        fields = [f"{m}_actual_load_kw" for m in MICROGRIDS]
        if _available(context, day, fields):
            local = day[f"{mg}_actual_load_kw"] - day[f"{mg}_renewable_available_kw"]
            peers = sum(day[f"{m}_actual_load_kw"] - day[f"{m}_renewable_available_kw"]
                        for m in MICROGRIDS if m != mg)
            _stairs(context, local, color=COLORS["ink"], label="Local")
            _stairs(context, peers, color=COLORS["grey"], linestyle="--", label="Other two MGs")
            context.axhline(0, color=COLORS["grey"], linewidth=0.7)
            context.legend(ncols=2, fontsize=7)
        if index == 2:
            ax.set_xlabel("Hour / state boundary", fontsize=9)
            context.set_xlabel("Hour interval", fontsize=9)


def _flexible(subfig: Any, day: pd.DataFrame, microgrids: Mapping[str, Any], reference: pd.DataFrame | None) -> None:
    subfig.suptitle("3 | Flexible-load power, states and service", fontsize=13, fontweight="bold")
    axes = subfig.subplots(4, 2)
    powers = ("mg1_mill_power_kw", "mg2_hvac_effective_signed_power_kw", "mg3_ev_effective_power_kw")
    for index, (mg, field) in enumerate(zip(MICROGRIDS, powers)):
        ax = axes[index, 0]
        title = f"{mg.upper()} flexible power"
        if index == 1:
            title += " (+ heating / - cooling)"
        _axis(ax, title, "Power (kW)")
        if _available(ax, day, [field]):
            values = day[field].to_numpy()
            ax.bar(day.hour + 0.5, values, color=np.where(values >= 0, COLORS["grid"], COLORS["discharge"]), label="Policy")
            if reference is not None and field in reference:
                _stairs(ax, reference[field], color=COLORS["grey"], linestyle="--", label="Rule reference")
            if index == 2 and "mg3_ev_mapping_high_kw" in day:
                _stairs(ax, day.mg3_ev_mapping_high_kw, color=COLORS["green"], linestyle=":", label="Available charging limit")
            ax.axhline(0, color=COLORS["grey"], linewidth=0.7)
            ax.legend(fontsize=7, ncols=2)
    ax = axes[0, 1]
    _axis(ax, "MG1 silo inventory", "Inventory (t)")
    if _available(ax, day, ["mg1_silo_level_t"]):
        dr = microgrids.get("mg1", {}).get("dr", {})
        if "silo_min_t" in dr and "silo_max_t" in dr:
            ax.axhspan(dr["silo_min_t"], dr["silo_max_t"], color=COLORS["green"], alpha=0.15)
        initial = day["mg1_mill_silo_before_t"].iloc[0] if "mg1_mill_silo_before_t" in day else dr.get("silo_initial_t", np.nan)
        ax.plot(np.arange(25), np.r_[initial, day.mg1_silo_level_t], color=COLORS["green"], marker=".")
        if "raw_mill_unserved_t" in day:
            ax.text(0.02, 0.04, f"Downstream unserved: {day.raw_mill_unserved_t.sum():.3g} t", transform=ax.transAxes, fontsize=8)
    ax = axes[1, 1]
    _axis(ax, "MG2 indoor temperature and comfort bounds", "Temperature (C)")
    if _available(ax, day, ["mg2_t_air_c", "mg2_result_comfort_low_c", "mg2_result_comfort_high_c"]):
        x = day.hour + 1
        ax.fill_between(x, day.mg2_result_comfort_low_c, day.mg2_result_comfort_high_c,
                        color=COLORS["green"], alpha=0.15, step="mid", label="Comfort band")
        ax.plot(x, day.mg2_t_air_c, color=COLORS["grid"], label="Indoor")
        ax.plot(day.hour + 0.5, day.mg2_t_out_c, color=COLORS["grey"], linestyle="--", label="Outdoor (interval)")
        bad = (day.mg2_t_air_c < day.mg2_result_comfort_low_c - 1e-8) | (day.mg2_t_air_c > day.mg2_result_comfort_high_c + 1e-8)
        ax.scatter(x[bad], day.mg2_t_air_c[bad], color=COLORS["red"], s=14, label="Violation")
        ax.legend(fontsize=7, ncols=2)
    ax = axes[2, 1]
    _axis(ax, "MG3 connected EV cohorts; gaps when absent", "EV SOC (%)")
    if _available(ax, day, ["mg3_ev_soc", "mg3_ev_hours_to_departure"]):
        values = day.mg3_ev_soc.where(day.mg3_ev_hours_to_departure > 0) * 100
        ax.plot(day.hour + 1, values, color=COLORS["green"], marker=".")
        target = microgrids.get("mg3", {}).get("dr", {}).get("target_soc")
        if target is not None:
            ax.axhline(target * 100, color=COLORS["grey"], linestyle="--")
        ax.set_ylim(0, 100)
    ax = axes[3, 0]
    _axis(ax, "MG3 cumulative EV departure service", "Cohorts")
    if _available(ax, day, ["mg3_ev_departure_count", "mg3_ev_completed_group_count"]):
        for field, label, color in (("mg3_ev_departure_count", "Departed", "grey"),
                                    ("mg3_ev_completed_group_count", "Completed", "green")):
            ax.step(np.arange(25), np.r_[0, day[field]], where="post", color=COLORS[color], label=label)
        ax.legend(fontsize=7)
    ax = axes[3, 1]
    _axis(ax, "MG3 incremental departure shortfall", "Shortfall (kWh)")
    if _available(ax, day, ["raw_ev_departure_shortfall_kwh"]):
        ax.bar(day.hour + 1, day.raw_ev_departure_shortfall_kwh, color=COLORS["red"], width=0.5)
        ax.set_ylim(0, max(0.01, float(day.raw_ev_departure_shortfall_kwh.max()) * 1.3))
        ax.text(0.02, 0.85, f"Daily total: {day.raw_ev_departure_shortfall_kwh.sum():.3g} kWh", transform=ax.transAxes, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Hour / state boundary", fontsize=9)


def dispatch_overview_figure(day: pd.DataFrame, env: Mapping[str, Any], label: str,
                             date: str, season: str, reference: pd.DataFrame | None = None) -> Any:
    import matplotlib.pyplot as plt

    day = day.sort_values("hour").reset_index(drop=True)
    if reference is not None:
        reference = reference.sort_values("hour").reset_index(drop=True)
        if reference.hour.tolist() != list(range(24)):
            raise ValueError(f"Saved DR reference {date} must contain exactly 24 hours")
        if "date" in reference and set(reference.date.astype(str)) != {date}:
            raise ValueError(f"Saved DR reference must match representative date {date}")
    microgrids = {m["id"]: m for m in env.get("microgrids", [])}
    fig = plt.figure(figsize=(24, 18), layout="constrained")
    fig.suptitle(f"{label} dispatch overview | {season.title()} | {date}", fontsize=19, fontweight="bold")
    columns = fig.subfigures(1, 2, width_ratios=(0.96, 1.04), wspace=0.04)
    left = columns[0].subfigures(2, 1, height_ratios=(1, 1.05), hspace=0.04)
    right = columns[1].subfigures(2, 1, height_ratios=(1.6, 0.65), hspace=0.04)
    _dispatch(left[0], day)
    _storage(left[1], day, microgrids, env)
    _flexible(right[0], day, microgrids, reference)
    right[1].suptitle("4 | Three-microgrid net P2P power", fontsize=13, fontweight="bold")
    ax = right[1].subplots()
    _axis(ax, "Positive = net import; negative = net export", "Net P2P (kW)")
    fields = [f"{mg}_{suffix}" for mg in MICROGRIDS for suffix in ("p2p_buy_kw", "p2p_sell_kw")]
    if _available(ax, day, fields):
        for mg, color, marker in (("mg3", "#80baff", "s"), ("mg1", "#eb79b9", "D"), ("mg2", "#777b80", "o")):
            ax.plot(day.hour + 0.5, day[f"{mg}_p2p_buy_kw"] - day[f"{mg}_p2p_sell_kw"],
                    color=color, marker=marker, markersize=4, label=mg.upper())
        ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
        ax.legend(ncols=3, fontsize=9)
    ax.set_xlabel("Hour interval", fontsize=9)
    fig.supxlabel("Power: hourly intervals. SOC, silo, indoor temperature and departures: end-of-hour states. One real test day per season.", fontsize=10)
    return fig
