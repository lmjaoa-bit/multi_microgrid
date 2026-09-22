from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.check_p2p_value_gate import evaluate_p2p_value_gate


def write_saved_run(
    root: Path,
    *,
    slug: str,
    acceptance: float,
    offer: float,
    accepted: float,
    total_p2p: float,
    strict_feasible: float = 1.0,
    unmet: float = 0.0,
    comfort_pct: float = 99.6,
    hvac_joint_pct: float = 50.0,
    hvac_late_power_p95: float = 5.0,
    mill_terminal_pct: float = 100.0,
    mill_production_pct: float = 100.0,
    ev_pct: float = 100.0,
    mg2_projection_pct: float = 0.0,
    mg3_projection_pct: float = 0.0,
    cost: float = 870.0,
    renewable: float = 0.97,
    peak_hourly: float = 40.0,
    checkpoint: str = "checkpoints/best_model.pt",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_metadata.json").write_text(
        json.dumps({"algorithm_slug": slug, "run_id": root.name}), encoding="utf-8"
    )
    (root / "training_summary.json").write_text(
        json.dumps({
            "checkpoint_selection": {
                "selection_mode": "constrained_profit",
                "constraint_eligible": True,
                "best_checkpoint": checkpoint,
            }
        }),
        encoding="utf-8",
    )
    per_agent_offer = {"mg1": offer / 3, "mg2": offer / 3, "mg3": offer / 3}
    per_agent_accepted = {
        "mg1": accepted / 3,
        "mg2": accepted / 3,
        "mg3": accepted / 3,
    }
    (root / "test_summary.json").write_text(
        json.dumps({
            "checkpoint_used": checkpoint,
            "strict_feasible_day_rate": strict_feasible,
            "unmet_energy_mean_kwh": unmet,
            "renewable_utilization_rate": renewable,
            "p2p_total_energy_kwh_mean": total_p2p,
            "algorithm": {
                "p2p": {
                    "total_energy_mean_kwh": total_p2p,
                    "ess_offer": {
                        "offer_mean_kwh": per_agent_offer,
                        "accepted_mean_kwh": per_agent_accepted,
                        "energy_acceptance_rate": acceptance,
                    },
                }
            },
        }),
        encoding="utf-8",
    )
    metrics = {
        "hvac_terminal_joint_success_rate": hvac_joint_pct,
        "hvac_late_power_p95_kw": hvac_late_power_p95,
        "hvac_occupied_comfort_rate": comfort_pct,
        "mill_terminal_success_rate": mill_terminal_pct,
        "mill_production_completion_rate": mill_production_pct,
        "ev_departure_target_success_rate": ev_pct,
        "mg2_dr_projection_rate": mg2_projection_pct,
        "mg3_dr_projection_rate": mg3_projection_pct,
        "total_operating_cost_day": cost,
    }
    (root / "paper_metrics.json").write_text(
        json.dumps({
            "metrics": [
                {"metric_id": key, "value": value} for key, value in metrics.items()
            ]
        }),
        encoding="utf-8",
    )
    (root / "dr_shift_metrics.csv").write_text(
        "date,peak_load_reduction_pct\n2018-04-27,5.0\n", encoding="utf-8"
    )
    hourly = root / "test_hourly_trajectories"
    hourly.mkdir(exist_ok=True)
    rows = ["date,hour,grid_import_kwh\n"] + [
        f"2018-04-27,{hour},{peak_hourly if 16 <= hour <= 21 else 1.0}\n"
        for hour in range(24)
    ]
    (hourly / "2018-04-27.csv").write_text("".join(rows), encoding="utf-8")
    return root


@pytest.fixture
def saved_gate_fixture(tmp_path: Path):
    def _create(**candidate_overrides):
        cand_root = tmp_path / "candidate"
        ref_root = tmp_path / "reference"
        pred_root = tmp_path / "predictive"

        cand_args = {
            "slug": "matd3_hierarchical_transformer_p2p_value",
            "acceptance": 0.37,
            "offer": 340.0,
            "accepted": 145.0,
            "total_p2p": 1060.0,
            "strict_feasible": 1.0,
            "unmet": 0.0,
            "comfort_pct": 99.6,
            "hvac_joint_pct": 50.0,
            "hvac_late_power_p95": 5.0,
            "mill_terminal_pct": 100.0,
            "mill_production_pct": 100.0,
            "ev_pct": 100.0,
            "mg2_projection_pct": 0.0,
            "mg3_projection_pct": 0.0,
            "cost": 870.0,
            "renewable": 0.97,
            "peak_hourly": 40.0,
        }
        cand_args.update(candidate_overrides)

        candidate = write_saved_run(cand_root, **cand_args)
        reference = write_saved_run(
            ref_root,
            slug="matd3_hierarchical_transformer",
            acceptance=0.364865,
            offer=412.65,
            accepted=150.56,
            total_p2p=1105.03,
            cost=874.63,
            renewable=0.971,
            peak_hourly=40.0,
        )
        predictive = write_saved_run(
            pred_root,
            slug="matd3_hierarchical_transformer_predictive_dr",
            acceptance=0.312712,
            offer=427.62,
            accepted=133.72,
            total_p2p=1120.66,
            hvac_joint_pct=50.0,
            hvac_late_power_p95=5.0,
            cost=880.0,
            renewable=0.972,
            peak_hourly=40.0,
        )
        return candidate, reference, predictive

    return _create


def test_gate_passes_all_conditions(saved_gate_fixture):
    candidate, reference, predictive = saved_gate_fixture()
    report = evaluate_p2p_value_gate(candidate, reference, predictive)
    assert report["overall_status"] == "PASS"
    for cond, res in report["conditions"].items():
        assert res["passed"] is True, f"Condition {cond} failed unexpectedly: {res}"


@pytest.mark.parametrize(
    ("override", "condition"),
    [
        ({"acceptance": 0.36}, "p2p_acceptance_ge_36_49pct"),
        ({"acceptance": 0.35}, "p2p_acceptance_improvement_ge_5pp"),
        ({"offer": 320.0}, "p2p_offer_volume_ge_80pct_reference"),
        ({"accepted": 140.0}, "p2p_accepted_volume_ge_95pct_reference"),
        ({"total_p2p": 1000.0}, "total_p2p_ge_95pct_reference"),
        ({"strict_feasible": 0.99}, "strict_feasible_rate_100pct"),
        ({"unmet": 0.01}, "unmet_energy_eq_0_kwh"),
        ({"comfort_pct": 99.4}, "hvac_comfort_ge_99_5pct"),
        ({"hvac_joint_pct": 44.9}, "hvac_joint_no_worse_than_predictive_5pp"),
        ({"hvac_late_power_p95": 5.3}, "hvac_late_power_no_worse_than_predictive_5pct"),
        (
            {"mill_terminal_pct": 99.0, "mill_production_pct": 99.0},
            "mill_terminal_or_production_ge_99_5pct",
        ),
        ({"ev_pct": 99.0}, "ev_departure_success_100pct"),
        ({"mg2_projection_pct": 0.1}, "mg2_projection_rate_0"),
        ({"mg3_projection_pct": 0.1}, "mg3_projection_rate_0"),
        ({"cost": 950.0}, "economic_cost_le_102pct_reference"),
        ({"peak_hourly": 50.0}, "economic_peak_import_le_105pct_reference"),
        ({"renewable": 0.96}, "economic_res_utilization_within_0_5pct"),
    ],
)
def test_each_gate_fails_independently(saved_gate_fixture, override, condition):
    candidate, reference, predictive = saved_gate_fixture(**override)
    report = evaluate_p2p_value_gate(candidate, reference, predictive)
    assert report["overall_status"] == "FAIL"
    assert report["conditions"][condition]["passed"] is False


def test_malformed_missing_file_raises_error(saved_gate_fixture):
    candidate, reference, predictive = saved_gate_fixture()
    (candidate / "paper_metrics.json").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_p2p_value_gate(candidate, reference, predictive)


def test_malformed_wrong_slug_raises_error(saved_gate_fixture):
    candidate, reference, predictive = saved_gate_fixture(slug="wrong_slug")
    with pytest.raises(ValueError, match="algorithm_slug"):
        evaluate_p2p_value_gate(candidate, reference, predictive)


def test_malformed_non_best_checkpoint_raises_error(saved_gate_fixture):
    candidate, reference, predictive = saved_gate_fixture(
        checkpoint="checkpoints/final_model.pt"
    )
    with pytest.raises(ValueError, match="checkpoint_used"):
        evaluate_p2p_value_gate(candidate, reference, predictive)


def _rewrite_json(path: Path, update) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    update(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("replacement", [None, 1, "true"])
def test_constraint_eligible_must_be_present_boolean(saved_gate_fixture, replacement):
    candidate, reference, predictive = saved_gate_fixture()
    path = candidate / "training_summary.json"

    def mutate(payload):
        selection = payload["checkpoint_selection"]
        if replacement is None:
            selection.pop("constraint_eligible")
        else:
            selection["constraint_eligible"] = replacement

    _rewrite_json(path, mutate)
    with pytest.raises(ValueError, match="constraint_eligible"):
        evaluate_p2p_value_gate(candidate, reference, predictive)


@pytest.mark.parametrize("metric_id", ["hvac_late_power_p95_kw", "mg2_dr_projection_rate"])
@pytest.mark.parametrize("replacement", [None, True, "5", float("nan"), float("inf")])
def test_required_paper_metrics_must_be_present_finite_numbers(
    saved_gate_fixture, metric_id, replacement
):
    candidate, reference, predictive = saved_gate_fixture()
    path = candidate / "paper_metrics.json"

    def mutate(payload):
        metrics = payload["metrics"]
        item = next(row for row in metrics if row["metric_id"] == metric_id)
        if replacement is None:
            metrics.remove(item)
        else:
            item["value"] = replacement

    _rewrite_json(path, mutate)
    with pytest.raises(ValueError, match=metric_id):
        evaluate_p2p_value_gate(candidate, reference, predictive)
