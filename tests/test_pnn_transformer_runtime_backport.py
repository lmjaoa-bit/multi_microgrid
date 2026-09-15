from __future__ import annotations

from pathlib import Path

import torch
import yaml

from algorithms.pnn_transformer.actor import PnnTransformerDispatchActor


ROOT = Path(__file__).resolve().parents[1]


def _historical_schema() -> list[str]:
    schema = [
        "hour_sin", "hour_cos",
        "load", "pv", "wind", "soc",
        "available_charge_ratio", "available_discharge_ratio", "terminal_soc_gap",
        "feasible_charge_min_ratio", "feasible_charge_max_ratio",
        "feasible_discharge_min_ratio", "feasible_discharge_max_ratio",
        "mg2_net_load", "mg2_soc", "mg3_net_load", "mg3_soc",
        "grid_buy_price", "grid_sell_price",
        "previous_p2p_net", "previous_grid_net", "previous_system_p2p",
        "previous_ess_p2p_offer", "previous_ess_p2p_accepted", "previous_ess_p2p_unmatched",
        "previous_system_buy_demand", "previous_system_sell_offer", "previous_clearing_ratio",
    ]
    for k in range(1, 5):
        schema.extend([
            f"load_t+{k}", f"pv_t+{k}", f"wind_t+{k}", f"price_t+{k}",
            f"mg2_net_load_t+{k}", f"mg3_net_load_t+{k}",
        ])
    return schema


def test_historical_899_config_keeps_forecast_horizon_four():
    cfg = yaml.safe_load(
        (ROOT / "configs" / "algorithms" / "pnn_transformer.yaml").read_text(encoding="utf-8")
    )
    assert cfg["algorithm"]["network"]["forecast_horizon"] == 4


def test_actor_diagnostics_can_be_disabled_without_changing_action():
    from algorithms.registry import get_algorithm_spec

    get_algorithm_spec("pnn_transformer")  # installs runtime-only Actor method overrides
    torch.manual_seed(42)
    actor = PnnTransformerDispatchActor(
        _historical_schema(), ["mg1", "mg2", "mg3"], "mg1", forecast_horizon=4
    ).eval()
    observation = torch.randn(8, len(_historical_schema()))

    with torch.no_grad():
        reference = actor(observation)
        diagnostics = dict(actor.last_diagnostics)
        optimized = actor(observation, collect_diagnostics=False)

    assert torch.equal(reference, optimized)
    assert actor.last_diagnostics == diagnostics


def test_plugin_installs_runtime_layer_without_replacing_algorithm_definition():
    from algorithms.pnn_transformer.algorithm import PnnTransformer
    from algorithms.registry import get_algorithm_spec
    from algorithms.runtime_helpers import optimized_single_dispatch_update, select_single_dispatch_actions

    spec = get_algorithm_spec("pnn_transformer")
    assert spec.algorithm_class is PnnTransformer
    assert getattr(PnnTransformer, "_runtime_optimizations_installed", False)
    assert PnnTransformer.update is optimized_single_dispatch_update
    assert PnnTransformer.select_actions is select_single_dispatch_actions
