import numpy as np

from envs.cooperative_energy_sharing import CooperativeEnergySharing


def test_structured_pool_shares_one_capacity_across_both_pair_directions():
    pool = CooperativeEnergySharing(pair_capacity_kw=75.0)
    result = pool.allocate_structured(
        renewable_offers_kw={"mg1": 75.0, "mg2": 75.0, "mg3": 0.0},
        ess_offers_kw={"mg1": 0.0, "mg2": 0.0, "mg3": 0.0},
        load_demands_kw={"mg1": 0.0, "mg2": 0.0, "mg3": 0.0},
        charge_demands_kw={"mg1": 75.0, "mg2": 75.0, "mg3": 0.0},
        grid_sell_price=0.1,
        grid_buy_price=0.3,
    )

    forward = result.pair_flows_kwh.get("mg1->mg2", 0.0)
    reverse = result.pair_flows_kwh.get("mg2->mg1", 0.0)
    assert forward + reverse <= 75.0 + 1e-9
    # Symmetric simultaneous requests should not gain an ID-order advantage.
    assert np.isclose(forward, reverse, atol=1e-9)
    assert np.isclose(forward + reverse, 75.0, atol=1e-9)


def test_unconstrained_counterfactual_reports_only_pair_capacity_blocking():
    pool = CooperativeEnergySharing(pair_capacity_kw=10.0)
    inputs = dict(
        renewable_offers_kw={"mg1": 30.0, "mg2": 0.0, "mg3": 0.0},
        ess_offers_kw={"mg1": 0.0, "mg2": 0.0, "mg3": 0.0},
        load_demands_kw={"mg1": 0.0, "mg2": 30.0, "mg3": 0.0},
        charge_demands_kw={"mg1": 0.0, "mg2": 0.0, "mg3": 0.0},
        grid_sell_price=0.1,
        grid_buy_price=0.3,
    )
    actual = pool.allocate_structured(**inputs)
    unconstrained = pool.allocate_structured_unconstrained(**inputs)
    assert actual.total_p2p_kwh == 10.0
    assert unconstrained.total_p2p_kwh == 30.0
    assert unconstrained.total_p2p_kwh - actual.total_p2p_kwh == 20.0
