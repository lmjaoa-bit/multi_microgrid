import torch

from core.action_codec import ActionCodec
from core.joint_action_projection import JointActionProjector


def synthetic_config():
    battery = {
        "capacity_kwh": 100.0,
        "max_charge_kw": 50.0,
        "max_discharge_kw": 50.0,
        "eta_charge": 0.95,
        "eta_discharge": 0.95,
        "soc_min": 0.1,
        "soc_max": 0.9,
        "initial_soc": 0.5,
    }
    return {
        "env": {"mode": "cooperative"},
        "p2p": {"pair_capacity_kw": 75.0, "enabled": False},
        "ess_sharing": {"ess_p2p_enabled": True},
        "microgrids": [
            {"id": "mg1", "battery": dict(battery),
             "dr": {"type": "industrial_mill", "rated_power_kw": 20.0}},
            {"id": "mg2", "battery": dict(battery),
             "dr": {"type": "commercial_hvac", "heating_max_kw": 30.0,
                    "cooling_max_kw": 30.0}},
            {"id": "mg3", "battery": dict(battery),
             "dr": {"type": "ev_charging", "max_charge_kw": 40.0}},
        ],
    }


def synthetic_schema():
    names = []
    for mg in ("mg1", "mg2", "mg3"):
        names += [
            f"{mg}_load",
            f"{mg}_pv",
            f"{mg}_wind",
            f"{mg}_grid_buy_price",
            f"{mg}_grid_sell_price",
            f"{mg}_feasible_charge_min",
            f"{mg}_feasible_charge_max",
            f"{mg}_feasible_discharge_min",
            f"{mg}_feasible_discharge_max",
            f"{mg}_dr_min",
            f"{mg}_dr_max",
        ]
    return names


def synthetic_state(schema):
    state = torch.zeros((1, len(schema)), dtype=torch.float32)
    index = {name: i for i, name in enumerate(schema)}
    for mg in ("mg1", "mg2", "mg3"):
        state[:, index[f"{mg}_load"]] = 0.20
        state[:, index[f"{mg}_pv"]] = 0.0
        state[:, index[f"{mg}_wind"]] = 0.0
        state[:, index[f"{mg}_grid_buy_price"]] = 0.8
        state[:, index[f"{mg}_grid_sell_price"]] = 0.2
        state[:, index[f"{mg}_feasible_charge_min"]] = 0.0
        state[:, index[f"{mg}_feasible_charge_max"]] = 1.0
        state[:, index[f"{mg}_feasible_discharge_min"]] = 0.0
        state[:, index[f"{mg}_feasible_discharge_max"]] = 1.0
    state[:, index["mg1_dr_min"]] = 0.25
    state[:, index["mg1_dr_max"]] = 0.75
    state[:, index["mg2_dr_min"]] = -0.40
    state[:, index["mg2_dr_max"]] = 0.60
    state[:, index["mg3_dr_min"]] = 0.10
    state[:, index["mg3_dr_max"]] = 0.50
    return state


def test_action_codec_preserves_legacy_and_adds_dr_contract():
    legacy = ActionCodec("single_dispatch")
    assert legacy.action_dim == 1
    assert legacy.decode([0.25]) == {"battery": 0.25}

    dr = ActionCodec("single_dispatch_dr")
    assert dr.action_dim == 2
    assert dr.decode([0.25, -0.5]) == {"battery": 0.25, "dr": -0.5}


def test_joint_projector_maps_two_raw_controls_to_three_effective_controls():
    config = synthetic_config()
    schema = synthetic_schema()
    normalizers = {
        mg: {"load": 100.0, "pv": 100.0, "wind": 100.0, "price": 1.0}
        for mg in ("mg1", "mg2", "mg3")
    }
    projector = JointActionProjector(config, schema, normalizers, price_norm=1.0)
    state = synthetic_state(schema)
    raw = torch.tensor(
        [[[0.4, -1.0], [-0.2, 0.8], [0.1, -0.7]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    effective, plans = projector(state, raw)

    assert effective.shape == (1, 3, 3)
    assert torch.allclose(effective[:, 0, 2], torch.tensor([0.25]))
    assert torch.allclose(effective[:, 1, 2], torch.tensor([0.60]))
    assert torch.allclose(effective[:, 2, 2], torch.tensor([0.15]))
    assert len(plans) == 3
    assert plans[0]["dr_projected"].item() is True
    assert plans[1]["dr_projected"].item() is True

    effective.sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert raw.grad.shape == raw.shape


def test_dr_flexible_load_is_included_before_local_first_ess_projection():
    config = synthetic_config()
    schema = synthetic_schema()
    normalizers = {
        mg: {"load": 100.0, "pv": 100.0, "wind": 100.0, "price": 1.0}
        for mg in ("mg1", "mg2", "mg3")
    }
    projector = JointActionProjector(config, schema, normalizers, price_norm=1.0)
    state = synthetic_state(schema)
    # MG1 base load is 20 kW. Raw DR=+1 is projected to duty 0.75, adding
    # 15 kW. A full ESS-discharge request must therefore see a 35 kW deficit.
    raw = torch.tensor([[[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]])
    effective, plans = projector(state, raw)

    assert torch.allclose(plans[0]["actual_load_kw"], torch.tensor([35.0]))
    assert torch.allclose(plans[0]["local"], torch.tensor([35.0]))
    assert torch.allclose(effective[0, 0, 0], torch.tensor(35.0 / 50.0))


def test_joint_projector_rejects_wrong_raw_shape():
    config = synthetic_config()
    schema = synthetic_schema()
    normalizers = {
        mg: {"load": 100.0, "pv": 100.0, "wind": 100.0, "price": 1.0}
        for mg in ("mg1", "mg2", "mg3")
    }
    projector = JointActionProjector(config, schema, normalizers, price_norm=1.0)
    state = synthetic_state(schema)
    try:
        projector(state, torch.zeros((1, 3, 1)))
    except ValueError as exc:
        assert "[batch, agents, 2]" in str(exc)
    else:
        raise AssertionError("wrong raw action shape should fail")
