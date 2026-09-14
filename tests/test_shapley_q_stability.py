import pytest
import torch
from torch import nn

from algorithms.matd3_shapley_q.stability import (
    clip_module_grad_norm,
    conservative_cooperation_q,
    lower_soc_discharge_gradient_proxy,
)

AGENTS = ("mg1", "mg2", "mg3")


def test_conservative_cooperation_q_takes_min_after_full_reconstruction():
    q1 = {mg: torch.tensor([[v]]) for mg, v in zip(AGENTS, (10.0, 1.0, 10.0))}
    q2 = {mg: torch.tensor([[v]]) for mg, v in zip(AGENTS, (1.0, 10.0, 1.0))}

    total1, total2, conservative, selected = conservative_cooperation_q(q1, q2, AGENTS)

    assert total1.item() == pytest.approx(21.0)
    assert total2.item() == pytest.approx(12.0)
    assert conservative.item() == pytest.approx(12.0)
    assert sum(selected.values()).item() == pytest.approx(12.0)
    assert sum(torch.minimum(q1[mg], q2[mg]) for mg in AGENTS).item() == pytest.approx(3.0)


def test_lower_soc_proxy_preserves_forward_value_but_restores_backward_gradient():
    raw = torch.tensor([0.8], requires_grad=True)
    physical = raw * 0.0

    proxied, mask = lower_soc_discharge_gradient_proxy(
        physical, raw, torch.zeros_like(raw)
    )

    assert mask.item() is True
    assert proxied.detach().item() == pytest.approx(physical.detach().item())
    proxied.sum().backward()
    assert raw.grad.item() == pytest.approx(1.0)


def test_lower_soc_proxy_does_not_fire_for_charging_direction():
    raw = torch.tensor([-0.8], requires_grad=True)
    physical = raw * 0.0

    proxied, mask = lower_soc_discharge_gradient_proxy(
        physical, raw, torch.zeros_like(raw)
    )

    assert mask.item() is False
    proxied.sum().backward()
    assert raw.grad.item() == pytest.approx(0.0)


def test_critic_gradient_clip_reports_pre_and_post_norms():
    module = nn.Linear(1, 1, bias=False)
    module.weight.grad = torch.tensor([[100.0]])

    pre, post, clipped = clip_module_grad_norm(module, 10.0)

    assert pre == pytest.approx(100.0)
    assert post == pytest.approx(10.0, rel=1e-5)
    assert clipped is True


def test_conservative_cooperation_q_selects_one_complete_head_per_sample():
    q1 = {
        "mg1": torch.tensor([[5.0], [0.0]]),
        "mg2": torch.tensor([[5.0], [0.0]]),
        "mg3": torch.tensor([[5.0], [0.0]]),
    }
    q2 = {
        "mg1": torch.tensor([[0.0], [4.0]]),
        "mg2": torch.tensor([[0.0], [4.0]]),
        "mg3": torch.tensor([[0.0], [4.0]]),
    }

    _, _, conservative, selected = conservative_cooperation_q(q1, q2, AGENTS)

    torch.testing.assert_close(conservative, torch.tensor([[0.0], [0.0]]))
    torch.testing.assert_close(sum(selected.values()), conservative)
    for mg in AGENTS:
        torch.testing.assert_close(selected[mg], torch.tensor([[0.0], [0.0]]))
