from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class LossTerms:
    plan: torch.Tensor
    dynamics: torch.Tensor
    first: torch.Tensor
    metrics: dict


def _delta(power, battery, dt):
    return torch.where(power >= 0, -power / float(battery['eta_discharge']),
                       -power * float(battery['eta_charge'])) * dt / float(battery['capacity_kwh'])


def plan_loss_terms(output, batch, agent_id, projected_plan, battery_config, delta_t):
    mask = batch.mask.to(output.soc_after.dtype)
    scale = max(float(battery_config['max_charge_kw']), float(battery_config['max_discharge_kw']))
    mean = lambda value: ((value * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()
    soc_error = output.soc_after - batch.soc_after[agent_id]
    power_error = (output.ess_net_kw - batch.ess_net_kw[agent_id]) / scale
    plan = mean(F.smooth_l1_loss(output.soc_after, batch.soc_after[agent_id], reduction='none'))
    plan = plan + mean(F.smooth_l1_loss(output.ess_net_kw / scale, batch.ess_net_kw[agent_id] / scale, reduction='none'))
    previous = torch.cat([output.current_soc, output.soc_after[:, :-1]], -1)
    expected = previous + _delta(output.ess_net_kw, battery_config, delta_t)
    dynamics = mean(F.smooth_l1_loss(output.soc_after, expected, reduction='none'))
    actual = projected_plan['actual_discharge_kw'] - projected_plan['actual_charge_kw']
    first_soc = output.current_soc[:, 0] + _delta(actual, battery_config, delta_t)
    first = F.smooth_l1_loss(output.ess_net_kw[:, 0] / scale, actual / scale)
    first = first + F.smooth_l1_loss(output.soc_after[:, 0], first_soc)
    return LossTerms(plan, dynamics, first, dict(soc_mae=float(mean(soc_error.abs()).detach()),
        power_mae_kw=float(mean(power_error.abs()).detach() * scale),
        first_power_error_kw=float((output.ess_net_kw[:, 0] - actual).abs().mean().detach())))
