"""Nominal requests; accepted power and actual SOC remain environment outputs."""
from __future__ import annotations

import torch


def decode_nominal_plan(intents, current_soc, mask, battery, delta_t):
    if intents.ndim != 2 or mask.shape != intents.shape:
        raise ValueError('Expected plan and mask [batch, horizon]')
    soc = current_soc.reshape(-1)
    if len(soc) != len(intents) or not torch.isfinite(intents).all():
        raise ValueError('Invalid plan or current SOC')
    low, high = float(battery['soc_min']), float(battery['soc_max'])
    if not torch.isfinite(soc).all() or torch.any((soc < low-1e-6) | (soc > high+1e-6)):
        raise ValueError('Current SOC outside battery bounds')
    capacity = float(battery['capacity_kwh'])
    ec, ed = float(battery['eta_charge']), float(battery['eta_discharge'])
    powers, states = [], []
    for j in range(intents.shape[1]):
        charge = ((high-soc).clamp_min(0)*capacity/(ec*delta_t)).clamp_max(float(battery['max_charge_kw']))
        discharge = ((soc-low).clamp_min(0)*capacity*ed/delta_t).clamp_max(float(battery['max_discharge_kw']))
        u = intents[:, j].clamp(-1, 1)
        power = torch.where(u >= 0, u*discharge, u*charge)
        power = torch.where(mask[:, j], power, torch.zeros_like(power))
        soc = soc + torch.where(power >= 0, -power/ed, -power*ec)*delta_t/capacity
        powers.append(power)
        states.append(soc)
    return torch.stack(powers, 1), torch.stack(states, 1)
