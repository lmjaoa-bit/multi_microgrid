from __future__ import annotations

from typing import Any

import torch

from core.single_dispatch import SingleDispatchProjector


class JointActionProjector:
    """Differentiable raw [ESS, DR] to effective [ESS-local, ESS-trade, DR]."""

    semantics = "single_dispatch_dr_v1"

    def __init__(self, config, schema, normalizers, price_norm):
        self.ids = [str(mg["id"]) for mg in config["microgrids"]]
        if len(self.ids) != 3:
            raise ValueError("JointActionProjector requires exactly three microgrids")
        self.index = {name: i for i, name in enumerate(schema)}
        self.ess = SingleDispatchProjector(config, schema, normalizers, price_norm)
        for mg in self.ids:
            for suffix in ("dr_min", "dr_max"):
                name = f"{mg}_{suffix}"
                if name not in self.index:
                    raise ValueError(f"Global state is missing required DR bound: {name}")

    def _dr_requested(self, raw: torch.Tensor, agent_index: int) -> torch.Tensor:
        if self.ids[agent_index] == "mg2":
            return raw.clamp(-1.0, 1.0)
        return (raw.clamp(-1.0, 1.0) + 1.0) * 0.5

    def __call__(self, state: torch.Tensor, raw: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        if raw.ndim != 3 or raw.shape[1:] != (len(self.ids), 2):
            raise ValueError("single_dispatch_dr requires raw controls [batch, agents, 2]")
        if state.ndim != 2 or state.shape[0] != raw.shape[0]:
            raise ValueError("State and action batch dimensions must match")

        ess_effective, ess_plans = self.ess(state, raw[..., 0:1])
        dr_effective = []
        plans: list[dict[str, Any]] = []
        for i, mg in enumerate(self.ids):
            requested = self._dr_requested(raw[:, i, 1], i)
            lower = state[:, self.index[f"{mg}_dr_min"]]
            upper = state[:, self.index[f"{mg}_dr_max"]]
            if torch.any(lower > upper + 1e-7):
                raise ValueError(f"{mg}: invalid DR feasible interval in global state")
            effective = torch.maximum(lower, torch.minimum(requested, upper))
            dr_effective.append(effective)
            plan = dict(ess_plans[i])
            plan.update(
                dr_requested=requested,
                dr_effective=effective,
                dr_min=lower,
                dr_max=upper,
                dr_projection_delta=effective - requested,
                dr_projected=(effective - requested).abs() > 1e-7,
            )
            plans.append(plan)

        dr_channel = torch.stack(dr_effective, dim=1).unsqueeze(-1)
        return torch.cat([ess_effective, dr_channel], dim=-1), plans
