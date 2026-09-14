"""Vectorized exact-Shapley helpers for MATD3-ShapleyQ.

These helpers preserve the original coalition semantics while avoiding repeated
base physical projections and one-critic-forward-per-predecessor execution.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from algorithms.matd3_shapley_q.stability import lower_soc_discharge_gradient_proxy


class CoalitionAwareProjector:
    """Keep local dispatch unchanged while batching coalition offer gating."""

    def __init__(self, projector, agent_ids):
        self.projector = projector
        self.agent_ids = tuple(agent_ids)

    def _membership_mask(self, coalition, batch_size, device):
        if isinstance(coalition, torch.Tensor):
            if coalition.ndim != 2 or coalition.shape != (batch_size, len(self.agent_ids)):
                raise ValueError("Coalition mask must have shape [batch, agents]")
            if coalition.dtype == torch.bool:
                return coalition.to(device=device)
            if not torch.isfinite(coalition).all() or not torch.all(
                (coalition == 0) | (coalition == 1)
            ):
                raise ValueError("Coalition mask values must be finite booleans or 0/1")
            return coalition.to(device=device, dtype=torch.bool)

        try:
            coalition_ids = frozenset(coalition)
        except TypeError as exc:
            raise ValueError("Coalition must be an agent-id collection or batch mask") from exc
        if not coalition_ids.issubset(self.agent_ids):
            raise ValueError("Unknown agent in coalition")
        row = torch.tensor(
            [mg in coalition_ids for mg in self.agent_ids],
            dtype=torch.bool,
            device=device,
        )
        return row.unsqueeze(0).expand(batch_size, -1)

    def _project_memberships(self, effective, plans, raw, memberships):
        """Apply K coalition membership masks to one shared base projection.

        Args:
            memberships: bool tensor [K, batch, agents].
        Returns:
            projected: [K, batch, agents, 2]
            details: per-agent coalition tensors used by legacy __call__
            proxy_masks: per-agent [batch] lower-SOC blocked-discharge masks
        """
        if memberships.ndim != 3:
            raise ValueError("memberships must have shape [coalitions, batch, agents]")
        coalition_count, batch_size, agent_count = memberships.shape
        if batch_size != raw.shape[0] or agent_count != len(self.agent_ids):
            raise ValueError("membership batch/agent dimensions do not match actions")

        channels = []
        details = []
        proxy_masks = {}
        for i, mg in enumerate(self.agent_ids):
            plan = plans[i]
            member = memberships[:, :, i]
            peer_cap = plan["local"].new_zeros((coalition_count, batch_size))
            if self.projector.enabled:
                for j, other in enumerate(plans):
                    if j == i:
                        continue
                    eligible_pair = member & memberships[:, :, j]
                    price_compatible = (other["buy"] + 1e-10 >= plan["sell"]).unsqueeze(0)
                    demand = other["demand"].clamp(max=self.projector.pair_capacity).unsqueeze(0)
                    peer_cap = peer_cap + torch.where(
                        eligible_pair & price_compatible,
                        demand.expand_as(peer_cap),
                        torch.zeros_like(peer_cap),
                    )

            charging = plan["charging"].unsqueeze(0)
            local_deficit = (plan["deficit"] - plan["local"] > 1e-7).unsqueeze(0)
            remaining = plan["remaining"].unsqueeze(0).expand_as(peer_cap)
            cap = torch.where(
                (~member) | charging | local_deficit,
                torch.zeros_like(peer_cap),
                torch.minimum(remaining, peer_cap),
            )
            uncapped_offer = plan["uncapped_offer"].unsqueeze(0).expand_as(cap)
            offer = torch.minimum(uncapped_offer, cap)
            scale = max(
                self.projector.batteries[mg]["max_charge_kw"],
                self.projector.batteries[mg]["max_discharge_kw"],
            )
            effective_local, proxy_mask = lower_soc_discharge_gradient_proxy(
                effective[:, i, 0], raw[:, i, 0], plan["dis_max"]
            )
            local = effective_local.unsqueeze(0).expand(coalition_count, -1)
            channels.append(torch.stack([local, offer / scale], dim=-1))
            proxy_masks[mg] = proxy_mask
            details.append({
                "member": member,
                "cap": cap,
                "offer": offer,
                "mask": cap > 1e-9,
                "peer_cap": peer_cap,
            })

        return torch.stack(channels, dim=2), details, proxy_masks

    def project_many(self, state, raw, coalitions: Iterable):
        """Project many coalitions with exactly one base physical projection."""
        coalitions = tuple(coalitions)
        if not coalitions:
            return {}, {mg: torch.zeros(state.shape[0], dtype=torch.bool, device=state.device)
                        for mg in self.agent_ids}
        effective, plans = self.projector(state, raw)
        memberships = torch.stack(
            [self._membership_mask(c, state.shape[0], state.device) for c in coalitions],
            dim=0,
        )
        projected_tensor, _, proxy_masks = self._project_memberships(
            effective, plans, raw, memberships
        )
        projected = {coalition: projected_tensor[i] for i, coalition in enumerate(coalitions)}
        return projected, proxy_masks

    def __call__(self, state, raw, coalition):
        """Compatibility path for one coalition, preserving returned plan diagnostics."""
        effective, plans = self.projector(state, raw)
        membership = self._membership_mask(coalition, state.shape[0], state.device).unsqueeze(0)
        projected, details, proxy_masks = self._project_memberships(
            effective, plans, raw, membership
        )
        for i, mg in enumerate(self.agent_ids):
            detail = details[i]
            plans[i].update(
                in_coalition=detail["member"][0],
                cap=detail["cap"][0],
                offer=detail["offer"][0],
                mask=detail["mask"][0],
                peer_cap=detail["peer_cap"][0],
                lower_soc_discharge_proxy=proxy_masks[mg],
            )
        return projected[0], plans


def batched_agent_shapley_values(
    state: torch.Tensor,
    projected: dict,
    predecessor_specs: Sequence,
    active,
    critic,
    agent_ids: Sequence,
):
    """Evaluate all exact predecessor marginals for one agent in one critic call."""
    predecessor_specs = tuple(predecessor_specs)
    if not predecessor_specs:
        zeros = state.new_zeros((state.shape[0], 1))
        return zeros, zeros

    batch_size = state.shape[0]
    before = torch.cat([projected[pred] for pred, _ in predecessor_specs], dim=0)
    after = torch.cat([projected[pred | {active}] for pred, _ in predecessor_specs], dim=0)
    predecessor_mask = torch.cat([
        state.new_tensor([mg in pred for mg in agent_ids]).expand(batch_size, -1)
        for pred, _ in predecessor_specs
    ], dim=0)
    repeated_state = state.repeat(len(predecessor_specs), 1)
    q1, q2 = critic(repeated_state, before, after, predecessor_mask)

    output_shape = (len(predecessor_specs), batch_size) + tuple(q1.shape[1:])
    q1 = q1.reshape(output_shape)
    q2 = q2.reshape(output_shape)
    weight_shape = (len(predecessor_specs),) + (1,) * (q1.ndim - 1)
    weights = state.new_tensor([weight for _, weight in predecessor_specs]).reshape(weight_shape)
    return (weights * q1).sum(dim=0), (weights * q2).sum(dim=0)
