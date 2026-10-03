from __future__ import annotations

import copy

import torch
from torch import nn

from algorithms.matd3_hierarchical_transformer_q_guided.algorithm import (
    Matd3HierarchicalTransformerQGuided,
)
from algorithms.matd3_hierarchical_transformer_q_readout.actor import DeviceForecastReadoutActor
from core.reward_v2 import stable_contract_hash


class Matd3HierarchicalTransformerQReadout(Matd3HierarchicalTransformerQGuided):
    """T-Q with retained base-load forecasts and separate ESS/DR forecast reads."""

    SLUG = "matd3_hierarchical_transformer_q_readout"
    FORECAST_CONTEXT_VERSION = "peer_balance_base_load_h8_device_readout_v1"

    def __init__(self, env, config, device="cpu", seed=42):
        super().__init__(env, config, device=device, seed=seed)
        settings = config.get("algorithm", config)
        net = settings.get("network", {})
        if int(net.get("forecast_horizon", 8)) != 8 or not net.get("include_peer_forecast", False):
            raise ValueError("Device forecast readout requires H8 with peer energy-balance forecasts")
        self.actors = nn.ModuleDict({
            mg: DeviceForecastReadoutActor(
                self.observation_schemas[mg], self.agent_ids, mg, env.cfg_by_id[mg],
                power_normalizers=self.actor_feature_normalizers[mg],
                embed_dim=int(net.get("embed_dim", 32)),
                num_heads=int(net.get("num_heads", 4)),
                num_layers=int(net.get("num_layers", 2)),
                hidden_dim=int(net.get("hidden_dim", 64)),
                forecast_horizon=8,
                include_peer_forecast=True,
                peer_forecast_features=tuple(net.get("peer_forecast_features", ["energy_balance"])),
            )
            for mg in self.agent_ids
        }).to(self.device)
        self.target_actors = copy.deepcopy(self.actors).to(self.device)
        self.actor_optimizers = {
            mg: torch.optim.Adam(self.actors[mg].parameters(), lr=float(settings.get("actor_lr", 2e-4)))
            for mg in self.agent_ids
        }
        # Ordinary T caches parameter tuples for sampled diagnostics. Rebind them
        # after replacing its actors so they inspect the live network and reads.
        self._actor_parameters = {
            mg: tuple(actor.parameters()) for mg, actor in self.actors.items()
        }
        self._actor_branch_parameters = {
            mg: {name: tuple(p for module in modules for p in module.parameters())
                 for name, modules in actor.branch_modules().items()}
            for mg, actor in self.actors.items()
        }
        self._actor_layer_parameters = {
            mg: {f"layer_{i}": tuple(layer.parameters())
                 for i, layer in enumerate(actor.transformer.layers)}
            for mg, actor in self.actors.items()
        }
        self.actor_forecast_context_version = self.FORECAST_CONTEXT_VERSION
        if self.reward_v2_contract is not None:
            contract = copy.deepcopy(self.reward_v2_contract)
            contract.pop("signature", None)
            contract["actor_forecast_context_version"] = self.actor_forecast_context_version
            contract["signature"] = stable_contract_hash(contract)
            self.reward_v2_contract = contract

    @staticmethod
    def _ess_pre_tanh_logit(actor: DeviceForecastReadoutActor, observations: torch.Tensor) -> torch.Tensor:
        return actor.action_logits(observations)[:, 0]
