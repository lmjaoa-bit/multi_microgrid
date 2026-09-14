from __future__ import annotations

from collections.abc import Mapping
import torch
from torch import nn


MARKET_FEATURES = frozenset({
    "previous_p2p_net", "previous_grid_net", "previous_system_p2p",
    "previous_ess_p2p_offer", "previous_ess_p2p_accepted", "previous_ess_p2p_unmatched",
    "previous_system_buy_demand", "previous_system_sell_offer", "previous_clearing_ratio",
})


class PNNDispatchActor(nn.Module):
    """Single-dispatch actor with Product-based Neural Network (PNN) feature interactions."""

    def __init__(
        self,
        schema: tuple[str, ...],
        agent_ids: tuple[str, ...] | list[str],
        active_agent: str,
        *,
        embed_dim: int = 32,
        hidden_dims: tuple[int, ...] = (96, 64),
        product_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.schema = tuple(schema)
        if len(set(self.schema)) != len(self.schema):
            raise ValueError("Duplicate observation features in schema")

        peers = tuple(mg + "_" for mg in agent_ids if mg != active_agent)
        groups: dict[str, list[int]] = {"local": [], "temporal": [], "market": [], "peer": []}
        for i, name in enumerate(self.schema):
            if name in MARKET_FEATURES:
                groups["market"].append(i)
            elif name.startswith(peers):
                groups["peer"].append(i)
            elif "_t+" in name:
                groups["temporal"].append(i)
            else:
                groups["local"].append(i)

        if any(not group for group in groups.values()):
            raise ValueError("PNNDispatchActor requires local, temporal, market, and peer observations")

        for name, indices in groups.items():
            self.register_buffer("_" + name + "_indices", torch.tensor(indices, dtype=torch.long), persistent=False)

        self.feature_dim = int(embed_dim)
        self.product_scale = float(product_scale)
        if not 0 < self.product_scale <= 1:
            raise ValueError("product_scale must be in (0, 1]")

        def encoder(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
            )

        self.local_encoder = encoder(len(groups["local"]), self.feature_dim)
        self.temporal_encoder = encoder(len(groups["temporal"]), self.feature_dim)
        self.market_encoder = encoder(len(groups["market"]), self.feature_dim)
        self.peer_encoder = encoder(len(groups["peer"]), self.feature_dim)
        self.product_norm = nn.LayerNorm(self.feature_dim * 3)

        total_width = self.feature_dim * 7
        layers: list[nn.Module] = []
        width = total_width
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.ReLU()])
            width = hidden
        self.dispatch_backbone = nn.Sequential(*layers)

        final_linear = nn.Linear(width, 1)
        nn.init.uniform_(final_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(final_linear.bias, -3e-3, 3e-3)
        self.dispatch_head = nn.Sequential(final_linear, nn.Tanh())

        self._parameter_count = sum(p.numel() for p in self.parameters())
        self._product_metrics: dict[str, float] = {}
        self._last_product_norms: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    def forward(
        self, observation: torch.Tensor, *, collect_diagnostics: bool = True
    ) -> torch.Tensor:
        if observation.shape[-1] != len(self.schema):
            raise ValueError(f"Observation schema mismatch: expected {len(self.schema)}, got {observation.shape[-1]}")

        e_local = self.local_encoder(observation.index_select(-1, self._local_indices))
        e_temporal = self.temporal_encoder(observation.index_select(-1, self._temporal_indices))
        e_market = self.market_encoder(observation.index_select(-1, self._market_indices))
        e_peer = self.peer_encoder(observation.index_select(-1, self._peer_indices))

        linear_signals = torch.cat([e_local, e_temporal, e_market, e_peer], dim=-1)
        p_temporal = e_local * e_temporal
        p_spatial = e_local * e_peer
        p_market = e_local * e_market
        normed_products = self.product_norm(torch.cat([p_temporal, p_spatial, p_market], dim=-1))
        product_signals = self.product_scale * normed_products

        if collect_diagnostics:
            self._last_product_norms = (
                linear_signals.detach().norm(dim=-1).mean(),
                product_signals.detach().norm(dim=-1).mean(),
                p_temporal.detach().norm(dim=-1).mean(),
                p_spatial.detach().norm(dim=-1).mean(),
            )

        combined = torch.cat([linear_signals, product_signals], dim=-1)
        return self.dispatch_head(self.dispatch_backbone(combined))

    def product_diagnostics(self) -> Mapping[str, float]:
        if self._last_product_norms is not None:
            lin_norm, prod_norm, temp_norm, spat_norm = self._last_product_norms
            lin_value = float(lin_norm)
            prod_value = float(prod_norm)
            self._product_metrics = {
                "linear_signal_norm": lin_value,
                "product_signal_norm": prod_value,
                "product_to_linear_ratio": float(prod_value / max(lin_value, 1e-8)),
                "temporal_product_norm": float(temp_norm),
                "spatial_product_norm": float(spat_norm),
            }
            self._last_product_norms = None
        return dict(self._product_metrics)

    def branch_modules(self) -> Mapping[str, tuple[nn.Module, ...]]:
        return {
            "local_features": (self.local_encoder,),
            "temporal_features": (self.temporal_encoder,),
            "market_features": (self.market_encoder,),
            "peer_features": (self.peer_encoder,),
            "product_norm": (self.product_norm,),
            "dispatch": (self.dispatch_backbone, self.dispatch_head),
        }
