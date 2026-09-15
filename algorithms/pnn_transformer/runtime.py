from __future__ import annotations

import torch

from algorithms.pnn_transformer.actor import PnnTransformerDispatchActor


def _forward_components(
    self: PnnTransformerDispatchActor,
    observation: torch.Tensor,
    *,
    collect_diagnostics: bool = True,
) -> dict[str, torch.Tensor]:
    """Historical H=4 forward math with diagnostics made optional for runtime hot paths."""
    is_single = observation.ndim == 1
    if is_single:
        observation = observation.unsqueeze(0)

    if observation.shape[-1] != len(self.schema):
        raise ValueError(
            f"Observation schema mismatch: expected {len(self.schema)}, got {observation.shape[-1]}"
        )

    batch_size = observation.shape[0]
    hour_feats = observation.index_select(-1, self._hour_indices)
    local_feats = observation.index_select(-1, self._local_indices)
    market_feats = observation.index_select(-1, self._market_indices)
    peer_raw_feats = observation.index_select(-1, self._all_peer_indices)
    forecast_raw_feats = observation.index_select(-1, self._all_forecast_indices)

    dispatch_token = self.dispatch_norm(
        self.dispatch_token.expand(batch_size, 1, self.embed_dim)
    )
    hour_sin = hour_feats[:, 0]
    hour_cos = hour_feats[:, 1]
    hour_idx = self._decode_hour_index(hour_sin, hour_cos)
    hour_token = self.hour_norm(
        self.hour_embedding(hour_idx) + self.hour_linear(hour_feats)
    ).unsqueeze(1)

    local_rep = self.local_encoder(local_feats)
    local_token = local_rep.unsqueeze(1)

    f_all = forecast_raw_feats.reshape(batch_size, self.forecast_horizon, -1)
    f_tokens = self.forecast_encoder(f_all)
    future_hour_idx = (hour_idx.unsqueeze(-1) + self._step_offsets) % 24
    future_hour_emb = self.hour_embedding(future_hour_idx)
    horizon_ids = torch.arange(
        self.forecast_horizon, device=observation.device, dtype=torch.long
    )
    horizon_emb = self.horizon_embedding(horizon_ids)
    forecast_tokens = self.forecast_norm(
        f_tokens + horizon_emb.unsqueeze(0) + future_hour_emb
    )

    peer_all = peer_raw_feats.reshape(batch_size, len(self.peers), 2)
    peer_encoded = self.peer_encoder(peer_all)
    peer_spatial_emb = self.peer_id_embedding(self._peer_agent_indices)
    peer_tokens = self.peer_norm(peer_encoded + peer_spatial_emb.unsqueeze(0))
    market_token = self.market_encoder(market_feats).unsqueeze(1)

    tokens = torch.cat(
        [dispatch_token, hour_token, local_token, forecast_tokens, peer_tokens, market_token],
        dim=1,
    )
    attended = self.transformer(tokens)
    dispatch_rep = attended[:, 0, :]

    e_pnn_local = self.pnn_local_encoder(local_feats)
    e_pnn_temporal = self.pnn_temporal_encoder(forecast_raw_feats)
    e_pnn_peer = self.pnn_peer_encoder(peer_raw_feats)
    e_pnn_market = self.pnn_market_encoder(market_feats)

    p_temporal = e_pnn_local * e_pnn_temporal
    p_spatial = e_pnn_local * e_pnn_peer
    p_market = e_pnn_local * e_pnn_market
    raw_products = torch.cat([p_temporal, p_spatial, p_market], dim=-1)
    normed_products = self.product_norm(raw_products)
    product_signals = self.product_scale * normed_products

    fused = torch.cat([dispatch_rep, product_signals, local_rep], dim=-1)
    features = self.dispatch_backbone(fused)
    action = self.dispatch_head(features)

    if collect_diagnostics:
        with torch.no_grad():
            d_norm = float(dispatch_rep.norm(dim=-1).mean().detach().cpu())
            p_norm = float(product_signals.norm(dim=-1).mean().detach().cpu())
            l_norm = float(local_rep.norm(dim=-1).mean().detach().cpu())
            t_prod_norm = float(p_temporal.norm(dim=-1).mean().detach().cpu())
            s_prod_norm = float(p_spatial.norm(dim=-1).mean().detach().cpu())
            m_prod_norm = float(p_market.norm(dim=-1).mean().detach().cpu())
            self.last_diagnostics = {
                "dispatch_norm": d_norm,
                "product_norm": p_norm,
                "local_norm": l_norm,
                "temporal_product_norm": t_prod_norm,
                "spatial_product_norm": s_prod_norm,
                "market_product_norm": m_prod_norm,
                "product_to_dispatch_ratio": float(p_norm / max(d_norm, 1e-8)),
                "local_to_dispatch_ratio": float(l_norm / max(d_norm, 1e-8)),
            }

    if is_single:
        return {
            "action": action.squeeze(0),
            "dispatch_rep": dispatch_rep.squeeze(0),
            "product_signals": product_signals.squeeze(0),
            "local_rep": local_rep.squeeze(0),
            "fused": fused.squeeze(0),
            "attended": attended.squeeze(0),
            "tokens": tokens.squeeze(0),
        }
    return {
        "action": action,
        "dispatch_rep": dispatch_rep,
        "product_signals": product_signals,
        "local_rep": local_rep,
        "fused": fused,
        "attended": attended,
        "tokens": tokens,
    }


def _forward(
    self: PnnTransformerDispatchActor,
    observation: torch.Tensor,
    *,
    collect_diagnostics: bool = True,
) -> torch.Tensor:
    return self.forward_components(
        observation, collect_diagnostics=collect_diagnostics
    )["action"]


def install_historical_actor_runtime_optimizations():
    """Make diagnostics opt-in on hot paths without changing historical Actor math."""
    cls = PnnTransformerDispatchActor
    if getattr(cls, "_historical_runtime_optimizations_installed", False):
        return cls
    cls.forward_components = _forward_components
    cls.forward = _forward
    cls._historical_runtime_optimizations_installed = True
    return cls
