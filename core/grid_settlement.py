from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalGrid:
    """Loosely coupled external-grid balancing interface for v1."""

    import_limit_kw: float | None
    export_limit_kw: float

    def settle(self, residual_net_kw: float) -> dict[str, float]:
        net = float(residual_net_kw)
        if net >= 0.0:
            grid_export = min(net, self.export_limit_kw)
            curtailment = max(0.0, net - grid_export)
            grid_import = unmet = 0.0
        else:
            demand = -net
            grid_import = demand if self.import_limit_kw is None else min(demand, self.import_limit_kw)
            unmet = max(0.0, demand - grid_import)
            grid_export = curtailment = 0.0
        residual = net + grid_import + unmet - grid_export - curtailment
        return {
            "grid_import_kw": float(grid_import),
            "grid_export_kw": float(grid_export),
            "unmet_load_kw": float(unmet),
            "renewable_curtailment_kw": float(curtailment),
            "power_balance_residual_kw": float(residual),
        }
