from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class P2POrder:
    microgrid_id: str
    role: str  # buyer / seller
    quantity_kwh: float
    bid_price: float


@dataclass(frozen=True)
class P2PTrade:
    buyer: str
    seller: str
    quantity_kwh: float
    buyer_bid_price: float
    seller_bid_price: float
    trade_price: float


class DoubleAuctionMarket:
    """Lossless sealed-bid double auction with midpoint bilateral settlement.

    Price priority:
      buyers high -> low; sellers low -> high.
    Equal-price orders share scarce volume proportionally by order quantity.
    """

    def __init__(self, min_trade_energy_kwh: float = 1.0, tolerance: float = 1e-9):
        min_trade_energy_kwh = float(min_trade_energy_kwh)
        tolerance = float(tolerance)
        if not math.isfinite(min_trade_energy_kwh) or min_trade_energy_kwh < 0.0:
            raise ValueError("min_trade_energy_kwh must be finite and nonnegative.")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Market tolerance must be finite and positive.")
        self.min_trade_energy_kwh = min_trade_energy_kwh
        self.tol = tolerance

    def _levels(self, orders: List[dict], reverse: bool) -> List[List[dict]]:
        orders = sorted(orders, key=lambda x: x["bid_price"], reverse=reverse)
        levels: List[List[dict]] = []
        for order in orders:
            if not levels or abs(order["bid_price"] - levels[-1][0]["bid_price"]) > self.tol:
                levels.append([order])
            else:
                levels[-1].append(order)
        return levels

    def _proportional(self, level: List[dict], volume: float) -> Dict[str, float]:
        total = sum(x["remaining"] for x in level)
        if total <= self.tol or volume <= self.tol:
            return {x["id"]: 0.0 for x in level}
        volume = min(volume, total)
        return {x["id"]: volume * x["remaining"] / total for x in level}

    def is_quantity_eligible(self, quantity_kwh: float) -> bool:
        quantity = float(quantity_kwh)
        if not math.isfinite(quantity):
            raise ValueError("P2P order quantity must be finite.")
        return quantity > self.tol and quantity + self.tol >= self.min_trade_energy_kwh

    def _validate_orders(self, orders: List[P2POrder]) -> None:
        ids = set()
        for order in orders:
            if not order.microgrid_id:
                raise ValueError("P2P order microgrid_id cannot be empty.")
            if order.microgrid_id in ids:
                raise ValueError(f"Microgrid {order.microgrid_id!r} submitted more than one P2P order.")
            ids.add(order.microgrid_id)
            if order.role not in {"buyer", "seller"}:
                raise ValueError(f"Unsupported P2P order role: {order.role!r}")
            quantity = float(order.quantity_kwh)
            price = float(order.bid_price)
            if not math.isfinite(quantity) or quantity < 0.0:
                raise ValueError("P2P order quantities must be finite and nonnegative.")
            if not math.isfinite(price) or price < 0.0:
                raise ValueError("P2P order prices must be finite and nonnegative.")

    def clear(self, orders: Iterable[P2POrder], no_trade_price: float | None = None) -> Dict:
        orders = list(orders)
        self._validate_orders(orders)
        if no_trade_price is not None:
            no_trade_price = float(no_trade_price)
            if not math.isfinite(no_trade_price) or no_trade_price < 0.0:
                raise ValueError("no_trade_price must be finite and nonnegative.")
        active = [o for o in orders if self.is_quantity_eligible(o.quantity_kwh)]
        buyers = [{"id": o.microgrid_id, "bid_price": float(o.bid_price), "remaining": float(o.quantity_kwh), "original": float(o.quantity_kwh)} for o in active if o.role == "buyer"]
        sellers = [{"id": o.microgrid_id, "bid_price": float(o.bid_price), "remaining": float(o.quantity_kwh), "original": float(o.quantity_kwh)} for o in active if o.role == "seller"]
        b_levels = self._levels(buyers, reverse=True)
        s_levels = self._levels(sellers, reverse=False)
        trades: List[P2PTrade] = []
        bi = si = 0

        while bi < len(b_levels) and si < len(s_levels):
            bl = b_levels[bi]
            sl = s_levels[si]
            bp = bl[0]["bid_price"]
            sp = sl[0]["bid_price"]
            if bp + self.tol < sp:
                break
            buy_total = sum(x["remaining"] for x in bl)
            sell_total = sum(x["remaining"] for x in sl)
            if buy_total <= self.tol:
                bi += 1
                continue
            if sell_total <= self.tol:
                si += 1
                continue
            volume = min(buy_total, sell_total)
            ba = self._proportional(bl, volume)
            sa = self._proportional(sl, volume)
            bq = [[x["id"], ba[x["id"]]] for x in bl if ba[x["id"]] > self.tol]
            sq = [[x["id"], sa[x["id"]]] for x in sl if sa[x["id"]] > self.tol]
            i = j = 0
            price = 0.5 * (bp + sp)
            while i < len(bq) and j < len(sq):
                q = min(bq[i][1], sq[j][1])
                if q > self.tol:
                    trades.append(P2PTrade(bq[i][0], sq[j][0], q, bp, sp, price))
                    bq[i][1] -= q
                    sq[j][1] -= q
                if bq[i][1] <= self.tol:
                    i += 1
                if sq[j][1] <= self.tol:
                    j += 1
            for x in bl:
                x["remaining"] = max(0.0, x["remaining"] - ba[x["id"]])
            for x in sl:
                x["remaining"] = max(0.0, x["remaining"] - sa[x["id"]])
            if sum(x["remaining"] for x in bl) <= self.tol:
                bi += 1
            if sum(x["remaining"] for x in sl) <= self.tol:
                si += 1

        mg_ids = sorted({o.microgrid_id for o in orders})
        by_mg = {
            mg: {
                "p2p_buy_kwh": 0.0,
                "p2p_sell_kwh": 0.0,
                "p2p_buy_cost": 0.0,
                "p2p_sell_revenue": 0.0,
                "p2p_net_cost": 0.0,
                "p2p_net_kwh": 0.0,
                "weighted_average_trade_price": 0.0,
            }
            for mg in mg_ids
        }
        value_sum = {mg: 0.0 for mg in mg_ids}
        total_q = total_value = 0.0
        for tr in trades:
            value = tr.quantity_kwh * tr.trade_price
            total_q += tr.quantity_kwh
            total_value += value
            b = by_mg[tr.buyer]
            s = by_mg[tr.seller]
            b["p2p_buy_kwh"] += tr.quantity_kwh
            b["p2p_buy_cost"] += value
            b["p2p_net_kwh"] += tr.quantity_kwh
            s["p2p_sell_kwh"] += tr.quantity_kwh
            s["p2p_sell_revenue"] += value
            s["p2p_net_kwh"] -= tr.quantity_kwh
            value_sum[tr.buyer] += value
            value_sum[tr.seller] += value

        for mg, item in by_mg.items():
            item["p2p_net_cost"] = item["p2p_buy_cost"] - item["p2p_sell_revenue"]
            gross = item["p2p_buy_kwh"] + item["p2p_sell_kwh"]
            if gross > self.tol:
                item["weighted_average_trade_price"] = value_sum[mg] / gross

        if total_q <= self.tol and no_trade_price is None:
            raise ValueError("no_trade_price is required when no P2P trade clears.")
        market_price = total_value / total_q if total_q > self.tol else no_trade_price
        assert market_price is not None
        return {
            "trades": [t.__dict__.copy() for t in trades],
            "by_microgrid": by_mg,
            "total_trade_kwh": total_q,
            "cleared_buy_kwh": total_q,
            "cleared_sell_kwh": total_q,
            "weighted_average_price": float(market_price),
            "submitted_buy_kwh": sum(x["original"] for x in buyers),
            "submitted_sell_kwh": sum(x["original"] for x in sellers),
        }
