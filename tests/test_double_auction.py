import pytest

from market.double_auction import DoubleAuctionMarket, P2POrder


def test_price_priority_and_midpoint_settlement():
    m = DoubleAuctionMarket(min_trade_energy_kwh=0.0)
    result = m.clear([
        P2POrder("mg1", "buyer", 50.0, 0.60),
        P2POrder("mg2", "seller", 20.0, 0.40),
        P2POrder("mg3", "seller", 40.0, 0.50),
    ])
    assert result["total_trade_kwh"] == pytest.approx(50.0)
    assert result["trades"][0]["seller"] == "mg2"
    assert result["trades"][0]["trade_price"] == pytest.approx(0.50)
    assert result["trades"][1]["trade_price"] == pytest.approx(0.55)


def test_equal_price_sellers_share_proportionally():
    m = DoubleAuctionMarket(min_trade_energy_kwh=0.0)
    result = m.clear([
        P2POrder("buyer", "buyer", 35.0, 0.60),
        P2POrder("s1", "seller", 30.0, 0.40),
        P2POrder("s2", "seller", 40.0, 0.40),
    ])
    assert result["by_microgrid"]["s1"]["p2p_sell_kwh"] == pytest.approx(15.0)
    assert result["by_microgrid"]["s2"]["p2p_sell_kwh"] == pytest.approx(20.0)


def test_equal_price_buyers_share_proportionally():
    m = DoubleAuctionMarket(min_trade_energy_kwh=0.0)
    result = m.clear([
        P2POrder("b1", "buyer", 30.0, 0.60),
        P2POrder("b2", "buyer", 70.0, 0.60),
        P2POrder("seller", "seller", 50.0, 0.40),
    ])
    assert result["by_microgrid"]["b1"]["p2p_buy_kwh"] == pytest.approx(15.0)
    assert result["by_microgrid"]["b2"]["p2p_buy_kwh"] == pytest.approx(35.0)


def test_multiple_price_levels_and_conservation():
    m = DoubleAuctionMarket(min_trade_energy_kwh=0.0)
    result = m.clear([
        P2POrder("b1", "buyer", 25.0, 0.70),
        P2POrder("b2", "buyer", 40.0, 0.60),
        P2POrder("s1", "seller", 20.0, 0.30),
        P2POrder("s2", "seller", 30.0, 0.50),
        P2POrder("s3", "seller", 40.0, 0.65),
    ])
    assert result["total_trade_kwh"] == pytest.approx(50.0)
    assert result["cleared_buy_kwh"] == pytest.approx(50.0)
    assert result["cleared_sell_kwh"] == pytest.approx(50.0)
    assert sum(x["p2p_net_cost"] for x in result["by_microgrid"].values()) == pytest.approx(0.0)
    for order in [
        P2POrder("b1", "buyer", 25.0, 0.70),
        P2POrder("b2", "buyer", 40.0, 0.60),
        P2POrder("s1", "seller", 20.0, 0.30),
        P2POrder("s2", "seller", 30.0, 0.50),
        P2POrder("s3", "seller", 40.0, 0.65),
    ]:
        item = result["by_microgrid"][order.microgrid_id]
        cleared = item["p2p_buy_kwh"] + item["p2p_sell_kwh"]
        assert cleared <= order.quantity_kwh + 1e-9


def test_no_cross_or_one_sided_market_uses_explicit_reference_price():
    m = DoubleAuctionMarket(min_trade_energy_kwh=0.0)
    no_cross = m.clear([
        P2POrder("buyer", "buyer", 10.0, 0.30),
        P2POrder("seller", "seller", 10.0, 0.40),
    ], no_trade_price=0.35)
    one_sided = m.clear([P2POrder("buyer", "buyer", 10.0, 0.60)], no_trade_price=0.45)
    assert no_cross["total_trade_kwh"] == 0.0
    assert no_cross["weighted_average_price"] == pytest.approx(0.35)
    assert one_sided["total_trade_kwh"] == 0.0
    assert one_sided["weighted_average_price"] == pytest.approx(0.45)
    with pytest.raises(ValueError, match="no_trade_price"):
        m.clear([P2POrder("buyer", "buyer", 10.0, 0.60)])


def test_exact_minimum_quantity_is_eligible():
    m = DoubleAuctionMarket(min_trade_energy_kwh=1.0)
    result = m.clear([
        P2POrder("buyer", "buyer", 1.0, 0.60),
        P2POrder("seller", "seller", 1.0, 0.40),
    ])
    assert result["total_trade_kwh"] == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_invalid_market_quantity_fails_fast(bad):
    m = DoubleAuctionMarket(min_trade_energy_kwh=0.0)
    with pytest.raises(ValueError):
        m.clear([P2POrder("buyer", "buyer", bad, 0.60)])


def test_invalid_market_price_role_and_duplicate_id_fail_fast():
    m = DoubleAuctionMarket(min_trade_energy_kwh=0.0)
    with pytest.raises(ValueError):
        m.clear([P2POrder("buyer", "buyer", 1.0, float("nan"))])
    with pytest.raises(ValueError):
        m.clear([P2POrder("buyer", "invalid", 1.0, 0.50)])
    with pytest.raises(ValueError):
        m.clear([
            P2POrder("same", "buyer", 1.0, 0.60),
            P2POrder("same", "seller", 1.0, 0.40),
        ])


@pytest.mark.parametrize("minimum", [float("nan"), float("inf"), -1.0])
def test_invalid_market_configuration_fails_fast(minimum):
    with pytest.raises(ValueError):
        DoubleAuctionMarket(min_trade_energy_kwh=minimum)
