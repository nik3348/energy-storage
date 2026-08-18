import pytest

from energy_storage.market import Offer, clear_hour

CAP = 3000.0
FLOOR = -100.0


def offer(price, mw, name="g", tech="gas"):
    return Offer(name, tech, price, mw)


def test_marginal_offer_sets_uniform_price():
    offers = [offer(10.0, 100.0, "cheap"), offer(50.0, 100.0, "mid"), offer(90.0, 100.0, "dear")]
    price, dispatch = clear_hour(offers, 150.0, CAP, FLOOR)
    assert price == 50.0
    assert [(o.generator, mw) for o, mw in dispatch] == [("cheap", 100.0), ("mid", 50.0)]


def test_offers_cleared_in_merit_order_regardless_of_input_order():
    offers = [offer(90.0, 100.0, "dear"), offer(10.0, 100.0, "cheap")]
    price, dispatch = clear_hour(offers, 100.0, CAP, FLOOR)
    assert price == 10.0
    assert dispatch[0][0].generator == "cheap"


def test_shortfall_prices_at_cap():
    price, dispatch = clear_hour([offer(10.0, 50.0)], 80.0, CAP, FLOOR)
    assert price == CAP
    assert len(dispatch) == 1 and dispatch[0][1] == 50.0


def test_no_offers_is_scarcity():
    price, dispatch = clear_hour([], 100.0, CAP, FLOOR)
    assert price == CAP
    assert dispatch == []


def test_zero_demand_clears_at_cheapest_offer():
    price, dispatch = clear_hour([offer(-25.0, 100.0), offer(40.0, 100.0)], 0.0, CAP, FLOOR)
    assert price == -25.0
    assert dispatch == []


def test_price_clipped_to_floor():
    price, _ = clear_hour([offer(-200.0, 100.0)], 50.0, CAP, FLOOR)
    assert price == FLOOR


def test_exact_supply_demand_balance_is_not_scarcity():
    price, dispatch = clear_hour([offer(35.0, 100.0)], 100.0, CAP, FLOOR)
    assert price == 35.0
    assert dispatch[0][1] == pytest.approx(100.0)
