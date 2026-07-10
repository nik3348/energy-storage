import numpy as np
import pytest

from energy_storage import Market


@pytest.fixture(scope="module")
def year():
    market = Market(seed=42)
    return [market.simulate_day() for _ in range(365)]


def stack(year, attr):
    return np.concatenate([getattr(d, attr) for d in year])


def test_prices_within_bounds(year):
    prices = stack(year, "prices")
    cfg = Market().config
    assert prices.min() >= cfg.price_floor
    assert prices.max() <= cfg.price_cap


def test_evening_peak_pricier_than_night(year):
    prices = np.array([d.prices for d in year])
    assert prices[:, 18:21].mean() > prices[:, 2:5].mean() + 5.0


def test_holidays_cheaper(year):
    holiday = np.array([d.prices.mean() for d in year if d.is_holiday])
    workday = np.array([d.prices.mean() for d in year if not d.is_holiday and not d.is_weekend])
    assert holiday.mean() < workday.mean()


def test_negative_prices_rare_but_present(year):
    prices = stack(year, "prices")
    negative_share = (prices < 0).mean()
    assert 0.0 < negative_share < 0.15


def test_scarcity_rare(year):
    cap = Market().config.price_cap
    prices = stack(year, "prices")
    assert (prices >= cap).mean() < 0.015


def test_supply_meets_demand_except_scarcity(year):
    for d in year:
        total_gen = sum(d.generation_mw.values())
        np.testing.assert_allclose(total_gen + d.unserved_mw, d.demand_mw, rtol=1e-6)


def test_solar_zero_at_night(year):
    for d in year:
        if "solar" in d.generation_mw:
            assert d.generation_mw["solar"][:4].sum() == 0.0
            assert d.generation_mw["solar"][22:].sum() == 0.0


def test_windy_hours_cheaper(year):
    prices = stack(year, "prices")
    wind = np.concatenate([d.weather.wind_speed_ms for d in year])
    windy = prices[wind > np.percentile(wind, 80)]
    calm = prices[wind < np.percentile(wind, 20)]
    assert windy.mean() < calm.mean()


def test_reproducible_and_seed_sensitive():
    a = [d.prices for d in (Market(seed=7).simulate_day() for _ in range(10))]
    b = [d.prices for d in (Market(seed=7).simulate_day() for _ in range(10))]
    c = [d.prices for d in (Market(seed=8).simulate_day() for _ in range(10))]
    np.testing.assert_array_equal(np.array(a), np.array(b))
    assert not np.array_equal(np.array(a), np.array(c))


def test_price_level_sane(year):
    prices = stack(year, "prices")
    assert 30.0 < np.median(prices) < 90.0
