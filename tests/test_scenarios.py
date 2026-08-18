import numpy as np

from energy_storage import ColdSnap, Drought, FuelShock, HeatWave, Market, PlantOutage


def run(days, seed=3, scenarios=None):
    market = Market(seed=seed, scenarios=scenarios)
    return [market.simulate_day() for _ in range(days)]


def test_cold_snap_colder_more_demand_pricier():
    base = run(30)
    cold = run(30, scenarios=[ColdSnap(start_day=10, duration_days=14)])
    window = slice(12, 22)  # fully ramped-in days
    for b, c in zip(base[window], cold[window]):
        assert c.weather.temperature_c.mean() < b.weather.temperature_c.mean() - 8.0
        assert c.demand_mw.mean() > b.demand_mw.mean()
    base_price = np.mean([d.prices.mean() for d in base[window]])
    cold_price = np.mean([d.prices.mean() for d in cold[window]])
    assert cold_price > base_price
    # Untouched outside the window.
    np.testing.assert_array_equal(base[5].prices, cold[5].prices)


def test_heat_wave_raises_demand_in_summer():
    # Must land in summer: extra heat drives cooling load. (In winter it
    # would offset heating and lower demand.)
    base = run(200)
    hot = run(200, scenarios=[HeatWave(start_day=185, duration_days=10)])
    assert hot[190].demand_mw.mean() > base[190].demand_mw.mean()
    assert hot[190].weather.temperature_c.mean() > base[190].weather.temperature_c.mean()


def test_fuel_shock_spikes_then_decays():
    shock = FuelShock(start_day=10, duration_days=10, fuel="gas", magnitude=3.0)
    base = run(60)
    shocked = run(60, scenarios=[shock])
    # During the window gas is exactly 3x the same-seed baseline.
    assert shocked[15].fuel_prices.gas_per_mwh_th == base[15].fuel_prices.gas_per_mwh_th * 3.0
    assert np.mean(shocked[15].prices) > np.mean(base[15].prices)
    # Decays monotonically back towards baseline after the window.
    ratio_mid = shocked[30].fuel_prices.gas_per_mwh_th / base[30].fuel_prices.gas_per_mwh_th
    ratio_late = shocked[55].fuel_prices.gas_per_mwh_th / base[55].fuel_prices.gas_per_mwh_th
    assert 3.0 > ratio_mid > ratio_late > 1.0
    assert ratio_late < 1.2  # ~3.5 half-lives out
    assert "fuel-shock" in shocked[15].active_scenarios
    assert "fuel-shock" not in shocked[55].active_scenarios


def test_drought_curtails_hydro_generation():
    # The reservoir self-stabilizes (scarcity raises the bid, which slows
    # dispatch), so the drought shows up as less hydro generation.
    days = 60
    base = run(days)
    dry = run(days, scenarios=[Drought(start_day=0, duration_days=days)])
    base_hydro = sum(d.generation_mw.get("hydro", np.zeros(24)).sum() for d in base)
    dry_hydro = sum(d.generation_mw.get("hydro", np.zeros(24)).sum() for d in dry)
    assert dry_hydro < base_hydro * 0.75


def test_plant_outage_removes_unit():
    scenarios = [PlantOutage(start_day=5, generator_name="ccgt-1", duration_days=10)]
    market = Market(seed=3, scenarios=scenarios)
    availability = []
    for _ in range(20):
        market.simulate_day()
        unit = next(g for g in market.generators if g.name == "ccgt-1")
        availability.append(unit.available)
    assert not any(availability[5:15])


def test_scenarios_preserve_determinism():
    a = run(10, scenarios=[ColdSnap(start_day=3)])
    b = run(10, scenarios=[ColdSnap(start_day=3)])
    for da, db in zip(a, b):
        np.testing.assert_array_equal(da.prices, db.prices)


def test_active_scenario_labels():
    days = run(10, scenarios=[ColdSnap(start_day=3, duration_days=4)])
    assert days[2].active_scenarios == []
    assert days[4].active_scenarios == ["cold-snap"]
    assert days[8].active_scenarios == []
