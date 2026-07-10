import math

import pytest

from energy_storage import Battery, BatteryConfig


def make_battery(**overrides):
    return Battery(BatteryConfig(**overrides))


def test_charge_increases_soc_and_degrades_soh():
    b = make_battery()
    info = b.step(20.0, dt_h=1.0)
    assert b.soc > 0.5
    assert b.soh < 1.0
    assert info["grid_energy_kwh"] == pytest.approx(20.0)
    assert info["throughput_kwh"] == pytest.approx(20.0 * math.sqrt(0.9))


def test_discharge_decreases_soc_and_degrades_soh():
    b = make_battery()
    info = b.step(-50.0, dt_h=1.0)
    assert b.soc < 0.5
    assert b.soh < 1.0
    assert info["grid_energy_kwh"] < 0.0


def test_idle_only_incurs_calendar_aging():
    b = make_battery()
    info = b.step(0.0, dt_h=1.0)
    assert info["cycle_loss"] == 0.0
    assert info["calendar_loss"] > 0.0
    assert b.soh < 1.0
    # Idle for 15 years at SoC 0.5 should hit end of life.
    b.reset()
    b.step(0.0, dt_h=15 * 8760.0)
    assert b.soh == pytest.approx(b.config.eol_soh)


def test_calendar_aging_faster_at_high_soc():
    hi = make_battery(initial_soc=0.9)
    lo = make_battery(initial_soc=0.1)
    hi_loss = hi.step(0.0, dt_h=1.0)["calendar_loss"]
    lo_loss = lo.step(0.0, dt_h=1.0)["calendar_loss"]
    assert hi_loss > lo_loss


def test_cycle_stress_penalizes_high_power():
    # Same 10 kWh of grid energy, delivered gently vs at full power.
    gentle = make_battery()
    hard = make_battery()
    gentle_loss = sum(gentle.step(10.0, dt_h=0.2)["cycle_loss"] for _ in range(5))
    hard_loss = hard.step(50.0, dt_h=0.2)["cycle_loss"]
    assert gentle.soc == pytest.approx(hard.soc, rel=1e-4)
    assert hard_loss > gentle_loss


def test_soc_respects_bounds():
    b = make_battery()
    for _ in range(100):
        b.step(50.0, dt_h=1.0)
    assert b.soc <= b.config.soc_max + 1e-9
    for _ in range(100):
        b.step(-50.0, dt_h=1.0)
    assert b.soc >= b.config.soc_min - 1e-9


def test_power_is_clipped_to_limits():
    b = make_battery(max_charge_kw=10.0)
    info = b.step(1000.0, dt_h=1.0)
    assert info["grid_energy_kwh"] == pytest.approx(10.0)


def test_round_trip_loses_energy():
    # Start empty, put 10 kWh in from the grid, then drain fully:
    # the grid gets back round_trip_efficiency of what it paid.
    b = make_battery(initial_soc=0.05)
    charged = b.step(10.0, dt_h=1.0)["grid_energy_kwh"]
    discharged = -b.step(-50.0, dt_h=1.0)["grid_energy_kwh"]
    assert discharged < charged
    assert discharged / charged == pytest.approx(0.9, rel=1e-3)


def test_soh_floors_at_eol():
    b = make_battery(cycle_life=1.0)  # degrade fast
    for _ in range(50):
        b.step(50.0, dt_h=1.0)
        b.step(-50.0, dt_h=1.0)
    assert b.soh == pytest.approx(b.config.eol_soh)


def test_degradation_shrinks_effective_capacity():
    b = make_battery()
    b.step(50.0, dt_h=1.0)
    assert b.effective_capacity_kwh < b.config.capacity_kwh


def test_energy_conserved_through_degradation():
    b = make_battery()
    stored_before = 0.5 * 100.0
    b.step(0.0)  # idle: calendar aging shrinks capacity but conserves energy
    assert b.energy_stored_kwh == pytest.approx(stored_before)
    info = b.step(10.0, dt_h=1.0)
    expected = stored_before + info["throughput_kwh"]
    assert b.energy_stored_kwh == pytest.approx(expected)
