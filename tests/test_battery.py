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
    assert info.grid_energy_kwh == pytest.approx(20.0)
    assert info.throughput_kwh == pytest.approx(20.0 * math.sqrt(0.9))


def test_discharge_decreases_soc_and_degrades_soh():
    b = make_battery()
    info = b.step(-50.0, dt_h=1.0)
    assert b.soc < 0.5
    assert b.soh < 1.0
    assert info.grid_energy_kwh < 0.0


def test_idle_only_incurs_calendar_aging():
    b = make_battery()
    info = b.step(0.0, dt_h=1.0)
    assert info.cycle_loss == 0.0
    assert info.calendar_loss > 0.0
    assert b.soh < 1.0
    # Idle for 15 years at SoC 0.5 should hit end of life.
    b.reset()
    b.step(0.0, dt_h=15 * 8760.0)
    assert b.soh == pytest.approx(b.config.eol_soh)


def test_calendar_aging_faster_at_high_soc():
    hi = make_battery(initial_soc=0.9)
    lo = make_battery(initial_soc=0.1)
    hi_loss = hi.step(0.0, dt_h=1.0).calendar_loss
    lo_loss = lo.step(0.0, dt_h=1.0).calendar_loss
    assert hi_loss > lo_loss


def test_constraint_clipped_flags_infeasible_requests():
    cfg = BatteryConfig()
    # Full battery asked to charge hard: the SoC bound refuses the request.
    b = make_battery()
    b.reset(soc=cfg.soc_max)
    assert b.step(cfg.max_charge_kw).constraint_clipped
    # Empty battery asked to discharge hard: same, on the floor.
    b.reset(soc=cfg.soc_min)
    assert b.step(-cfg.max_discharge_kw).constraint_clipped
    # A feasible mid-SoC charge, an idle step, and charging exactly to the bound
    # are all feasible: not flagged.
    b.reset(soc=0.5)
    assert not b.step(10.0).constraint_clipped
    assert not b.step(0.0).constraint_clipped
    b.reset(soc=0.5)
    headroom_kwh = (cfg.soc_max - 0.5) * cfg.capacity_kwh
    exact_power = headroom_kwh / b._one_way_eff  # charges precisely to soc_max
    assert not b.step(exact_power).constraint_clipped


def test_cycle_stress_penalizes_high_power():
    # Same 10 kWh of grid energy, delivered gently vs at full power.
    gentle = make_battery()
    hard = make_battery()
    gentle_loss = sum(gentle.step(10.0, dt_h=0.2).cycle_loss for _ in range(5))
    hard_loss = hard.step(50.0, dt_h=0.2).cycle_loss
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
    assert info.grid_energy_kwh == pytest.approx(10.0)


def test_round_trip_loses_energy():
    # Start empty, put 10 kWh in from the grid, then drain fully:
    # the grid gets back round_trip_efficiency of what it paid.
    b = make_battery(initial_soc=0.05)
    charged = b.step(10.0, dt_h=1.0).grid_energy_kwh
    discharged = -b.step(-50.0, dt_h=1.0).grid_energy_kwh
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
    expected = stored_before + info.throughput_kwh
    assert b.energy_stored_kwh == pytest.approx(expected)


def test_nonlinear_knobs_default_to_current_linear_behavior():
    # cycle_stress_exponent=1, knee_gain=0 must reproduce the plain formula.
    b = make_battery()
    info = b.step(50.0, dt_h=1.0)  # 50 kW == max_charge_kw -> power_fraction 1.0
    expected_cycle = info.throughput_kwh * b._soh_loss_per_kwh * (1.0 + b.config.cycle_stress)
    assert info.cycle_loss == pytest.approx(expected_cycle)
    assert info.soh_loss == pytest.approx(info.cycle_loss + info.calendar_loss)


def test_cycle_stress_exponent_is_convex_in_power():
    def rate(exponent: float, power_frac: float) -> float:
        b = make_battery(cycle_stress_exponent=exponent, initial_soc=0.5)
        info = b.step(power_frac * b.config.max_charge_kw, dt_h=0.1)
        return info.cycle_loss / info.throughput_kwh

    # Linear (default exponent=1): equally spaced power fractions give equal
    # forward differences.
    r1, r2, r3 = (rate(1.0, pf) for pf in (1 / 3, 2 / 3, 1.0))
    assert (r2 - r1) == pytest.approx(r3 - r2, rel=1e-6)

    # Quadratic (exponent=2): forward differences strictly increase (convex) --
    # pushing from 2/3 to full power costs more than 1/3 to 2/3.
    q1, q2, q3 = (rate(2.0, pf) for pf in (1 / 3, 2 / 3, 1.0))
    assert (q2 - q1) < (q3 - q2)


def test_knee_gain_accelerates_degradation_near_eol():
    # Same feasible action (unclipped, so throughput_kwh is identical either
    # way), compared fresh vs right at the EoL threshold.
    fresh = make_battery(knee_gain=4.0, knee_exponent=3.0)
    fresh.reset(soc=0.5, soh=1.0)
    fresh_info = fresh.step(10.0, dt_h=0.1)

    worn = make_battery(knee_gain=4.0, knee_exponent=3.0)
    worn.reset(soc=0.5, soh=worn.config.eol_soh)
    worn_info = worn.step(10.0, dt_h=0.1)

    # wear_frac=0 at soh=1.0 (knee_factor=1) vs wear_frac=1 at soh=eol_soh
    # (knee_factor=1+knee_gain=5). Checked on cycle_loss, which (unlike
    # calendar_loss) doesn't also depend on soc's fractional increment --
    # that increment is itself capacity-dependent, so soh_loss only matches
    # the 5x ratio approximately.
    assert worn_info.cycle_loss == pytest.approx(fresh_info.cycle_loss * 5.0, rel=1e-9)
    assert worn_info.soh_loss > fresh_info.soh_loss
