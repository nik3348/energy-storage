import numpy as np
import pytest

from energy_storage import Battery, BatteryConfig, EnvConfig
from energy_storage.baselines import collect_episode, heuristic_policy
from energy_storage.oracle import (
    RollingHorizonOracle,
    SocGridPlanner,
    make_hindsight_policy,
)


def net(traj) -> float:
    return float(traj["profit"].sum() - traj["degradation_cost"].sum())


def test_planner_economics_match_battery_step():
    cfg = BatteryConfig()
    planner = SocGridPlanner(cfg, replacement_cost_per_kwh=250.0)
    for soc_from, soc_to in [(0.5, 0.9), (0.5, 0.52), (0.9, 0.5), (0.3, 0.05), (0.4, 0.4)]:
        energy, deg_cost, feasible = planner.step_economics(soc_from, soc_to, soh=1.0)
        assert feasible
        battery = Battery(cfg)
        battery.reset(soc=soc_from)
        result = battery.step(float(energy), dt_h=1.0)
        assert result.grid_energy_kwh == pytest.approx(float(energy), abs=1e-9)
        # SoC lands on target modulo the post-step re-expression (tiny).
        assert battery.soc == pytest.approx(soc_to, abs=1e-4)
        assert result.soh_loss * 250.0 * cfg.capacity_kwh == pytest.approx(
            float(deg_cost), rel=1e-9
        )


def test_planner_power_limits_are_infeasible():
    cfg = BatteryConfig()  # 50 kW limits, 100 kWh
    planner = SocGridPlanner(cfg, replacement_cost_per_kwh=250.0)
    _, _, feasible = planner.step_economics(0.05, 0.95, soh=1.0)  # needs ~95 kW
    assert not feasible
    _, _, feasible = planner.step_economics(0.95, 0.05, soh=1.0)
    assert not feasible


def test_staying_put_is_always_feasible():
    planner = SocGridPlanner(BatteryConfig(), replacement_cost_per_kwh=250.0)
    g = planner.soc_grid
    _, _, feasible = planner.step_economics(g, g, soh=0.9)
    assert feasible.all()


CONFIG = EnvConfig(episode_days=3)
SEEDS = range(5)


def test_rolling_oracle_beats_heuristic_paired():
    oracle_net = sum(net(collect_episode(RollingHorizonOracle(), CONFIG, s)) for s in SEEDS)
    heuristic_net = sum(net(collect_episode(heuristic_policy, CONFIG, s)) for s in SEEDS)
    assert oracle_net > heuristic_net


def test_hindsight_bounds_rolling_paired():
    for seed in SEEDS:
        hindsight = net(collect_episode(make_hindsight_policy(CONFIG, seed), CONFIG, seed))
        rolling = net(collect_episode(RollingHorizonOracle(), CONFIG, seed))
        # Full-information optimum; small slack for the SoC grid quantization.
        assert hindsight >= rolling - 0.5
