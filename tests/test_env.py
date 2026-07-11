import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from energy_storage import BatteryArbitrageEnv, EnvConfig, FuelShock
from energy_storage.env import N_SCALAR_FEATURES


def test_passes_gymnasium_checker():
    check_env(BatteryArbitrageEnv(), skip_render_check=True)


def test_observation_shape_and_finiteness():
    env = BatteryArbitrageEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    for _ in range(50):
        obs, *_ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs))


def test_reset_seeding_deterministic():
    a = BatteryArbitrageEnv()
    b = BatteryArbitrageEnv()
    obs_a, _ = a.reset(seed=123)
    obs_b, _ = b.reset(seed=123)
    np.testing.assert_array_equal(obs_a, obs_b)
    action = np.array([0.7], dtype=np.float32)
    for _ in range(30):
        oa, ra, *_ = a.step(action)
        ob, rb, *_ = b.step(action)
        np.testing.assert_array_equal(oa, ob)
        assert ra == rb
    obs_c, _ = BatteryArbitrageEnv().reset(seed=124)
    assert not np.array_equal(obs_a, obs_c)


def test_reward_is_scaled_profit_minus_degradation():
    env = BatteryArbitrageEnv()
    env.reset(seed=1)
    for action in ([1.0], [-1.0], [0.0], [0.3]):
        _, reward, _, _, info = env.step(np.array(action, dtype=np.float32))
        expected = (info["profit"] - info["degradation_cost"]) * env.config.reward_scale
        assert reward == pytest.approx(expected)


def test_charging_at_positive_price_costs_money():
    env = BatteryArbitrageEnv()
    env.reset(seed=2)
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    assert info["price"] > 0
    assert info["profit"] < 0


def test_price_window_aligns_with_settlement():
    env = BatteryArbitrageEnv()
    obs, _ = env.reset(seed=3)
    for _ in range(30):
        first_price_obs = obs[N_SCALAR_FEATURES]
        obs, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        assert first_price_obs == pytest.approx(env._norm_price(info["price"]), abs=1e-6)


def test_episode_truncates_at_horizon():
    env = BatteryArbitrageEnv(EnvConfig(episode_days=2))
    env.reset(seed=4)
    steps = 0
    truncated = False
    while not truncated:
        *_, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        assert steps <= 48
    assert steps == 48


def test_idle_battery_still_pays_calendar_degradation():
    env = BatteryArbitrageEnv()
    env.reset(seed=5)
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert info["profit"] == 0.0
    assert info["degradation_cost"] > 0.0


def test_scenario_sampler_applies():
    def sampler(rng, start_day):
        return [FuelShock(start_day=start_day, duration_days=30, magnitude=3.0)]

    env = BatteryArbitrageEnv(EnvConfig(scenario_sampler=sampler))
    env.reset(seed=6)
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert "fuel-shock" in info["active_scenarios"]


def test_initial_state_randomized_across_episodes():
    env = BatteryArbitrageEnv()
    socs, days = [], []
    for _ in range(10):
        env.reset()
        socs.append(env.battery.soc)
        days.append(env.today.day)
    assert len(set(np.round(socs, 6))) > 1
    assert len(set(days)) > 1
