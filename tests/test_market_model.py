import numpy as np
import pytest

from energy_storage import BatteryArbitrageEnv, EnvConfig
from energy_storage.env import N_SCALAR_FEATURES, PRICE_WINDOW
from energy_storage.market_history import (
    MarketHistory,
    ReplayedMarket,
    build_dataset,
    denorm_price,
    norm_price,
)


@pytest.fixture(scope="module")
def history():
    return MarketHistory.collect(config=None, seed=11, days=30)


def test_norm_price_matches_env():
    env = BatteryArbitrageEnv()
    cap = env.config.market.price_cap
    prices = np.array([-100.0, -1.0, 0.0, 5.0, 64.0, cap])
    np.testing.assert_allclose(norm_price(prices, cap), env._norm_price(prices))


def test_denorm_inverts_norm():
    prices = np.array([-100.0, -0.5, 0.0, 42.0, 999.9])
    np.testing.assert_allclose(denorm_price(norm_price(prices, 1000.0), 1000.0), prices)


def test_history_roundtrip(tmp_path, history):
    path = tmp_path / "history.npz"
    history.save(path)
    loaded = MarketHistory.load(path)
    np.testing.assert_array_equal(loaded.prices, history.prices)
    np.testing.assert_array_equal(loaded.day_of_year, history.day_of_year)
    np.testing.assert_array_equal(loaded.is_weekend, history.is_weekend)
    np.testing.assert_array_equal(loaded.is_holiday, history.is_holiday)


def test_build_dataset_shapes_and_alignment(history):
    x, y = build_dataset(history, cap=1000.0)
    assert x.shape == (len(history) - 1, 7)
    assert y.shape == (len(history) - 1, 24)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    # Row t targets day t+1's prices, conditioned on day t+1's calendar.
    np.testing.assert_allclose(y[0], norm_price(history.prices[1], 1000.0), rtol=1e-6)


def test_replayed_market_serves_stored_days(history):
    market = ReplayedMarket(history)
    market.day = 3
    result = market.simulate_day()
    np.testing.assert_array_equal(result.prices, history.prices[3])
    assert result.day_of_year == history.day_of_year[3]
    assert market.day == 4
    # Wraps modulo the history length.
    market.day = len(history) + 2
    np.testing.assert_array_equal(market.simulate_day().prices, history.prices[2])


def test_env_round_trip_with_replayed_market(history):
    config = EnvConfig(
        episode_days=2,
        market_factory=lambda market_config, seed: ReplayedMarket(history, market_config),
    )
    env = BatteryArbitrageEnv(config)
    obs, _ = env.reset(seed=5)
    assert obs.shape == (N_SCALAR_FEATURES + PRICE_WINDOW,)
    # The first observed price is a stored day's hour-0 price.
    day0 = env.market.day - 2  # reset simulated burn-in, today, tomorrow
    stored = history.prices[(day0) % len(history)]
    np.testing.assert_allclose(
        obs[N_SCALAR_FEATURES], norm_price(stored[0], env.config.market.price_cap), rtol=1e-5
    )
    for _ in range(48):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs))
    assert truncated and not terminated


torch = pytest.importorskip("torch")

from energy_storage.market_model import (  # noqa: E402
    LearnedMarket,
    MarketModelEnsemble,
)


@pytest.fixture(scope="module")
def fitted_ensemble(history):
    x, y = build_dataset(history, cap=1000.0)
    ensemble = MarketModelEnsemble(k=2, hidden=32, seed=0)
    ensemble.fit(x, y, epochs=30, seed=0)
    return ensemble


def test_ensemble_fit_and_sample(fitted_ensemble, history):
    x, _ = build_dataset(history, cap=1000.0)
    sample = fitted_ensemble.sample_day(x[0], np.random.default_rng(0), member=0)
    assert sample.shape == (24,)
    assert np.all(np.isfinite(sample))
    assert fitted_ensemble.disagreement(x[0]) >= 0.0


def test_ensemble_sampling_deterministic(fitted_ensemble, history):
    x, _ = build_dataset(history, cap=1000.0)
    a = fitted_ensemble.sample_day(x[0], np.random.default_rng(7), member=1)
    b = fitted_ensemble.sample_day(x[0], np.random.default_rng(7), member=1)
    np.testing.assert_array_equal(a, b)


def test_ensemble_save_load_roundtrip(tmp_path, fitted_ensemble, history):
    path = tmp_path / "ensemble.pt"
    fitted_ensemble.save(path)
    loaded = MarketModelEnsemble.load(path)
    x, _ = build_dataset(history, cap=1000.0)
    a = fitted_ensemble.sample_day(x[0], np.random.default_rng(3), member=0)
    b = loaded.sample_day(x[0], np.random.default_rng(3), member=0)
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_learned_market_produces_valid_days(fitted_ensemble):
    market = LearnedMarket(fitted_ensemble, seed=0)
    market.day = -3
    for expected_day in range(-3, 4):
        result = market.simulate_day()
        assert result.day == expected_day
        assert result.prices.shape == (24,)
        assert np.all(result.prices >= market.config.price_floor)
        assert np.all(result.prices <= market.config.price_cap)
        assert result.day_of_year == expected_day % 365


def test_env_round_trip_with_learned_market(fitted_ensemble):
    config = EnvConfig(
        episode_days=2,
        market_factory=lambda market_config, seed: LearnedMarket(
            fitted_ensemble, market_config, seed=seed
        ),
    )
    env = BatteryArbitrageEnv(config)
    obs, _ = env.reset(seed=9)
    assert isinstance(env.market, LearnedMarket)
    truncated = False
    while not truncated:
        obs, reward, _, truncated, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)


def test_learned_market_seed_determinism(fitted_ensemble):
    days_a = [LearnedMarket(fitted_ensemble, seed=4).simulate_day().prices for _ in range(1)]
    days_b = [LearnedMarket(fitted_ensemble, seed=4).simulate_day().prices for _ in range(1)]
    days_c = [LearnedMarket(fitted_ensemble, seed=5).simulate_day().prices for _ in range(1)]
    np.testing.assert_array_equal(days_a[0], days_b[0])
    assert not np.array_equal(days_a[0], days_c[0])
