"""Tests for the adaptation streaming harness and nightly model fine-tuning."""

import numpy as np
import pytest

from energy_storage.adaptation import DayRecord, PolicyArm, run_stream
from energy_storage.baselines import heuristic_policy, idle_policy
from energy_storage.env import EnvConfig
from energy_storage.market.scenarios import FuelShock
from energy_storage.market_history import MarketHistory, build_dataset, norm_price


class RecordingArm(PolicyArm):
    """Frozen policy that keeps every settled day and logs a nightly value."""

    def __init__(self, policy_fn):
        super().__init__(policy_fn)
        self.days: list[DayRecord] = []

    def end_of_day(self, record):
        self.days.append(record)
        if record.index % 2 == 0:  # exercise NaN padding for skipped days
            return {"mean_price": record.prices.mean()}
        return None


def _config(days: int, sampler=None) -> EnvConfig:
    return EnvConfig(episode_days=days, scenario_sampler=sampler)


def test_stream_shapes_and_determinism():
    result_a = run_stream(PolicyArm(heuristic_policy), _config(4), seed=7)
    result_b = run_stream(PolicyArm(heuristic_policy), _config(4), seed=7)
    assert result_a.net.shape == (4,)
    np.testing.assert_allclose(result_a.net, result_b.net)


def test_stream_extras_padding():
    arm = RecordingArm(idle_policy)
    result = run_stream(arm, _config(5), seed=3)
    trace = result.extras["mean_price"]
    assert trace.shape == (5,)
    assert not np.isnan(trace[0]) and np.isnan(trace[1])
    assert len(arm.days) == 5
    assert all(len(day.transitions) == 24 for day in arm.days)


def test_stream_prices_are_policy_independent():
    """Price-taker pairing: two arms on the same seed see identical days."""
    idle = RecordingArm(idle_policy)
    busy = RecordingArm(heuristic_policy)
    run_stream(idle, _config(3), seed=11)
    run_stream(busy, _config(3), seed=11)
    for day_i, day_b in zip(idle.days, busy.days):
        np.testing.assert_allclose(day_i.prices, day_b.prices)


def test_stream_scenario_pairs_outside_window():
    """A persistent shift starting at stream day 2 leaves days 0-1 untouched
    and changes prices afterwards (same seed, scenario consumes no rng)."""
    calm = RecordingArm(idle_policy)
    shifted = RecordingArm(idle_policy)
    sampler = lambda rng, start: [
        FuelShock(start_day=start + 2, duration_days=30, magnitude=3.0)
    ]
    run_stream(calm, _config(6), seed=5)
    run_stream(shifted, _config(6, sampler), seed=5)
    for d in range(2):
        np.testing.assert_allclose(calm.days[d].prices, shifted.days[d].prices)
    post_calm = np.concatenate([day.prices for day in calm.days[2:]])
    post_shift = np.concatenate([day.prices for day in shifted.days[2:]])
    assert not np.allclose(post_calm, post_shift)
    assert post_shift.mean() > post_calm.mean()  # 3x gas must raise prices


def test_history_append_and_tail():
    history = MarketHistory(
        prices=np.arange(48, dtype=float).reshape(2, 24),
        day_of_year=np.array([10, 11]),
        is_weekend=np.array([False, True]),
        is_holiday=np.array([False, False]),
    )
    history.append_day(12, False, True, np.full(24, 99.0))
    assert len(history) == 3
    assert history.day_of_year[-1] == 12 and history.is_holiday[-1]
    tail = history.tail(2)
    assert len(tail) == 2
    np.testing.assert_allclose(tail.prices[-1], 99.0)
    tail.prices[:] = 0.0  # tail is a copy, not a view
    np.testing.assert_allclose(history.prices[-1], 99.0)


def _synthetic_dataset(level: float, n: int, seed: int):
    """Flat-ish normalized days around `level` with mild noise."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 7)).astype(np.float32) * 0.3
    y = (level + 0.02 * rng.standard_normal((n, 24))).astype(np.float32)
    return x, y


def test_fine_tune_tracks_a_shifted_regime():
    pytest.importorskip("torch")
    from energy_storage.market_model import MarketModelEnsemble

    ensemble = MarketModelEnsemble(k=2, hidden=16, seed=0)
    x_calm, y_calm = _synthetic_dataset(0.2, n=40, seed=1)
    ensemble.fit(x_calm, y_calm, epochs=200, seed=2)

    x_shift, y_shift = _synthetic_dataset(0.6, n=20, seed=3)
    nll_before = np.mean([ensemble.day_nll(x, y) for x, y in zip(x_shift, y_shift)])
    ensemble.fine_tune(x_shift, y_shift, epochs=200, seed=4)
    nll_after = np.mean([ensemble.day_nll(x, y) for x, y in zip(x_shift, y_shift)])
    assert nll_after < nll_before


def test_day_nll_flags_out_of_distribution_days():
    pytest.importorskip("torch")
    from energy_storage.market_model import MarketModelEnsemble

    ensemble = MarketModelEnsemble(k=2, hidden=16, seed=0)
    x, y = _synthetic_dataset(0.2, n=40, seed=1)
    ensemble.fit(x, y, epochs=200, seed=2)
    in_dist = ensemble.day_nll(x[0], y[0])
    out_dist = ensemble.day_nll(x[0], y[0] + 0.4)
    assert np.isfinite(in_dist)
    assert out_dist > in_dist


def test_build_dataset_matches_norm():
    history = MarketHistory.collect(None, seed=0, days=3)
    x, y = build_dataset(history, cap=1000.0)
    assert x.shape == (2, 7) and y.shape == (2, 24)
    np.testing.assert_allclose(
        y[0], norm_price(history.prices[1], 1000.0), rtol=1e-6
    )
