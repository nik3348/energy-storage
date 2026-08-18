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


class _CrashAfterNDays(RecordingArm):
    """Simulates a killed process: raises inside end_of_day after N calls,
    after the wrapped arm has already done its (recorded) work — matching
    where a real crash lands relative to run_stream's checkpoint write."""

    def __init__(self, policy_fn, crash_after: int):
        super().__init__(policy_fn)
        self._crash_after = crash_after

    def end_of_day(self, record):
        logs = super().end_of_day(record)
        if len(self.days) == self._crash_after:
            raise RuntimeError("simulated crash")
        return logs


def test_run_stream_resume_reproduces_prices_and_results(tmp_path):
    """The core resume guarantee: a crash-then-resume run must see exactly
    the same prices (and thus results) as an uninterrupted run at the same
    seed — paired-seed comparisons depend on this regardless of which arm
    is running."""
    config = _config(8)
    baseline = run_stream(PolicyArm(heuristic_policy), config, seed=99)

    ckpt_dir = tmp_path / "ckpt"
    crasher = _CrashAfterNDays(heuristic_policy, crash_after=3)
    with pytest.raises(RuntimeError):
        run_stream(crasher, config, seed=99, checkpoint_dir=ckpt_dir)
    assert (ckpt_dir / "progress.npz").exists()

    resumed = run_stream(PolicyArm(heuristic_policy), config, seed=99, checkpoint_dir=ckpt_dir)
    assert not ckpt_dir.exists()  # cleaned up on completion
    np.testing.assert_allclose(resumed.net, baseline.net)
    np.testing.assert_allclose(resumed.profit, baseline.profit)


def test_adaptive_dyna_resume_after_crash_matches_prices(tmp_path):
    """Smoke test for the expensive arm: resume must not crash, must produce
    a complete stream, and — the invariant that actually matters for paired
    comparisons — prices must be unaffected by the crash/resume regardless
    of how the arm's own (stochastic) SAC fine-tuning continues."""
    pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_util import make_vec_env

    from energy_storage.adaptation import AdaptiveDynaArm, AdaptiveDynaConfig
    from energy_storage.env import BatteryArbitrageEnv
    from energy_storage.market_history import build_dataset
    from energy_storage.market_model import MarketModelEnsemble

    stream_config = _config(6)
    sac_env = make_vec_env(
        lambda: BatteryArbitrageEnv(stream_config), n_envs=1, seed=0
    )
    model = sb3.SAC(
        "MlpPolicy", sac_env, seed=0, learning_starts=0, buffer_size=500,
        policy_kwargs={"net_arch": [8, 8]},
    )
    model.learn(total_timesteps=20)
    sac_path = tmp_path / "sac.zip"
    model.save(sac_path)

    history = MarketHistory.collect(None, seed=0, days=30)
    x, y = build_dataset(history, cap=1000.0)
    ensemble = MarketModelEnsemble(k=2, hidden=8, seed=0)
    ensemble.fit(x, y, epochs=10, seed=0)
    ensemble_path = tmp_path / "ensemble.pt"
    ensemble.save(ensemble_path)

    cfg = AdaptiveDynaConfig(
        window_days=10, anchor_days=5, ft_epochs=2, sac_steps=20,
        demo_days=3, buffer_size=500, imagination_episode_days=4,
    )

    def make_arm():
        return AdaptiveDynaArm(
            sac_path, ensemble_path, history, stream_config, cfg, seed=0
        )

    baseline = run_stream(make_arm(), stream_config, seed=123)

    class CrashingAdaptiveDyna(AdaptiveDynaArm):
        def end_of_day(self, record):
            logs = super().end_of_day(record)
            if record.index == 2:
                raise RuntimeError("simulated crash")
            return logs

    ckpt_dir = tmp_path / "ckpt"
    crasher = CrashingAdaptiveDyna(
        sac_path, ensemble_path, history, stream_config, cfg, seed=0
    )
    with pytest.raises(RuntimeError):
        run_stream(crasher, stream_config, seed=123, checkpoint_dir=ckpt_dir)

    resumed_result = run_stream(
        make_arm(), stream_config, seed=123, checkpoint_dir=ckpt_dir
    )
    assert not ckpt_dir.exists()
    assert resumed_result.net.shape == baseline.net.shape == (6,)
    # nll is a pure function of (arm-independent) prices + the fine-tuned
    # ensemble; days 0-2 are loaded verbatim from the checkpoint, so they
    # must match exactly regardless of what happens post-resume.
    np.testing.assert_allclose(
        resumed_result.extras["nll"][:3], baseline.extras["nll"][:3]
    )
    np.testing.assert_allclose(resumed_result.net[:3], baseline.net[:3])


def test_build_dataset_matches_norm():
    history = MarketHistory.collect(None, seed=0, days=3)
    x, y = build_dataset(history, cap=1000.0)
    assert x.shape == (2, 7) and y.shape == (2, 24)
    np.testing.assert_allclose(
        y[0], norm_price(history.prices[1], 1000.0), rtol=1e-6
    )
