"""Battery surrogate tests.

Skipped entirely without the 'train' extra — the surrogates import torch, same
rule as market_model. The point of these tests is the PINN's *hard
admissibility*: no matter the inputs or how little it was trained, it must never
predict a physically impossible transition (rising SoH, wrong-sign grid energy,
SoC out of bounds). That guarantee is the whole reason to prefer it as a world
model an optimizing agent trains against."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from energy_storage import BatteryArbitrageEnv, EnvConfig
from energy_storage.battery import BatteryConfig
from energy_storage import battery_model as bm


@pytest.fixture(scope="module")
def fitted():
    cfg = BatteryConfig()
    x, y = bm.sample_transitions(400, cfg, seed=0)
    mlp = bm.MLPBatteryModel(cfg, hidden=32, seed=1)
    pinn = bm.PINNBatteryModel(cfg, hidden=32, seed=1)
    bm.fit(mlp, x, y, bm.FitConfig(epochs=50), seed=1)
    bm.fit(pinn, x, y, bm.FitConfig(epochs=50), seed=1)
    return cfg, mlp, pinn


def test_sample_transitions_shapes_and_ranges():
    cfg = BatteryConfig()
    x, y = bm.sample_transitions(64, cfg, seed=3)
    assert x.shape == (64, 3) and y.shape == (64, 3)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    soc, soh, pf = x.T
    assert (soc >= cfg.soc_min).all() and (soc <= cfg.soc_max).all()
    assert (soh >= cfg.eol_soh).all() and (soh <= 1.0).all()
    assert (pf >= -1.0).all() and (pf <= 1.0).all()


def test_fit_is_seed_deterministic():
    cfg = BatteryConfig()
    x, y = bm.sample_transitions(128, cfg, seed=0)
    outs = []
    for _ in range(2):
        m = bm.PINNBatteryModel(cfg, hidden=32, seed=7)
        bm.fit(m, x, y, bm.FitConfig(epochs=30), seed=7)
        outs.append(m.forward(torch.from_numpy(x)).detach().numpy())
    np.testing.assert_allclose(outs[0], outs[1])


def test_predict_finite_and_shaped(fitted):
    _, mlp, pinn = fitted
    for m in (mlp, pinn):
        dsoc, dsoh, grid = m.predict(0.5, 0.95, 0.3)
        assert all(np.isfinite(v) for v in (dsoc, dsoh, grid))


def test_pinn_is_physically_admissible_everywhere(fitted):
    """Dense sweep of the whole cube (plus out-of-range inputs): the PINN must
    never rise SoH, flip grid sign, or push SoC past the bounds."""
    cfg, _, pinn = fitted
    rng = np.random.default_rng(0)
    soc = rng.uniform(cfg.soc_min, cfg.soc_max, 4000)
    soh = rng.uniform(cfg.eol_soh, 1.0, 4000)
    pf = rng.uniform(-1.0, 1.0, 4000)
    x = np.stack([soc, soh, pf], axis=1).astype(np.float32)
    with torch.no_grad():
        dsoc, dsoh, grid = pinn.forward(torch.from_numpy(x)).numpy().T
    assert (dsoh <= 0.0).all(), "SoH must be monotone non-increasing"
    charging = pf >= 0.0
    assert (grid[charging] >= 0.0).all() and (grid[~charging] <= 0.0).all()
    assert (dsoc[charging] >= 0.0).all() and (dsoc[~charging] <= 0.0).all()
    next_soc = soc + dsoc
    assert (next_soc >= cfg.soc_min - 1e-6).all()
    assert (next_soc <= cfg.soc_max + 1e-6).all()


def test_learned_battery_round_trips_through_env(fitted):
    """A LearnedBattery drops into the env via battery_factory and produces a
    full, finite episode with SoH monotone and SoC in bounds."""
    cfg, _, pinn = fitted
    config = EnvConfig(
        episode_days=2,
        battery_factory=lambda bc: bm.LearnedBattery(pinn, bc),
    )
    env = BatteryArbitrageEnv(config)
    obs, _ = env.reset(seed=0)
    prev_soh = env.battery.soh
    done = False
    steps = 0
    while not done:
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(reward) and np.isfinite(obs).all()
        assert env.battery.soh <= prev_soh + 1e-9
        assert cfg.soc_min - 1e-6 <= env.battery.soc <= cfg.soc_max + 1e-6
        prev_soh = env.battery.soh
        steps += 1
        done = term or trunc
    assert steps == env.episode_steps


def test_save_load_round_trip(tmp_path, fitted):
    cfg, _, pinn = fitted
    path = tmp_path / "pinn.pt"
    bm.save_model(pinn, path)
    loaded = bm.load_model(path)
    x, _ = bm.sample_transitions(32, cfg, seed=5)
    with torch.no_grad():
        np.testing.assert_allclose(
            pinn.forward(torch.from_numpy(x)).numpy(),
            loaded.forward(torch.from_numpy(x)).numpy(),
        )
