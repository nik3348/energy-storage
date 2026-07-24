"""Learned battery surrogates — the PINN sample-efficiency experiment.

Companion to the Dyna market model (docs/pinn-battery-design.md). Where the
Dyna pipeline holds the battery *exact* and learns the market, this module does
the mirror image: it holds the market exact and learns the **battery** from a
log of transitions, to ask whether a physics-informed surrogate (PINN) needs
fewer real samples than a plain MLP — both as a predictive model and, downstream,
as the world model a SAC agent trains against.

A surrogate maps the env's own step signature,

    (soc, soh, power_frac)  ->  (dsoc, dsoh, grid_energy_kwh)

with `power_frac in [-1, 1]` the action (fraction of max power) and `dt = 1h`
fixed. `soh_loss = -dsoh`. `LearnedBattery` wraps a fitted surrogate to
duck-type `Battery` (reset / soc / soh / step) so the env can train against it
with no observation or policy change.

Two surrogates share a trunk and parameter count; the only difference is the
physics (see the two classes). Both are compared on held-out accuracy vs the
number of training transitions.

Imports torch: only usable under the 'train' extra (same rule as market_model.py
and viz.py) — the core package must not import this module.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from energy_storage.battery import Battery, BatteryConfig, BatteryStepResult

N_FEATURES = 3  # soc, soh, power_frac
N_TARGETS = 3  # dsoc, dsoh, grid_energy_kwh


# --------------------------------------------------------------------------- #
# Transition dataset from the exact battery
# --------------------------------------------------------------------------- #
def _power_frac_to_kw(power_frac: float, cfg: BatteryConfig) -> float:
    return power_frac * (cfg.max_charge_kw if power_frac >= 0.0 else cfg.max_discharge_kw)


def sample_transitions(
    n: int, cfg: BatteryConfig, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """`n` labelled transitions drawn uniformly across the operating envelope.

    Inputs `(soc, soh, power_frac)` are sampled uniformly over
    `soc in [soc_min, soc_max]`, `soh in [eol_soh, 1]`, `power_frac in [-1, 1]`;
    labels `(dsoc, dsoh, grid_energy_kwh)` come from the exact `Battery.step`.
    Returns `(X (n,3), Y (n,3))`, both float32."""
    rng = np.random.default_rng(seed)
    soc = rng.uniform(cfg.soc_min, cfg.soc_max, n)
    soh = rng.uniform(cfg.eol_soh, 1.0, n)
    pfrac = rng.uniform(-1.0, 1.0, n)

    battery = Battery(cfg)
    x = np.empty((n, N_FEATURES), dtype=np.float32)
    y = np.empty((n, N_TARGETS), dtype=np.float32)
    for i in range(n):
        battery.reset(soc=float(soc[i]), soh=float(soh[i]))
        res = battery.step(_power_frac_to_kw(float(pfrac[i]), cfg), dt_h=1.0)
        x[i] = (soc[i], soh[i], pfrac[i])
        # dsoc/dsoh are the *net applied* change (dsoh already folds in the EoL
        # clamp and the SoC re-expression on capacity fade — everything the env
        # would observe).
        y[i] = (res.soc - soc[i], res.soh - soh[i], res.grid_energy_kwh)
    return x, y


def _target_scales(y: np.ndarray) -> np.ndarray:
    """Per-target std used to balance the data MSE across the wildly different
    target magnitudes (dsoh ~ 1e-4 vs grid_energy ~ 50)."""
    return np.maximum(y.std(axis=0), 1e-8).astype(np.float32)


# --------------------------------------------------------------------------- #
# Feature/trunk shared by both surrogates
# --------------------------------------------------------------------------- #
def _normalize_features(x: torch.Tensor, cfg: BatteryConfig) -> torch.Tensor:
    """Center soc/soh into ~[-1,1] with the known nameplate ranges; power_frac
    is already in [-1,1]. (Ranges are nameplate metadata, not learned dynamics.)"""
    soc_mid = 0.5 * (cfg.soc_min + cfg.soc_max)
    soc_half = 0.5 * (cfg.soc_max - cfg.soc_min)
    soh_mid = 0.5 * (cfg.eol_soh + 1.0)
    soh_half = 0.5 * (1.0 - cfg.eol_soh)
    soc, soh, pf = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return torch.cat(
        [(soc - soc_mid) / soc_half, (soh - soh_mid) / soh_half, pf], dim=1
    )


class _Trunk(nn.Module):
    def __init__(self, hidden: int, out: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _BatterySurrogate(nn.Module):
    """Shared machinery: feature normalization, target scaling buffers, and a
    numpy `predict` used by `LearnedBattery`. Subclasses implement `raw` (the
    physical-space (dsoc, dsoh, grid) prediction from normalized features)."""

    def __init__(self, cfg: BatteryConfig, hidden: int, seed: int):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("target_scale", torch.ones(N_TARGETS))

    def set_target_scale(self, scale: np.ndarray) -> None:
        self.target_scale.copy_(torch.as_tensor(scale, dtype=torch.float32))

    def raw(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """(dsoc, dsoh, grid_energy) in physical units. `z` normalized features,
        `x` raw features (needed for the physical soc/power)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.raw(_normalize_features(x, self.cfg), x)

    @torch.no_grad()
    def predict(self, soc: float, soh: float, power_frac: float) -> tuple[float, float, float]:
        x = torch.tensor([[soc, soh, power_frac]], dtype=torch.float32)
        out = self.forward(x)[0]
        return float(out[0]), float(out[1]), float(out[2])


# --------------------------------------------------------------------------- #
# MLP baseline: three free heads, data MSE only, no physics.
# --------------------------------------------------------------------------- #
class MLPBatteryModel(_BatterySurrogate):
    def __init__(self, cfg: BatteryConfig | None = None, hidden: int = 64, seed: int = 0):
        cfg = cfg or BatteryConfig()
        super().__init__(cfg, hidden, seed)
        self.trunk = _Trunk(hidden, N_TARGETS, seed)

    def raw(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Outputs are predicted in scaled target space, then un-scaled to
        # physical units so `forward`/`predict` are comparable to the PINN.
        return self.trunk(z) * self.target_scale


# --------------------------------------------------------------------------- #
# PINN: hard admissibility (sign / monotonic SoH / SoC saturation) in the
# architecture, plus label-free physics residuals with trainable physical
# scalars (efficiency, capacity, aging rates).
# --------------------------------------------------------------------------- #
class PINNBatteryModel(_BatterySurrogate):
    """Same trunk, heads and parameter budget as the MLP; the only difference is
    that physics enters as **hard admissibility structure** on the outputs, so
    the network can never represent a physically impossible transition:

    - **SoH monotonicity**: `dsoh = -softplus(.) <= 0` — SoH never rises.
    - **Sign consistency**: charge draws from the grid and raises SoC, discharge
      delivers to it and lowers SoC — `sign(grid) = sign(dsoc) = sign(power)`,
      enforced by non-negative magnitude heads gated by the input power's sign.
    - **SoC saturation**: `dsoc` is capped at the available head/foot-room, so
      SoC can never leave `[soc_min, soc_max]`.

    No shared physical parameters and no soft residual: a residual between free
    predictions has a degenerate "shrink everything" minimizer, and fully
    deriving grid/soh_loss from a single magnitude makes the multiplicative
    parameter identification brittle across the 1e5 spread in target scales.
    Admissibility is the robust prior — it costs no expressiveness (three heads,
    like the MLP) and simply forbids the impossible."""

    def __init__(self, cfg: BatteryConfig | None = None, hidden: int = 64, seed: int = 0):
        cfg = cfg or BatteryConfig()
        super().__init__(cfg, hidden, seed)
        self.trunk = _Trunk(hidden, N_TARGETS, seed)
        # Known nameplate capacity (spec sheet) makes the soft energy residual
        # well-posed — with cap fixed it identifies eta rather than letting
        # eta/cap wander along a degenerate ratio. Everything *dynamic*
        # (efficiency, aging rates) is unknown and learned, and used only by
        # `physics_residual` (a soft constraint, weighted by lambda at fit time).
        self.register_buffer("cap", torch.tensor(float(cfg.capacity_kwh)))
        self.logit_eta = nn.Parameter(torch.tensor(2.0))  # sigmoid -> ~0.88 one-way
        self.log_kcyc = nn.Parameter(torch.tensor(-13.0))
        self.log_kcal = nn.Parameter(torch.tensor(-13.0))
        self.log_stress = nn.Parameter(torch.tensor(0.0))

    @property
    def eta(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_eta)

    def raw(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        soc, pf = x[:, 0:1], x[:, 2:3]
        heads = self.trunk(z)
        mag_soc = nn.functional.softplus(heads[:, 0:1]) * self.target_scale[0]
        mag_grid = nn.functional.softplus(heads[:, 1:2]) * self.target_scale[2]
        mag_soh = nn.functional.softplus(heads[:, 2:3]) * self.target_scale[1]

        sign_p = torch.sign(pf)
        charging = (pf >= 0.0).to(x.dtype)
        room = charging * (cfg.soc_max - soc) + (1.0 - charging) * (soc - cfg.soc_min)
        dsoc = sign_p * torch.minimum(mag_soc, room.clamp_min(0.0))
        grid = sign_p * mag_grid  # >0 charging, <0 discharging
        dsoh = -mag_soh  # <= 0 always
        return torch.cat([dsoc, dsoh, grid], dim=1)

    def physics_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Soft, label-free residuals on collocation points `x` — the energy /
        efficiency conservation identity and the degradation structural
        equation, both scale-normalized. Weighted by `FitConfig.physics_weight`
        and evaluable anywhere in the envelope, so at very low labelled N they
        constrain the regions no label reaches. The heads stay free (hard
        admissibility already fixes signs/monotonicity); the residual only pulls
        their magnitudes onto the physical manifold."""
        out = self.forward(x)
        dsoc, dsoh, grid = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        soc, soh, pf = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        charging = (pf >= 0.0).to(x.dtype)
        eta = self.eta
        stored = dsoc.abs() * self.cap * soh  # throughput, kWh
        # Charge draws stored/eta; discharge delivers stored*eta.
        expected_grid = stored * (charging / eta + (1.0 - charging) * eta)
        res_energy = (grid.abs() - expected_grid) / self.target_scale[2]

        kcyc, kcal, stress = (
            torch.exp(self.log_kcyc),
            torch.exp(self.log_kcal),
            torch.exp(self.log_stress),
        )
        soh_loss_phys = stored * kcyc * (1.0 + stress * pf.abs()) + kcal * (0.5 + soc)
        res_deg = ((-dsoh) - soh_loss_phys) / self.target_scale[1]
        return torch.cat([res_energy, res_deg], dim=1)


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
@dataclass
class FitConfig:
    epochs: int = 400
    batch_size: int = 64
    lr: float = 3e-3
    # PINN soft-constraint weight (lambda) and collocation-batch size. lambda=0
    # trains the pure hard-admissibility PINN (residual off); >0 adds the
    # label-free physics residual. Ignored by the MLP.
    physics_weight: float = 0.0
    n_collocation: int = 256
    # Separate (usually faster) LR for the PINN's few physical parameters
    # (eta, aging rates): with the residual on they must be identified quickly,
    # and a shared LR either stalls them or destabilizes the net. None => lr.
    lr_physics: float | None = None


def fit(
    model: _BatterySurrogate,
    x: np.ndarray,
    y: np.ndarray,
    config: FitConfig | None = None,
    seed: int = 0,
) -> float:
    """Fit a surrogate with a scale-balanced data MSE, plus (PINN only, when
    `physics_weight > 0`) a soft physics residual on fresh collocation points
    each step. Returns the final total loss."""
    config = config or FitConfig()
    model.set_target_scale(_target_scales(y))
    x_t, y_t = torch.from_numpy(x), torch.from_numpy(y)
    rng = np.random.default_rng(seed)
    cfg = model.cfg
    use_physics = (
        isinstance(model, PINNBatteryModel) and config.physics_weight > 0.0
    )
    if use_physics and config.lr_physics is not None:
        phys = [model.logit_eta, model.log_kcyc, model.log_kcal, model.log_stress]
        phys_ids = {id(p) for p in phys}
        net = [p for p in model.parameters() if id(p) not in phys_ids]
        opt = torch.optim.Adam(
            [{"params": net, "lr": config.lr},
             {"params": phys, "lr": config.lr_physics}]
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss = torch.tensor(0.0)
    for _ in range(config.epochs):
        order = torch.from_numpy(rng.permutation(len(x_t)))
        for start in range(0, len(x_t), config.batch_size):
            batch = order[start : start + config.batch_size]
            pred = model.forward(x_t[batch])
            loss = (((pred - y_t[batch]) / model.target_scale) ** 2).mean()
            if use_physics:
                col = np.empty((config.n_collocation, N_FEATURES), dtype=np.float32)
                col[:, 0] = rng.uniform(cfg.soc_min, cfg.soc_max, config.n_collocation)
                col[:, 1] = rng.uniform(cfg.eol_soh, 1.0, config.n_collocation)
                col[:, 2] = rng.uniform(-1.0, 1.0, config.n_collocation)
                res = model.physics_residual(torch.from_numpy(col))
                loss = loss + config.physics_weight * (res**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return float(loss.detach())


# --------------------------------------------------------------------------- #
# LearnedBattery: a fitted surrogate wrapped as a Battery stand-in.
# --------------------------------------------------------------------------- #
class LearnedBattery:
    """Duck-types `Battery` (reset / soc / soh / step) using a fitted surrogate.

    The env reads only `grid_energy_kwh`, `soh_loss`, and the updated soc/soh
    from a step; the remaining `BatteryStepResult` fields are filled with
    physically-consistent best-effort values (never read for reward). SoC/SoH
    are clamped to the nameplate bounds so a raw MLP can't leave the envelope."""

    def __init__(self, model: _BatterySurrogate, config: BatteryConfig | None = None):
        self.config = config or model.cfg
        self.model = model.eval()
        self.reset()

    def reset(self, soc: float | None = None, soh: float = 1.0) -> None:
        self.soc = self.config.initial_soc if soc is None else soc
        self.soh = soh

    @property
    def effective_capacity_kwh(self) -> float:
        return self.config.capacity_kwh * self.soh

    @property
    def energy_stored_kwh(self) -> float:
        return self.soc * self.effective_capacity_kwh

    def step(self, power_kw: float, dt_h: float = 1.0) -> BatteryStepResult:
        cfg = self.config
        power_kw = max(-cfg.max_discharge_kw, min(cfg.max_charge_kw, power_kw))
        max_power = cfg.max_charge_kw if power_kw >= 0.0 else cfg.max_discharge_kw
        pfrac = power_kw / max_power if max_power > 0.0 else 0.0

        dsoc, dsoh, grid_energy_kwh = self.model.predict(self.soc, self.soh, pfrac)
        soh_loss = max(0.0, -dsoh)
        throughput_kwh = abs(dsoc) * self.effective_capacity_kwh

        self.soc = float(np.clip(self.soc + dsoc, cfg.soc_min, cfg.soc_max))
        self.soh = max(cfg.eol_soh, self.soh - soh_loss)
        return BatteryStepResult(
            grid_energy_kwh=grid_energy_kwh,
            throughput_kwh=throughput_kwh,
            soh_loss=soh_loss,
            cycle_loss=soh_loss,  # unsplit; env never reads the split
            calendar_loss=0.0,
            soc=self.soc,
            soh=self.soh,
        )


def save_model(model: _BatterySurrogate, path: Path | str) -> None:
    torch.save(
        {
            "kind": type(model).__name__,
            "hidden": model.trunk.net[0].out_features,
            "config": model.cfg,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_model(path: Path | str) -> _BatterySurrogate:
    payload = torch.load(path, weights_only=False)
    cls = {"MLPBatteryModel": MLPBatteryModel, "PINNBatteryModel": PINNBatteryModel}[
        payload["kind"]
    ]
    model = cls(cfg=payload["config"], hidden=payload["hidden"])
    model.load_state_dict(payload["state_dict"])
    return model
