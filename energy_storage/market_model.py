"""Learned generative model of market days — the Dyna world model.

One sample = one day of 24 log-normalized prices, conditioned on calendar
features and yesterday's compressed price stats (see docs/dyna-design.md).
The day is generated hour-autoregressively: a GRU rolls over the 24 hours,
each hour a Gaussian conditioned on the sampled previous hours. Flat
diagonal (and low-rank) Gaussian heads were tried first and failed the
validation gate: with no within-day conditioning they must explain scarcity
blocks as fat per-hour tails, which over-disperses every season.

An ensemble of these models is fit on a budget of observed days;
`LearnedMarket` wraps a fitted ensemble as a `Market` stand-in for the env.

Imports torch: only usable under the 'train' extra (same rule as viz.py) —
the core package must not import this module.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from energy_storage.market.day_ahead import DayResult, MarketConfig
from energy_storage.market.weather import Calendar
from energy_storage.market_history import (
    N_CONDITIONING,
    calendar_features,
    denorm_price,
    prev_day_stats,
    price_only_day_result,
)

N_HOURS = 24
LOG_STD_MIN, LOG_STD_MAX = -6.0, 2.0


class MarketDayModel(nn.Module):
    """Hour-autoregressive day generator: the conditioning features seed the
    GRU state, then each hour's price is a Gaussian given the hours before it."""

    def __init__(self, hidden: int = 128, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.cond = nn.Sequential(nn.Linear(N_CONDITIONING, hidden), nn.Tanh())
        self.cell = nn.GRUCell(1, hidden)
        self.head = nn.Linear(hidden, 2)
        # Start near-deterministic so the mean path is learned first;
        # variance grows only where the NLL demands it.
        nn.init.normal_(self.head.weight, std=1e-3)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([0.0, -2.0]))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced pass. x: (B, N_CONDITIONING), y: (B, 24) targets.
        Returns per-hour (mu, log_std), each (B, 24)."""
        h = self.cond(x)
        prev = torch.zeros(len(x), 1, dtype=x.dtype, device=x.device)
        mus, log_stds = [], []
        for t in range(N_HOURS):
            h = self.cell(prev, h)
            mu, log_std = self.head(h).unbind(dim=-1)
            mus.append(mu)
            log_stds.append(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX))
            prev = y[:, t : t + 1]
        return torch.stack(mus, dim=1), torch.stack(log_stds, dim=1)

    @torch.no_grad()
    def rollout(self, x: np.ndarray, rng: np.random.Generator | None) -> np.ndarray:
        """Free-running generation of one day (24,) in log-normalized space.
        rng=None rolls out the mean path (used for ensemble disagreement)."""
        h = self.cond(torch.from_numpy(x.astype(np.float32)).unsqueeze(0))
        prev = torch.zeros(1, 1)
        out = np.empty(N_HOURS)
        for t in range(N_HOURS):
            h = self.cell(prev, h)
            mu, log_std = self.head(h).unbind(dim=-1)
            z = float(mu)
            if rng is not None:
                std = float(torch.exp(torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)))
                z += std * rng.standard_normal()
            out[t] = z
            prev = torch.full((1, 1), z, dtype=torch.float32)
        return out


def _fit_one(
    model: MarketDayModel,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    rng: np.random.Generator,
) -> float:
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        order = torch.from_numpy(rng.permutation(len(x_t)))
        for start in range(0, len(x_t), batch_size):
            batch = order[start : start + batch_size]
            mu, log_std = model(x_t[batch], y_t[batch])
            residual = (y_t[batch] - mu) * torch.exp(-log_std)
            loss = (0.5 * residual**2 + log_std).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return float(loss.detach())


class MarketModelEnsemble:
    """K bootstrap-resampled MarketDayModels. Each imagined episode draws one
    member; disagreement between members is an OOD signal."""

    def __init__(self, k: int = 5, hidden: int = 128, seed: int = 0):
        self.members = [MarketDayModel(hidden=hidden, seed=seed + i) for i in range(k)]
        self._hidden = hidden
        # Cold-start conditioning for LearnedMarket before its first
        # generated day; set to the training-set mean by fit().
        self.prev_stats_mean = np.zeros(3)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 2000,
        batch_size: int = 64,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> list[float]:
        """Fit every member on its own bootstrap resample; returns final losses."""
        rng = np.random.default_rng(seed)
        self.prev_stats_mean = x[:, 4:7].mean(axis=0).astype(np.float64)
        losses = []
        for model in self.members:
            idx = rng.integers(0, len(x), size=len(x))
            losses.append(_fit_one(model, x[idx], y[idx], epochs, batch_size, lr, rng))
        return losses

    def fine_tune(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 25,
        batch_size: int = 64,
        lr: float = 3e-4,
        seed: int = 0,
    ) -> list[float]:
        """Continue training every member from its current weights (fresh
        bootstrap per member, so ensemble diversity survives). The adaptation
        loop's nightly model update — deliberately not a refit from scratch:
        pre-shift structure (diurnal shape, calendar response) must persist."""
        rng = np.random.default_rng(seed)
        self.prev_stats_mean = x[:, 4:7].mean(axis=0).astype(np.float64)
        losses = []
        for model in self.members:
            idx = rng.integers(0, len(x), size=len(x))
            losses.append(_fit_one(model, x[idx], y[idx], epochs, batch_size, lr, rng))
        return losses

    def sample_day(self, x: np.ndarray, rng: np.random.Generator, member: int) -> np.ndarray:
        """One day of prices in the log-normalized space, shape (24,)."""
        return self.members[member].rollout(x, rng)

    @torch.no_grad()
    def day_nll(self, x: np.ndarray, y: np.ndarray) -> float:
        """Exact NLL of one observed day under the ensemble mixture: chain
        rule over hours (teacher-forced per-hour Gaussians), logsumexp over
        members. The adaptation loop's detection signal — a regime shift
        surprises every member at once, so this spikes before disagreement
        (which only measures where members differ) moves."""
        x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)
        log_probs = []
        for model in self.members:
            mu, log_std = model(x_t, y_t)
            residual = (y_t - mu) * torch.exp(-log_std)
            log_probs.append(
                (-0.5 * residual**2 - log_std - 0.5 * np.log(2.0 * np.pi)).sum()
            )
        mix = torch.logsumexp(torch.stack(log_probs), dim=0) - np.log(len(self.members))
        return float(-mix)

    def disagreement(self, x: np.ndarray) -> float:
        """Std of member mean rollouts, averaged over hours — an OOD signal."""
        means = np.stack([m.rollout(x, rng=None) for m in self.members])
        return float(means.std(axis=0).mean())

    def save(self, path: Path | str) -> None:
        torch.save(
            {
                "hidden": self._hidden,
                "state_dicts": [m.state_dict() for m in self.members],
                "prev_stats_mean": self.prev_stats_mean,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "MarketModelEnsemble":
        payload = torch.load(path, weights_only=False)
        ensemble = cls(k=len(payload["state_dicts"]), hidden=payload["hidden"])
        for model, state in zip(ensemble.members, payload["state_dicts"]):
            model.load_state_dict(state)
        ensemble.prev_stats_mean = payload["prev_stats_mean"]
        return ensemble


class LearnedMarket:
    """Experiment arm C: a fitted ensemble wrapped as a `Market` stand-in
    (`simulate_day() -> DayResult`, settable integer `day`). Calendar flags
    are real (they're known); prices are synthetic; everything else in the
    DayResult is dummies the env never reads. One ensemble member is drawn
    per instance, i.e. per episode."""

    def __init__(
        self,
        ensemble: MarketModelEnsemble,
        config: MarketConfig | None = None,
        seed: int = 0,
        member: int | None = None,
    ):
        self.config = config or MarketConfig()
        self.ensemble = ensemble
        self.rng = np.random.default_rng(seed)
        self.calendar = Calendar(self.config.calendar)
        self.scenarios: list = []
        self.day = 0
        self.member = (
            int(self.rng.integers(len(ensemble.members))) if member is None else member
        )
        self._prev_stats = np.asarray(ensemble.prev_stats_mean, dtype=np.float64).copy()

    def simulate_day(self) -> DayResult:
        day = self.day
        doy = self.calendar.day_of_year(day)
        is_weekend = self.calendar.is_weekend(day)
        is_holiday = self.calendar.is_holiday(day)
        cap, floor = self.config.price_cap, self.config.price_floor

        x = np.concatenate(
            [calendar_features(doy, is_weekend, is_holiday), self._prev_stats]
        )
        z = self.ensemble.sample_day(x, self.rng, member=self.member)
        prices = np.clip(denorm_price(z, cap), floor, cap)
        self._prev_stats = prev_day_stats(prices, cap)

        self.day += 1
        return price_only_day_result(day, doy, is_weekend, is_holiday, prices)
