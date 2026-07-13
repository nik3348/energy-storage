"""Fuel and carbon prices as mean-reverting (Ornstein-Uhlenbeck) daily processes."""

import math

from dataclasses import dataclass

import numpy as np


@dataclass
class FuelPrices:
    gas_per_mwh_th: float  # $/MWh thermal
    coal_per_mwh_th: float
    carbon_per_t: float  # $/tCO2


@dataclass
class FuelMarketConfig:
    gas_mean: float = 25.0
    # Optional seasonal swing of the gas mean (real European gas is dearer in
    # winter — heating demand). Off by default: the market's winter price
    # premium is left to emerge from U-shaped electricity consumption against
    # a sloped merit order, keeping the two mechanisms separable.
    gas_seasonal_amp: float = 0.0
    gas_peak_doy: int = 15
    gas_reversion: float = 0.03
    gas_sigma: float = 0.9
    gas_floor: float = 5.0
    coal_mean: float = 11.0
    coal_reversion: float = 0.03
    coal_sigma: float = 0.25
    coal_floor: float = 3.0
    carbon_mean: float = 30.0
    carbon_reversion: float = 0.02
    carbon_sigma: float = 0.35
    carbon_floor: float = 0.0
    # Draw starting prices from the OU stationary distribution instead of
    # the mean, so short episodes see varied price levels.
    randomize_initial: bool = False


class _OUProcess:
    """One floored mean-reverting daily price process, optionally reverting
    to a seasonally varying mean (cosine peaking at peak_doy)."""

    def __init__(
        self,
        mean: float,
        reversion: float,
        sigma: float,
        floor: float,
        seasonal_amp: float = 0.0,
        peak_doy: int = 0,
    ):
        self.mean = mean
        self.reversion = reversion
        self.sigma = sigma
        self.floor = floor
        self.seasonal_amp = seasonal_amp
        self.peak_doy = peak_doy
        self.value = mean

    def mean_at(self, day_of_year: int) -> float:
        if self.seasonal_amp == 0.0:
            return self.mean
        phase = 2.0 * math.pi * (day_of_year - self.peak_doy) / 365.0
        return self.mean * (1.0 + self.seasonal_amp * math.cos(phase))

    def randomize(self, rng: np.random.Generator, day_of_year: int) -> None:
        # OU stationary std is sigma / sqrt(2 * reversion).
        stationary_std = self.sigma / np.sqrt(2 * self.reversion)
        self.value = max(self.floor, rng.normal(self.mean_at(day_of_year), stationary_std))

    def step(self, rng: np.random.Generator, day_of_year: int) -> float:
        drift = self.reversion * (self.mean_at(day_of_year) - self.value)
        self.value = max(self.floor, self.value + drift + rng.normal(0, self.sigma))
        return self.value


class FuelMarket:
    def __init__(self, config: FuelMarketConfig | None = None, rng: np.random.Generator | None = None):
        self.config = config or FuelMarketConfig()
        self.rng = rng if rng is not None else np.random.default_rng(0)
        cfg = self.config
        # Order matters for reproducibility: gas, coal, carbon draw from the
        # shared rng in that order on every call.
        self._processes = [
            _OUProcess(
                cfg.gas_mean, cfg.gas_reversion, cfg.gas_sigma, cfg.gas_floor,
                seasonal_amp=cfg.gas_seasonal_amp, peak_doy=cfg.gas_peak_doy,
            ),
            _OUProcess(cfg.coal_mean, cfg.coal_reversion, cfg.coal_sigma, cfg.coal_floor),
            _OUProcess(cfg.carbon_mean, cfg.carbon_reversion, cfg.carbon_sigma, cfg.carbon_floor),
        ]
        # Randomization needs the start day-of-year (the stationary draw must
        # center on the *seasonal* mean), which is only known at the first
        # step_day call.
        self._pending_randomize = cfg.randomize_initial

    def step_day(self, day_of_year: int) -> FuelPrices:
        if self._pending_randomize:
            for process in self._processes:
                process.randomize(self.rng, day_of_year)
            self._pending_randomize = False
        return FuelPrices(*(process.step(self.rng, day_of_year) for process in self._processes))
