"""Fuel and carbon prices as mean-reverting (Ornstein-Uhlenbeck) daily processes."""

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
    """One floored mean-reverting daily price process."""

    def __init__(self, mean: float, reversion: float, sigma: float, floor: float):
        self.mean = mean
        self.reversion = reversion
        self.sigma = sigma
        self.floor = floor
        self.value = mean

    def randomize(self, rng: np.random.Generator) -> None:
        # OU stationary std is sigma / sqrt(2 * reversion).
        stationary_std = self.sigma / np.sqrt(2 * self.reversion)
        self.value = max(self.floor, rng.normal(self.mean, stationary_std))

    def step(self, rng: np.random.Generator) -> float:
        drift = self.reversion * (self.mean - self.value)
        self.value = max(self.floor, self.value + drift + rng.normal(0, self.sigma))
        return self.value


class FuelMarket:
    def __init__(self, config: FuelMarketConfig | None = None, rng: np.random.Generator | None = None):
        self.config = config or FuelMarketConfig()
        self.rng = rng if rng is not None else np.random.default_rng(0)
        cfg = self.config
        # Order matters for reproducibility: gas, coal, carbon draw from the
        # shared rng in that order here and in step_day.
        self._processes = [
            _OUProcess(cfg.gas_mean, cfg.gas_reversion, cfg.gas_sigma, cfg.gas_floor),
            _OUProcess(cfg.coal_mean, cfg.coal_reversion, cfg.coal_sigma, cfg.coal_floor),
            _OUProcess(cfg.carbon_mean, cfg.carbon_reversion, cfg.carbon_sigma, cfg.carbon_floor),
        ]
        if cfg.randomize_initial:
            for process in self._processes:
                process.randomize(self.rng)

    def step_day(self) -> FuelPrices:
        return FuelPrices(*(process.step(self.rng) for process in self._processes))
