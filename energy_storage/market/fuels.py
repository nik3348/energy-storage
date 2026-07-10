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


class FuelMarket:
    def __init__(self, config: FuelMarketConfig | None = None, rng: np.random.Generator | None = None):
        self.config = config or FuelMarketConfig()
        self.rng = rng if rng is not None else np.random.default_rng(0)
        cfg = self.config
        self.gas = cfg.gas_mean
        self.coal = cfg.coal_mean
        self.carbon = cfg.carbon_mean
        if cfg.randomize_initial:
            # OU stationary std is sigma / sqrt(2 * reversion).
            self.gas = max(
                cfg.gas_floor,
                self.rng.normal(cfg.gas_mean, cfg.gas_sigma / np.sqrt(2 * cfg.gas_reversion)),
            )
            self.coal = max(
                cfg.coal_floor,
                self.rng.normal(cfg.coal_mean, cfg.coal_sigma / np.sqrt(2 * cfg.coal_reversion)),
            )
            self.carbon = max(
                cfg.carbon_floor,
                self.rng.normal(
                    cfg.carbon_mean, cfg.carbon_sigma / np.sqrt(2 * cfg.carbon_reversion)
                ),
            )

    def step_day(self) -> FuelPrices:
        cfg = self.config
        self.gas = max(
            cfg.gas_floor,
            self.gas + cfg.gas_reversion * (cfg.gas_mean - self.gas) + self.rng.normal(0, cfg.gas_sigma),
        )
        self.coal = max(
            cfg.coal_floor,
            self.coal
            + cfg.coal_reversion * (cfg.coal_mean - self.coal)
            + self.rng.normal(0, cfg.coal_sigma),
        )
        self.carbon = max(
            cfg.carbon_floor,
            self.carbon
            + cfg.carbon_reversion * (cfg.carbon_mean - self.carbon)
            + self.rng.normal(0, cfg.carbon_sigma),
        )
        return FuelPrices(self.gas, self.coal, self.carbon)
