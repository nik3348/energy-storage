"""Stored market history and the price-only day representation.

The Dyna world model (docs/dyna-design.md) treats a market day as just its
24 day-ahead prices: the agent is a price-taker whose observation reads only
calendar flags and prices from a `DayResult`. This module holds everything
that needs no torch — the history container, the price normalization shared
with the env, the model dataset builder, and `ReplayedMarket` (experiment
arm B). The learned model itself lives in `market_model.py` (train extra).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from energy_storage.market.day_ahead import DayResult, Market, MarketConfig
from energy_storage.market.fuels import FuelPrices
from energy_storage.market.weather import WeatherDay

# Conditioning features for the day model:
# [sin_doy, cos_doy, is_weekend, is_holiday, prev-day price mean/min/max].
N_CONDITIONING = 7


def norm_price(price: np.ndarray | float, cap: float) -> np.ndarray:
    """Log-normalize prices; mirrors BatteryArbitrageEnv._norm_price (test-pinned)."""
    return np.sign(price) * np.log1p(np.abs(price)) / np.log1p(cap)


def denorm_price(z: np.ndarray | float, cap: float) -> np.ndarray:
    return np.sign(z) * np.expm1(np.abs(z) * np.log1p(cap))


def calendar_features(day_of_year: int, is_weekend: bool, is_holiday: bool) -> np.ndarray:
    angle = 2.0 * np.pi * day_of_year / 365.0
    return np.array([np.sin(angle), np.cos(angle), float(is_weekend), float(is_holiday)])


def prev_day_stats(prices: np.ndarray, cap: float) -> np.ndarray:
    """Compressed signature of yesterday's prices (log-normalized mean/min/max)."""
    z = norm_price(prices, cap)
    return np.array([z.mean(), z.min(), z.max()])


def price_only_day_result(
    day: int, day_of_year: int, is_weekend: bool, is_holiday: bool, prices: np.ndarray
) -> DayResult:
    """A DayResult carrying only what the env reads; the rest is dummies."""
    return DayResult(
        day=day,
        day_of_year=day_of_year,
        is_weekend=is_weekend,
        is_holiday=is_holiday,
        prices=prices,
        demand_mw=np.zeros(24),
        generation_mw={},
        unserved_mw=np.zeros(24),
        weather=WeatherDay(
            temperature_c=np.zeros(24),
            solar_cf=np.zeros(24),
            wind_speed_ms=np.zeros(24),
            hydro_inflow_mwh=0.0,
        ),
        fuel_prices=FuelPrices(0.0, 0.0, 0.0),
        active_scenarios=[],
    )


@dataclass
class MarketHistory:
    """D consecutive observed market days — the real-data budget."""

    prices: np.ndarray  # (D, 24)
    day_of_year: np.ndarray  # (D,) int
    is_weekend: np.ndarray  # (D,) bool
    is_holiday: np.ndarray  # (D,) bool

    def __len__(self) -> int:
        return len(self.prices)

    @classmethod
    def from_days(cls, days: list[DayResult]) -> "MarketHistory":
        return cls(
            prices=np.stack([d.prices for d in days]),
            day_of_year=np.array([d.day_of_year for d in days]),
            is_weekend=np.array([d.is_weekend for d in days]),
            is_holiday=np.array([d.is_holiday for d in days]),
        )

    @classmethod
    def collect(cls, config: MarketConfig | None, seed: int, days: int) -> "MarketHistory":
        market = Market(config=config, seed=seed)
        return cls.from_days([market.simulate_day() for _ in range(days)])

    def append_day(
        self, day_of_year: int, is_weekend: bool, is_holiday: bool, prices: np.ndarray
    ) -> None:
        """Grow the history by one observed day (the adaptation stream)."""
        self.prices = np.concatenate([self.prices, np.asarray(prices)[None, :]])
        self.day_of_year = np.append(self.day_of_year, day_of_year)
        self.is_weekend = np.append(self.is_weekend, is_weekend)
        self.is_holiday = np.append(self.is_holiday, is_holiday)

    def tail(self, days: int) -> "MarketHistory":
        """The most recent `days` days as an independent history."""
        return MarketHistory(
            prices=self.prices[-days:].copy(),
            day_of_year=self.day_of_year[-days:].copy(),
            is_weekend=self.is_weekend[-days:].copy(),
            is_holiday=self.is_holiday[-days:].copy(),
        )

    def save(self, path: Path | str) -> None:
        np.savez(
            path,
            prices=self.prices,
            day_of_year=self.day_of_year,
            is_weekend=self.is_weekend,
            is_holiday=self.is_holiday,
        )

    @classmethod
    def load(cls, path: Path | str) -> "MarketHistory":
        data = np.load(path)
        return cls(
            prices=data["prices"],
            day_of_year=data["day_of_year"],
            is_weekend=data["is_weekend"],
            is_holiday=data["is_holiday"],
        )


def build_dataset(history: MarketHistory, cap: float) -> tuple[np.ndarray, np.ndarray]:
    """(conditioning, target) pairs: day t is predicted from its calendar
    features plus day t-1's compressed price stats. Shapes (D-1, 7), (D-1, 24)."""
    rows_x, rows_y = [], []
    for t in range(1, len(history)):
        x = np.concatenate(
            [
                calendar_features(
                    int(history.day_of_year[t]),
                    bool(history.is_weekend[t]),
                    bool(history.is_holiday[t]),
                ),
                prev_day_stats(history.prices[t - 1], cap),
            ]
        )
        rows_x.append(x)
        rows_y.append(norm_price(history.prices[t], cap))
    return np.stack(rows_x).astype(np.float32), np.stack(rows_y).astype(np.float32)


class ReplayedMarket:
    """Experiment arm B: serves the stored real days verbatim, wrapping
    modulo the history length. Duck-types `Market` for the env
    (`simulate_day() -> DayResult`, settable integer `day`)."""

    def __init__(self, history: MarketHistory, config: MarketConfig | None = None):
        self.config = config or MarketConfig()
        self.history = history
        self.scenarios: list = []
        self.day = 0

    def simulate_day(self) -> DayResult:
        day = self.day
        i = day % len(self.history)
        result = price_only_day_result(
            day=day,
            day_of_year=int(self.history.day_of_year[i]),
            is_weekend=bool(self.history.is_weekend[i]),
            is_holiday=bool(self.history.is_holiday[i]),
            prices=self.history.prices[i].copy(),
        )
        self.day += 1
        return result
