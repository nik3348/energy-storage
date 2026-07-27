"""Gymnasium environment: battery arbitrage against the day-ahead market.

The agent is a price-taker operating one battery. Each step is one hour.
Action: Box(-1, 1) — fraction of max power, >0 charges, <0 discharges.
Reward: profit minus monetized SoH degradation, in scaled dollars.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from energy_storage.battery import Battery, BatteryConfig
from energy_storage.market import DayResult, FuelMarketConfig, Market, MarketConfig, Scenario

# Observation layout: N_SCALAR_FEATURES scalars, then the next PRICE_WINDOW
# hourly day-ahead prices (log-normalized).
# [soc, soh, sin_hour, cos_hour, sin_doy, cos_doy, is_weekend, is_holiday,
#  prices...]
# Weather is deliberately not observed: it affects reward only through
# prices, and the next 24 prices are known exactly (day-ahead market) —
# the rolling-horizon oracle hits 99.9% of hindsight from prices alone.
N_SCALAR_FEATURES = 8
PRICE_WINDOW = 24


def default_market_config() -> MarketConfig:
    return MarketConfig(fuels=FuelMarketConfig(randomize_initial=True))


@dataclass
class EnvConfig:
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    market: MarketConfig = field(default_factory=default_market_config)
    episode_days: int = 14
    # Degradation is monetized: each point of SoH lost costs a share of the
    # battery's replacement value.
    replacement_cost_per_kwh: float = 250.0
    reward_scale: float = 0.1
    initial_soc_range: tuple[float, float] = (0.2, 0.8)
    initial_soh_range: tuple[float, float] = (0.85, 1.0)
    # Days simulated before the episode starts, letting weather/fuel AR
    # states wander away from their initial values.
    burn_in_days: int = 3
    # Optional domain randomization: called at reset with (rng, start_day),
    # returns scenarios for the episode.
    scenario_sampler: Callable[[np.random.Generator, int], list[Scenario]] | None = None
    # Optional world-model hook (Dyna arms): called at reset with
    # (market config, seed) instead of constructing the real Market. The
    # return value must duck-type Market: `simulate_day() -> DayResult` and
    # a settable integer `day`.
    market_factory: Callable[[MarketConfig, int], Market] | None = None
    # Optional learned-battery hook (PINN experiment, docs/pinn-battery-design.md):
    # called at construction with the BatteryConfig instead of constructing the
    # real Battery. The return value must duck-type Battery: `reset(soc, soh)`,
    # mutable `soc`/`soh`, and `step(power_kw, dt_h) -> BatteryStepResult`.
    battery_factory: Callable[[BatteryConfig], object] | None = None


class BatteryArbitrageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.battery = (
            self.config.battery_factory(self.config.battery)
            if self.config.battery_factory is not None
            else Battery(self.config.battery)
        )
        self.market: Market | None = None
        self._today: DayResult | None = None
        self._tomorrow: DayResult | None = None
        self._hour = 0
        self._steps = 0

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_dim = N_SCALAR_FEATURES + PRICE_WINDOW
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    @property
    def episode_steps(self) -> int:
        return self.config.episode_days * 24

    @property
    def hour(self) -> int:
        """Hour of day (0-23) the next step will settle."""
        return self._hour

    @property
    def today(self) -> DayResult:
        """The market day the next step settles against."""
        if self._today is None:
            raise RuntimeError("call reset() before accessing today")
        return self._today

    def _norm_price(self, price: float | np.ndarray) -> np.ndarray:
        cap = self.config.market.price_cap
        return np.sign(price) * np.log1p(np.abs(price)) / np.log1p(cap)

    def price_window(self) -> np.ndarray:
        """The next PRICE_WINDOW hourly day-ahead prices ($/MWh), starting with
        the hour the next step settles. This is the agent's information set."""
        return np.concatenate(
            [self._today.prices[self._hour :], self._tomorrow.prices[: self._hour]]
        )

    def _obs(self) -> np.ndarray:
        day = self._today
        h = self._hour
        doy_angle = 2.0 * np.pi * day.day_of_year / 365.0
        hour_angle = 2.0 * np.pi * h / 24.0
        features = [
            self.battery.soc,
            self.battery.soh,
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(doy_angle),
            np.cos(doy_angle),
            float(day.is_weekend),
            float(day.is_holiday),
        ]
        assert len(features) == N_SCALAR_FEATURES
        parts = [features, self._norm_price(self.price_window())]
        return np.concatenate(parts).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg = self.config

        market_seed = int(self.np_random.integers(2**31))
        start_day = int(self.np_random.integers(365))
        if cfg.market_factory is not None:
            self.market = cfg.market_factory(cfg.market, market_seed)
        else:
            self.market = Market(config=cfg.market, seed=market_seed)
        if cfg.scenario_sampler is not None:
            self.market.scenarios = list(cfg.scenario_sampler(self.np_random, start_day))
        self.market.day = start_day - cfg.burn_in_days
        for _ in range(cfg.burn_in_days):
            self.market.simulate_day()
        self._today = self.market.simulate_day()
        self._tomorrow = self.market.simulate_day()
        self._hour = 0
        self._steps = 0

        self.battery.reset(
            soc=float(self.np_random.uniform(*cfg.initial_soc_range)),
            soh=float(self.np_random.uniform(*cfg.initial_soh_range)),
        )
        return self._obs(), {}

    def step(self, action):
        if self.market is None:
            raise RuntimeError("call reset() before step()")
        cfg = self.config
        bat_cfg = cfg.battery
        fraction = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        power_kw = fraction * (bat_cfg.max_charge_kw if fraction >= 0 else bat_cfg.max_discharge_kw)

        price = float(self._today.prices[self._hour])
        result = self.battery.step(power_kw, dt_h=1.0)
        profit = -price * result.grid_energy_kwh / 1000.0
        degradation_cost = (
            result.soh_loss * cfg.replacement_cost_per_kwh * bat_cfg.capacity_kwh
        )
        reward = (profit - degradation_cost) * cfg.reward_scale

        info = {
            "profit": profit,
            "degradation_cost": degradation_cost,
            "price": price,
            "grid_energy_kwh": result.grid_energy_kwh,
            "soc": self.battery.soc,
            "soh": self.battery.soh,
            "cycle_loss": result.cycle_loss,
            "calendar_loss": result.calendar_loss,
            "constraint_clipped": result.constraint_clipped,
            "day": self._today.day,
            "hour": self._hour,
            "active_scenarios": self._today.active_scenarios,
        }

        self._steps += 1
        self._hour += 1
        if self._hour == 24:
            self._hour = 0
            self._today = self._tomorrow
            self._tomorrow = self.market.simulate_day()

        terminated = self.battery.soh <= bat_cfg.eol_soh + 1e-12
        truncated = self._steps >= self.episode_steps
        return self._obs(), float(reward), bool(terminated), bool(truncated), info
