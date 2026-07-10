"""Day-ahead market: simulates one day of drivers and clears 24 hourly auctions."""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from energy_storage.market.clearing import clear_hour
from energy_storage.market.demand import Demand, DemandConfig
from energy_storage.market.fuels import FuelMarket, FuelMarketConfig, FuelPrices
from energy_storage.market.generators import Generator, default_fleet
from energy_storage.market.scenarios import Scenario
from energy_storage.market.weather import (
    Calendar,
    CalendarConfig,
    Weather,
    WeatherConfig,
    WeatherDay,
)


@dataclass
class MarketConfig:
    price_cap: float = 3000.0  # value of lost load: paid when supply falls short
    price_floor: float = -100.0
    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    fuels: FuelMarketConfig = field(default_factory=FuelMarketConfig)


@dataclass
class DayResult:
    day: int
    day_of_year: int
    is_weekend: bool
    is_holiday: bool
    prices: np.ndarray  # (24,) $/MWh
    demand_mw: np.ndarray  # (24,)
    generation_mw: dict[str, np.ndarray]  # tech -> (24,)
    unserved_mw: np.ndarray  # (24,) demand not met (scarcity)
    weather: WeatherDay
    fuel_prices: FuelPrices
    active_scenarios: list[str]


class Market:
    def __init__(
        self,
        config: MarketConfig | None = None,
        generators: list[Generator] | None = None,
        scenarios: list[Scenario] | None = None,
        seed: int = 0,
    ):
        self.config = config or MarketConfig()
        self.rng = np.random.default_rng(seed)
        self.calendar = Calendar(self.config.calendar)
        self.weather = Weather(self.config.weather, self.rng)
        self.demand = Demand(self.config.demand, self.rng)
        self.fuels = FuelMarket(self.config.fuels, self.rng)
        self.generators = generators if generators is not None else default_fleet()
        self.scenarios = list(scenarios) if scenarios is not None else []
        self.day = 0

    def add_scenario(self, scenario: Scenario) -> None:
        self.scenarios.append(scenario)

    def simulate_day(self) -> DayResult:
        day = self.day
        doy = self.calendar.day_of_year(day)
        is_weekend = self.calendar.is_weekend(day)
        is_holiday = self.calendar.is_holiday(day)

        weather = self.weather.simulate_day(doy)
        fuel_prices = self.fuels.step_day()
        # Scenario shocks apply before demand so e.g. a cold snap raises
        # heating load, and before new_day so hydro sees drought inflows.
        for scenario in self.scenarios:
            weather = scenario.modify_weather(day, weather)
            fuel_prices = scenario.modify_fuels(day, fuel_prices)
        demand = self.demand.simulate_day(weather.temperature_c, is_weekend, is_holiday)
        for gen in self.generators:
            gen.new_day(self.rng, weather)
        for scenario in self.scenarios:
            scenario.modify_generators(day, self.generators)

        prices = np.empty(24)
        unserved = np.zeros(24)
        generation: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(24))
        gens_by_name = {g.name: g for g in self.generators}

        for h in range(24):
            offers = [o for g in self.generators for o in g.offers(h, weather, fuel_prices)]
            price, dispatch = clear_hour(
                offers, demand[h], self.config.price_cap, self.config.price_floor
            )
            prices[h] = price
            dispatched_by_gen: dict[str, float] = defaultdict(float)
            for offer, mw in dispatch:
                generation[offer.tech][h] += mw
                dispatched_by_gen[offer.generator] += mw
            for name, mw in dispatched_by_gen.items():
                gens_by_name[name].settle(mw)
            unserved[h] = max(0.0, demand[h] - sum(mw for _, mw in dispatch))

        self.day += 1
        return DayResult(
            day=day,
            day_of_year=doy,
            is_weekend=is_weekend,
            is_holiday=is_holiday,
            prices=prices,
            demand_mw=demand,
            generation_mw=dict(generation),
            unserved_mw=unserved,
            weather=weather,
            fuel_prices=fuel_prices,
            active_scenarios=[s.name for s in self.scenarios if s.active(day)],
        )
