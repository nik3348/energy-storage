"""Scenarios: scheduled shocks layered on top of the market's stochastic drivers.

A scenario modifies the simulated drivers (weather, fuel prices, unit
availability) inside its window without consuming any randomness, so a run
with scenarios stays comparable to a same-seed baseline run.
"""

from dataclasses import dataclass, replace

from energy_storage.market.fuels import FuelPrices
from energy_storage.market.generators import Generator
from energy_storage.market.weather import WeatherDay


class Scenario:
    name: str = "scenario"
    start_day: int = 0
    duration_days: int = 0

    def active(self, day: int) -> bool:
        return self.start_day <= day < self.start_day + self.duration_days

    def _envelope(self, day: int, ramp_days: float) -> float:
        """0..1 severity: ramps in at the start of the window and out at the end."""
        if not Scenario.active(self, day):
            return 0.0
        if ramp_days <= 0:
            return 1.0
        days_in = day - self.start_day + 1
        days_left = self.start_day + self.duration_days - day
        return min(1.0, days_in / ramp_days, days_left / ramp_days)

    def modify_weather(self, day: int, weather: WeatherDay) -> WeatherDay:
        return weather

    def modify_fuels(self, day: int, fuels: FuelPrices) -> FuelPrices:
        return fuels

    def modify_generators(self, day: int, generators: list[Generator]) -> None:
        pass


@dataclass
class ColdSnap(Scenario):
    """Extreme winter event: severe cold, calm winds, overcast skies."""

    start_day: int
    duration_days: int = 14
    temperature_drop_c: float = 10.0
    wind_factor: float = 0.5
    solar_factor: float = 0.7
    ramp_days: float = 2.0
    name: str = "cold-snap"

    def modify_weather(self, day, weather):
        s = self._envelope(day, self.ramp_days)
        if s == 0.0:
            return weather
        return replace(
            weather,
            temperature_c=weather.temperature_c - self.temperature_drop_c * s,
            wind_speed_ms=weather.wind_speed_ms * (1.0 - (1.0 - self.wind_factor) * s),
            solar_cf=weather.solar_cf * (1.0 - (1.0 - self.solar_factor) * s),
        )


@dataclass
class HeatWave(Scenario):
    """Sustained heat: high cooling demand, weak winds, strong sun."""

    start_day: int
    duration_days: int = 10
    temperature_rise_c: float = 8.0
    wind_factor: float = 0.7
    ramp_days: float = 2.0
    name: str = "heat-wave"

    def modify_weather(self, day, weather):
        s = self._envelope(day, self.ramp_days)
        if s == 0.0:
            return weather
        return replace(
            weather,
            temperature_c=weather.temperature_c + self.temperature_rise_c * s,
            wind_speed_ms=weather.wind_speed_ms * (1.0 - (1.0 - self.wind_factor) * s),
        )


@dataclass
class FuelShock(Scenario):
    """Fuel price spike: jumps to magnitude x for the window, then decays
    back exponentially with the given half-life."""

    start_day: int
    duration_days: int = 30
    fuel: str = "gas"  # "gas", "coal", or "carbon"
    magnitude: float = 2.5
    decay_half_life_days: float = 10.0
    name: str = "fuel-shock"

    def factor(self, day: int) -> float:
        if day < self.start_day:
            return 1.0
        end = self.start_day + self.duration_days
        if day < end:
            return self.magnitude
        return 1.0 + (self.magnitude - 1.0) * 0.5 ** ((day - end) / self.decay_half_life_days)

    def active(self, day: int) -> bool:
        # Active while more than 10% of the shock remains (covers the plateau
        # plus ~3.3 half-lives of the decay tail).
        return self.factor(day) > 1.0 + 0.1 * (self.magnitude - 1.0)

    def modify_fuels(self, day, fuels):
        f = self.factor(day)
        if f == 1.0:
            return fuels
        if self.fuel == "gas":
            return replace(fuels, gas_per_mwh_th=fuels.gas_per_mwh_th * f)
        if self.fuel == "coal":
            return replace(fuels, coal_per_mwh_th=fuels.coal_per_mwh_th * f)
        if self.fuel == "carbon":
            return replace(fuels, carbon_per_t=fuels.carbon_per_t * f)
        raise ValueError(f"unknown fuel {self.fuel!r}")


@dataclass
class Drought(Scenario):
    """Dry spell: hydro inflows collapse, pushing hydro up the merit order."""

    start_day: int
    duration_days: int = 90
    inflow_factor: float = 0.25
    name: str = "drought"

    def modify_weather(self, day, weather):
        if not self.active(day):
            return weather
        return replace(weather, hydro_inflow_mwh=weather.hydro_inflow_mwh * self.inflow_factor)


@dataclass
class PlantOutage(Scenario):
    """Forced outage of a named unit (e.g. "ccgt-1") for the window."""

    start_day: int
    generator_name: str = "ccgt-1"
    duration_days: int = 21
    name: str = "plant-outage"

    def modify_generators(self, day, generators):
        if not self.active(day):
            return
        for gen in generators:
            if gen.name == self.generator_name and hasattr(gen, "available"):
                gen.available = False
