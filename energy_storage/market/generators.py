"""Generators submitting (price, quantity) offers to the day-ahead market."""

from dataclasses import dataclass

import numpy as np

from energy_storage.market.fuels import FuelPrices
from energy_storage.market.weather import WeatherDay


@dataclass
class Offer:
    generator: str
    tech: str
    price: float  # $/MWh
    quantity_mw: float


class Generator:
    name: str
    tech: str

    def new_day(self, rng: np.random.Generator, weather: WeatherDay) -> None:
        pass

    def offers(self, hour: int, weather: WeatherDay, fuels: FuelPrices) -> list[Offer]:
        raise NotImplementedError

    def settle(self, dispatched_mw: float) -> None:
        pass

    def force_outage(self) -> None:
        """Take the unit offline (scenario hook). No-op for units that
        cannot be forced out."""


class SolarFarm(Generator):
    """Bids negative: subsidies pay per MWh generated, so running at slightly
    negative prices still earns money."""

    tech = "solar"

    def __init__(self, name: str = "solar", capacity_mw: float = 500.0, bid: float = -25.0):
        self.name = name
        self.capacity_mw = capacity_mw
        self.bid = bid

    def offers(self, hour, weather, fuels):
        qty = self.capacity_mw * weather.solar_cf[hour]
        return [Offer(self.name, self.tech, self.bid, qty)] if qty > 0 else []


class WindFarm(Generator):
    tech = "wind"

    def __init__(
        self,
        name: str = "wind",
        capacity_mw: float = 350.0,
        bid: float = -27.0,
        cut_in_ms: float = 3.0,
        rated_ms: float = 11.0,
        cut_out_ms: float = 25.0,
    ):
        self.name = name
        self.capacity_mw = capacity_mw
        self.bid = bid
        self.cut_in_ms = cut_in_ms
        self.rated_ms = rated_ms
        self.cut_out_ms = cut_out_ms

    def power_curve(self, wind_ms: float) -> float:
        if wind_ms < self.cut_in_ms or wind_ms > self.cut_out_ms:
            return 0.0
        if wind_ms >= self.rated_ms:
            return 1.0
        return ((wind_ms - self.cut_in_ms) / (self.rated_ms - self.cut_in_ms)) ** 3

    def offers(self, hour, weather, fuels):
        qty = self.capacity_mw * self.power_curve(weather.wind_speed_ms[hour])
        return [Offer(self.name, self.tech, self.bid, qty)] if qty > 0 else []


class GeothermalPlant(Generator):
    """Must-run baseload: cycling off is not practical, so it bids negative
    to stay dispatched through low-price hours."""

    tech = "geothermal"

    def __init__(self, name: str = "geothermal", capacity_mw: float = 50.0, bid: float = -5.0):
        self.name = name
        self.capacity_mw = capacity_mw
        self.bid = bid

    def offers(self, hour, weather, fuels):
        return [Offer(self.name, self.tech, self.bid, self.capacity_mw * 0.95)]


class HydroPlant(Generator):
    """Reservoir hydro. Water itself is free, so it bids its opportunity cost:
    cheap when the reservoir is full, expensive when it is scarce."""

    tech = "hydro"

    def __init__(
        self,
        name: str = "hydro",
        capacity_mw: float = 150.0,
        reservoir_max_mwh: float = 45000.0,
        initial_fill: float = 0.55,
        reference_fill: float = 0.55,
        reference_bid: float = 12.0,
        min_bid: float = 3.0,
        max_bid: float = 250.0,
    ):
        self.name = name
        self.capacity_mw = capacity_mw
        self.reservoir_max_mwh = reservoir_max_mwh
        self.reservoir_mwh = initial_fill * reservoir_max_mwh
        self.reference_fill = reference_fill
        self.reference_bid = reference_bid
        self.min_bid = min_bid
        self.max_bid = max_bid

    def new_day(self, rng, weather):
        self.reservoir_mwh = min(
            self.reservoir_max_mwh, self.reservoir_mwh + weather.hydro_inflow_mwh
        )

    def offers(self, hour, weather, fuels):
        if self.reservoir_mwh <= 0:
            return []
        fill = max(self.reservoir_mwh / self.reservoir_max_mwh, 0.02)
        bid = float(np.clip(self.reference_bid * (self.reference_fill / fill) ** 2, self.min_bid, self.max_bid))
        return [Offer(self.name, self.tech, bid, min(self.capacity_mw, self.reservoir_mwh))]

    def settle(self, dispatched_mw):
        self.reservoir_mwh = max(0.0, self.reservoir_mwh - dispatched_mw)


class ThermalUnit(Generator):
    """Coal or gas unit. Marginal cost = fuel/efficiency + carbon + variable O&M.

    Units with min_output_fraction > 0 (coal) offer that block at a negative
    price: shutting down and restarting costs more than paying to keep the
    boiler warm for a few hours. This is what lets clearing prices go negative
    when renewables flood the market.
    """

    def __init__(
        self,
        name: str,
        fuel: str,  # "gas" or "coal"
        capacity_mw: float,
        efficiency: float,
        emissions_t_per_mwh: float,
        vom_per_mwh: float = 2.0,
        min_output_fraction: float = 0.0,
        min_run_bid: float = -15.0,
        outage_prob_per_day: float = 0.01,
        repair_prob_per_day: float = 0.25,
    ):
        self.name = name
        self.tech = fuel
        self.fuel = fuel
        self.capacity_mw = capacity_mw
        self.efficiency = efficiency
        self.emissions_t_per_mwh = emissions_t_per_mwh
        self.vom_per_mwh = vom_per_mwh
        self.min_output_fraction = min_output_fraction
        self.min_run_bid = min_run_bid
        self.outage_prob_per_day = outage_prob_per_day
        self.repair_prob_per_day = repair_prob_per_day
        self.available = True

    def new_day(self, rng, weather):
        if self.available:
            self.available = rng.random() >= self.outage_prob_per_day
        else:
            self.available = rng.random() < self.repair_prob_per_day

    def force_outage(self):
        self.available = False

    def marginal_cost(self, fuels: FuelPrices) -> float:
        fuel_price = fuels.gas_per_mwh_th if self.fuel == "gas" else fuels.coal_per_mwh_th
        return (
            fuel_price / self.efficiency
            + fuels.carbon_per_t * self.emissions_t_per_mwh
            + self.vom_per_mwh
        )

    def offers(self, hour, weather, fuels):
        if not self.available:
            return []
        mc = self.marginal_cost(fuels)
        min_mw = self.capacity_mw * self.min_output_fraction
        result = []
        if min_mw > 0:
            result.append(Offer(self.name, self.tech, self.min_run_bid, min_mw))
        result.append(Offer(self.name, self.tech, mc, self.capacity_mw - min_mw))
        return result


def default_fleet() -> list[Generator]:
    """A deliberately granular thermal fleet: units of varying vintage and
    efficiency make the merit order slope, so seasonal demand swings move the
    marginal cost — that slope is what turns U-shaped consumption into a
    winter price premium. Emissions scale inversely with efficiency."""
    return [
        SolarFarm(),
        WindFarm(),
        GeothermalPlant(),
        HydroPlant(),
        ThermalUnit("coal-1", "coal", 100.0, 0.42, 0.81, min_output_fraction=0.3),
        ThermalUnit("coal-2", "coal", 80.0, 0.36, 0.94, min_output_fraction=0.3),
        ThermalUnit("ccgt-1", "gas", 100.0, 0.60, 0.30),
        ThermalUnit("ccgt-2", "gas", 100.0, 0.54, 0.34),
        ThermalUnit("ccgt-3", "gas", 100.0, 0.48, 0.38),
        ThermalUnit("ccgt-4", "gas", 100.0, 0.43, 0.42),
        ThermalUnit("gas-steam-1", "gas", 60.0, 0.38, 0.48, vom_per_mwh=4.0),
        ThermalUnit("gas-steam-2", "gas", 60.0, 0.34, 0.53, vom_per_mwh=4.0),
        ThermalUnit("ocgt-1", "gas", 40.0, 0.32, 0.57, vom_per_mwh=8.0),
        ThermalUnit("ocgt-2", "gas", 40.0, 0.29, 0.62, vom_per_mwh=8.0),
        ThermalUnit("ocgt-3", "gas", 40.0, 0.26, 0.70, vom_per_mwh=8.0),
    ]
