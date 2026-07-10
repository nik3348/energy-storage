"""Synthetic calendar and weather driving demand and renewable output."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CalendarConfig:
    days_per_year: int = 365
    # Synthetic holiday calendar (days of year).
    holidays_doy: frozenset[int] = frozenset({0, 88, 120, 185, 245, 331, 358, 359})


class Calendar:
    def __init__(self, config: CalendarConfig | None = None):
        self.config = config or CalendarConfig()

    def day_of_year(self, day: int) -> int:
        return day % self.config.days_per_year

    def day_of_week(self, day: int) -> int:
        return day % 7

    def is_weekend(self, day: int) -> bool:
        return self.day_of_week(day) >= 5

    def is_holiday(self, day: int) -> bool:
        return self.day_of_year(day) in self.config.holidays_doy


@dataclass
class WeatherConfig:
    # Temperature (northern-hemisphere midlatitude): warmest around day 200.
    temp_mean_c: float = 13.0
    temp_seasonal_amp_c: float = 10.0
    temp_diurnal_amp_c: float = 4.0
    temp_anomaly_persistence: float = 0.85  # daily AR(1) -> multi-day cold snaps / heat waves
    temp_anomaly_sigma_c: float = 1.8
    temp_hourly_noise_c: float = 0.5
    # Solar: longest day around day 172, cloudier in winter.
    solar_peak_doy: int = 172
    daylight_mean_h: float = 12.0
    daylight_amp_h: float = 3.0
    cloud_mean: float = 0.5
    cloud_seasonal_amp: float = 0.15
    cloud_sigma: float = 0.08
    cloud_attenuation: float = 0.8
    # Wind: windier in winter.
    wind_mean_ms: float = 8.5
    wind_seasonal_amp_ms: float = 2.0
    wind_persistence: float = 0.95  # hourly AR(1)
    wind_sigma_ms: float = 0.6
    # Hydro inflow: spring-melt bump around day 120.
    hydro_inflow_base_mwh: float = 1500.0
    hydro_melt_peak_doy: int = 120
    hydro_melt_amp: float = 2.2
    hydro_melt_width_days: float = 25.0
    hydro_noise_sigma: float = 0.25


@dataclass
class WeatherDay:
    temperature_c: np.ndarray  # (24,)
    solar_cf: np.ndarray  # (24,) irradiance as fraction of rated output
    wind_speed_ms: np.ndarray  # (24,)
    hydro_inflow_mwh: float


class Weather:
    def __init__(self, config: WeatherConfig | None = None, rng: np.random.Generator | None = None):
        self.config = config or WeatherConfig()
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._temp_anomaly = 0.0
        self._cloud = self.config.cloud_mean
        self._wind = self.config.wind_mean_ms

    def simulate_day(self, day_of_year: int) -> WeatherDay:
        cfg = self.config
        rng = self.rng
        hours = np.arange(24)
        phase = 2.0 * np.pi * (day_of_year - 200) / 365.0

        # Temperature: seasonal + diurnal (peak mid-afternoon) + persistent anomaly.
        self._temp_anomaly = (
            cfg.temp_anomaly_persistence * self._temp_anomaly
            + rng.normal(0.0, cfg.temp_anomaly_sigma_c)
        )
        temperature = (
            cfg.temp_mean_c
            + cfg.temp_seasonal_amp_c * np.cos(phase)
            + cfg.temp_diurnal_amp_c * np.cos(2.0 * np.pi * (hours - 15) / 24.0)
            + self._temp_anomaly
            + rng.normal(0.0, cfg.temp_hourly_noise_c, 24)
        )

        # Solar: half-sine over daylight hours, scaled by seasonal sun height
        # and attenuated by slowly-varying cloud cover.
        solar_phase = 2.0 * np.pi * (day_of_year - cfg.solar_peak_doy) / 365.0
        daylight = cfg.daylight_mean_h + cfg.daylight_amp_h * np.cos(solar_phase)
        sunrise = 12.0 - daylight / 2.0
        sun_height = 0.65 + 0.35 * np.cos(solar_phase)
        clear_sky = np.where(
            (hours + 0.5 > sunrise) & (hours + 0.5 < sunrise + daylight),
            sun_height * np.sin(np.pi * (hours + 0.5 - sunrise) / daylight),
            0.0,
        )
        cloud_target = cfg.cloud_mean + cfg.cloud_seasonal_amp * np.cos(
            2.0 * np.pi * (day_of_year - 355) / 365.0
        )
        cloud = np.empty(24)
        for h in range(24):
            self._cloud += 0.1 * (cloud_target - self._cloud) + rng.normal(0.0, cfg.cloud_sigma)
            self._cloud = float(np.clip(self._cloud, 0.0, 1.0))
            cloud[h] = self._cloud
        solar_cf = clear_sky * (1.0 - cfg.cloud_attenuation * cloud)

        # Wind: hourly AR(1) around a seasonal mean.
        wind_target = cfg.wind_mean_ms + cfg.wind_seasonal_amp_ms * np.cos(
            2.0 * np.pi * (day_of_year - 15) / 365.0
        )
        wind = np.empty(24)
        for h in range(24):
            self._wind += (1.0 - cfg.wind_persistence) * (wind_target - self._wind) + rng.normal(
                0.0, cfg.wind_sigma_ms
            )
            self._wind = max(0.0, self._wind)
            wind[h] = self._wind

        # Hydro inflow: spring-melt bump with lognormal noise.
        melt = cfg.hydro_melt_amp * np.exp(
            -0.5 * ((day_of_year - cfg.hydro_melt_peak_doy) / cfg.hydro_melt_width_days) ** 2
        )
        inflow = cfg.hydro_inflow_base_mwh * (1.0 + melt) * rng.lognormal(0.0, cfg.hydro_noise_sigma)

        return WeatherDay(
            temperature_c=temperature,
            solar_cf=solar_cf,
            wind_speed_ms=wind,
            hydro_inflow_mwh=float(inflow),
        )
