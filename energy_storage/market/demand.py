"""Electricity demand model: diurnal shape, temperature response, holidays."""

from dataclasses import dataclass, field

import numpy as np

# MW by hour of day. Morning ramp, evening peak around 19:00.
WEEKDAY_PROFILE_MW = (
    530, 510, 500, 495, 500, 520, 580, 660, 720, 750, 765, 770,
    765, 755, 750, 750, 765, 810, 860, 880, 845, 780, 690, 600,
)
# Weekends: later morning rise, flatter and lower overall.
WEEKEND_PROFILE_MW = (
    540, 520, 505, 500, 500, 510, 530, 570, 620, 670, 700, 715,
    715, 710, 700, 700, 715, 750, 790, 805, 780, 730, 660, 590,
)


@dataclass
class DemandConfig:
    weekday_profile_mw: tuple = WEEKDAY_PROFILE_MW
    weekend_profile_mw: tuple = WEEKEND_PROFILE_MW
    heating_ref_c: float = 15.0
    cooling_ref_c: float = 21.0
    heating_mw_per_c: float = 10.0
    cooling_mw_per_c: float = 14.0
    holiday_factor: float = 0.85
    noise_sigma: float = 0.02  # multiplicative


class Demand:
    def __init__(self, config: DemandConfig | None = None, rng: np.random.Generator | None = None):
        self.config = config or DemandConfig()
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def simulate_day(
        self, temperature_c: np.ndarray, is_weekend: bool, is_holiday: bool
    ) -> np.ndarray:
        cfg = self.config
        profile = np.array(cfg.weekend_profile_mw if is_weekend else cfg.weekday_profile_mw, float)
        if is_holiday:
            profile = profile * cfg.holiday_factor
        heating = cfg.heating_mw_per_c * np.clip(cfg.heating_ref_c - temperature_c, 0.0, None)
        cooling = cfg.cooling_mw_per_c * np.clip(temperature_c - cfg.cooling_ref_c, 0.0, None)
        noise = self.rng.normal(1.0, cfg.noise_sigma, 24)
        return (profile + heating + cooling) * noise
