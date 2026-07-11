"""Battery model with state of charge (SoC) and state of health (SoH)."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryStepResult:
    grid_energy_kwh: float  # >0 drawn from the grid, <0 delivered to it
    throughput_kwh: float  # energy through the cells, always >= 0
    soh_loss: float
    cycle_loss: float
    calendar_loss: float
    soc: float
    soh: float


@dataclass
class BatteryConfig:
    capacity_kwh: float = 100.0
    max_charge_kw: float = 50.0
    max_discharge_kw: float = 50.0
    round_trip_efficiency: float = 0.90
    soc_min: float = 0.05
    soc_max: float = 0.95
    initial_soc: float = 0.5
    # Cycle aging: base rate is linear in energy throughput, calibrated so
    # cycle_life full equivalent cycles at low power takes SoH 1.0 -> eol_soh.
    # cycle_stress makes high-power (dis)charging cost more per kWh: the
    # per-kWh loss is scaled by (1 + cycle_stress * |power| / max_power).
    cycle_life: float = 5000.0
    cycle_stress: float = 1.0
    # Calendar aging: sitting idle at SoC 0.5 takes SoH 1.0 -> eol_soh in
    # calendar_life_years. Aging is faster at high SoC (see step()).
    calendar_life_years: float = 15.0
    eol_soh: float = 0.8


class Battery:
    """Simple battery with SoC tracking and throughput-based SoH degradation.

    Conventions:
    - power > 0 charges the battery, power < 0 discharges it.
    - SoC is the fraction of the *current* (degraded) usable capacity.
    - SoH scales the nominal capacity: effective capacity = capacity_kwh * soh.
    """

    def __init__(self, config: BatteryConfig | None = None):
        self.config = config or BatteryConfig()
        # Split round-trip efficiency evenly between charge and discharge.
        self._one_way_eff = math.sqrt(self.config.round_trip_efficiency)
        # One full equivalent cycle = charge + discharge = 2x capacity of throughput.
        self._soh_loss_per_kwh = (1.0 - self.config.eol_soh) / (
            2.0 * self.config.cycle_life * self.config.capacity_kwh
        )
        self._calendar_loss_per_hour = (1.0 - self.config.eol_soh) / (
            self.config.calendar_life_years * 8760.0
        )
        self.reset()

    def reset(self, soc: float | None = None, soh: float = 1.0) -> None:
        self.soc = self.config.initial_soc if soc is None else soc
        self.soh = soh

    @property
    def effective_capacity_kwh(self) -> float:
        return self.config.capacity_kwh * self.soh

    @property
    def energy_stored_kwh(self) -> float:
        return self.soc * self.effective_capacity_kwh

    def step(self, power_kw: float, dt_h: float = 1.0) -> BatteryStepResult:
        """Charge (power_kw > 0) or discharge (power_kw < 0) for dt_h hours.

        Requested power is clipped to power limits and to what the SoC
        bounds allow. Returns the grid-side energy actually exchanged and
        the SoH lost this step.
        """
        cfg = self.config
        capacity = self.effective_capacity_kwh
        power_kw = max(-cfg.max_discharge_kw, min(cfg.max_charge_kw, power_kw))
        soc_before = self.soc

        if power_kw >= 0.0:
            # Charging: grid energy in, losses applied before storage.
            headroom_kwh = (cfg.soc_max - self.soc) * capacity
            stored_kwh = min(power_kw * dt_h * self._one_way_eff, headroom_kwh)
            grid_energy_kwh = stored_kwh / self._one_way_eff
            self.soc += stored_kwh / capacity
            throughput_kwh = stored_kwh
        else:
            # Discharging: losses applied on the way out.
            available_kwh = (self.soc - cfg.soc_min) * capacity
            drawn_kwh = min(-power_kw * dt_h / self._one_way_eff, available_kwh)
            grid_energy_kwh = -drawn_kwh * self._one_way_eff
            self.soc -= drawn_kwh / capacity
            throughput_kwh = drawn_kwh

        # Cycle aging: per-kWh loss grows with how hard the battery is pushed.
        max_power = cfg.max_charge_kw if power_kw >= 0.0 else cfg.max_discharge_kw
        power_fraction = abs(power_kw) / max_power if max_power > 0.0 else 0.0
        cycle_loss = (
            throughput_kwh
            * self._soh_loss_per_kwh
            * (1.0 + cfg.cycle_stress * power_fraction)
        )

        # Calendar aging: always ticks, faster at high SoC. The (0.5 + soc)
        # factor makes a full battery age twice as fast as an empty one and
        # calibrates to calendar_life_years at SoC 0.5.
        soc_avg = 0.5 * (soc_before + self.soc)
        calendar_loss = self._calendar_loss_per_hour * (0.5 + soc_avg) * dt_h

        soh_loss = cycle_loss + calendar_loss
        old_capacity = capacity
        self.soh = max(cfg.eol_soh, self.soh - soh_loss)

        # Stored energy is conserved through degradation; re-express SoC
        # against the new, smaller capacity.
        if self.soh < 1.0 and self.effective_capacity_kwh > 0.0:
            self.soc = min(cfg.soc_max, self.soc * old_capacity / self.effective_capacity_kwh)

        return BatteryStepResult(
            grid_energy_kwh=grid_energy_kwh,
            throughput_kwh=throughput_kwh,
            soh_loss=soh_loss,
            cycle_loss=cycle_loss,
            calendar_loss=calendar_loss,
            soc=self.soc,
            soh=self.soh,
        )
