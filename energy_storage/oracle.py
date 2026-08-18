"""Oracle benchmarks: optimal dispatch via dynamic programming on a SoC grid.

Two reference points bounding what a policy can earn:

- RollingHorizonOracle: replans every hour over exactly the agent's
  information set (the next 24 day-ahead prices). Optimal decisions, same
  information as SAC — the fair "how close to optimal is the agent" benchmark.
- make_hindsight_policy / evaluate_hindsight: plans once over the episode's
  full realized prices. Valid because the battery is a price-taker (prices on
  a given seed do not depend on actions); an upper bound no causal policy can
  reach.

Both mirror Battery.step economics exactly (efficiency split, cycle stress,
calendar aging, degradation monetized at replacement cost) with SoH frozen at
its current value during planning — it moves ~3e-4 per episode, so the error
is negligible and replanning from true state absorbs it.
"""

import math

import numpy as np

from energy_storage.battery import BatteryConfig
from energy_storage.env import BatteryArbitrageEnv, EnvConfig


class SocGridPlanner:
    """Backward induction over hourly transitions between SoC gridpoints."""

    def __init__(
        self,
        battery: BatteryConfig,
        replacement_cost_per_kwh: float,
        n_soc: int = 101,
    ):
        self.cfg = battery
        self.replacement_cost_per_kwh = replacement_cost_per_kwh
        self.soc_grid = np.linspace(battery.soc_min, battery.soc_max, n_soc)
        self._one_way_eff = math.sqrt(battery.round_trip_efficiency)
        self._soh_loss_per_kwh = (1.0 - battery.eol_soh) / (
            2.0 * battery.cycle_life * battery.capacity_kwh
        )
        self._calendar_loss_per_hour = (1.0 - battery.eol_soh) / (
            battery.calendar_life_years * 8760.0
        )

    def step_economics(self, soc_from, soc_to, soh: float):
        """(grid_energy_kwh, degradation_cost, feasible) for a one-hour move
        soc_from -> soc_to. Broadcasts over array inputs; mirrors Battery.step."""
        cfg = self.cfg
        capacity = cfg.capacity_kwh * soh
        delta = np.asarray(soc_to) - np.asarray(soc_from)
        stored = np.clip(delta, 0.0, None) * capacity
        drawn = np.clip(-delta, 0.0, None) * capacity
        grid_energy = stored / self._one_way_eff - drawn * self._one_way_eff
        power = grid_energy  # dt = 1h, grid-side kW
        feasible = (power <= cfg.max_charge_kw + 1e-9) & (
            power >= -cfg.max_discharge_kw - 1e-9
        )
        power_fraction = np.where(
            power >= 0.0, power / cfg.max_charge_kw, -power / cfg.max_discharge_kw
        )
        cycle_loss = (
            (stored + drawn)
            * self._soh_loss_per_kwh
            * (1.0 + cfg.cycle_stress * power_fraction)
        )
        calendar_loss = self._calendar_loss_per_hour * (
            0.5 + 0.5 * (np.asarray(soc_from) + np.asarray(soc_to))
        )
        degradation_cost = (
            (cycle_loss + calendar_loss) * self.replacement_cost_per_kwh * cfg.capacity_kwh
        )
        return grid_energy, degradation_cost, feasible

    def backward(self, prices, soh: float, terminal_values):
        """Value iteration over the price sequence. Returns (values, choices):
        values[i] is the optimal net value starting the first hour at gridpoint
        i; choices[t][i] is the best next gridpoint index at hour t."""
        g = self.soc_grid
        energy, deg, feasible = self.step_economics(g[:, None], g[None, :], soh)
        values = np.asarray(terminal_values, dtype=float)
        choices = []
        for price in reversed(prices):
            reward = np.where(feasible, -price * energy / 1000.0 - deg, -np.inf)
            q = reward + values[None, :]
            best = np.argmax(q, axis=1)
            choices.append(best)
            values = q[np.arange(len(g)), best]
        choices.reverse()
        return values, choices

    def best_move(self, soc: float, soh: float, price: float, next_values):
        """Best next gridpoint from a (possibly off-grid) SoC given the value
        function for the following hour. Returns (target_soc, grid_power_kw)."""
        energy, deg, feasible = self.step_economics(soc, self.soc_grid, soh)
        q = np.where(feasible, -price * energy / 1000.0 - deg, -np.inf) + next_values
        j = int(np.argmax(q))
        return float(self.soc_grid[j]), float(energy[j])

    def power_to_action(self, power_kw: float) -> np.ndarray:
        cfg = self.cfg
        limit = cfg.max_charge_kw if power_kw >= 0.0 else cfg.max_discharge_kw
        return np.array([np.clip(power_kw / limit, -1.0, 1.0)], dtype=np.float32)


class RollingHorizonOracle:
    """Policy: optimal dispatch over the visible 24h price window, replanned
    every hour. Leftover energy at the window edge is valued at the window's
    median price (net of one-way losses and mid-power cycle degradation) — a
    continuation heuristic, since value past the window is unknowable."""

    def __init__(self, n_soc: int = 101, terminal_price_quantile: float = 0.5):
        self.n_soc = n_soc
        self.terminal_price_quantile = terminal_price_quantile
        self._planner: SocGridPlanner | None = None

    def _terminal_values(self, planner: SocGridPlanner, prices, soh: float):
        cfg = planner.cfg
        price = float(np.quantile(prices, self.terminal_price_quantile))
        deg_per_kwh = (
            planner._soh_loss_per_kwh
            * (1.0 + 0.5 * cfg.cycle_stress)
            * planner.replacement_cost_per_kwh
            * cfg.capacity_kwh
        )
        value_per_kwh = max(0.0, price / 1000.0 * planner._one_way_eff - deg_per_kwh)
        return planner.soc_grid * cfg.capacity_kwh * soh * value_per_kwh

    def __call__(self, env: BatteryArbitrageEnv, obs) -> np.ndarray:
        if self._planner is None:
            self._planner = SocGridPlanner(
                env.config.battery, env.config.replacement_cost_per_kwh, self.n_soc
            )
        planner = self._planner
        prices = env.price_window()
        soh = env.battery.soh
        terminal = self._terminal_values(planner, prices, soh)
        next_values, _ = planner.backward(prices[1:], soh, terminal)
        _, power = planner.best_move(env.battery.soc, soh, float(prices[0]), next_values)
        return planner.power_to_action(power)


def make_hindsight_policy(config: EnvConfig, seed: int, n_soc: int = 101):
    """Policy that plays the hindsight-optimal SoC path for this exact seed.

    Runs a same-seed probe episode to observe the full realized price series
    (actions don't move prices), plans the optimal gridpoint path with zero
    terminal value (leftover energy is worthless at truncation, matching how
    the env scores an episode), then replays it — recomputing each hour's
    power from the *actual* SoC so tiny SoH drift can't accumulate.
    """
    probe = BatteryArbitrageEnv(config)
    probe.reset(seed=seed)
    soc0, soh0 = probe.battery.soc, probe.battery.soh
    idle = np.zeros(1, dtype=np.float32)
    prices, done = [], False
    while not done:
        _, _, terminated, truncated, info = probe.step(idle)
        prices.append(info["price"])
        done = terminated or truncated
    prices = np.asarray(prices)

    planner = SocGridPlanner(config.battery, config.replacement_cost_per_kwh, n_soc)
    next_values, choices = planner.backward(
        prices[1:], soh0, np.zeros(len(planner.soc_grid))
    )
    first_target, _ = planner.best_move(soc0, soh0, float(prices[0]), next_values)
    targets = [first_target]
    i = int(np.argmin(np.abs(planner.soc_grid - first_target)))
    for choice in choices:
        i = int(choice[i])
        targets.append(float(planner.soc_grid[i]))

    step = 0

    def policy(env, obs):
        nonlocal step
        target = targets[min(step, len(targets) - 1)]
        step += 1
        energy, _, _ = planner.step_economics(env.battery.soc, target, env.battery.soh)
        return planner.power_to_action(float(energy))

    return policy


def evaluate_hindsight(
    config: EnvConfig, episodes: int, seed0: int, n_soc: int = 101
) -> dict[str, float]:
    """Mean per-episode economics of the hindsight upper bound; the same
    contract as baselines.evaluate but with a fresh per-seed plan."""
    from energy_storage.baselines import collect_episode

    profits, degradations, rewards = [], [], []
    for ep in range(episodes):
        seed = seed0 + ep
        traj = collect_episode(make_hindsight_policy(config, seed, n_soc), config, seed)
        profits.append(traj["profit"].sum())
        degradations.append(traj["degradation_cost"].sum())
        rewards.append(traj["reward"].sum())
    nets = np.asarray(profits) - np.asarray(degradations)
    return {
        "profit": float(np.mean(profits)),
        "degradation": float(np.mean(degradations)),
        "net": float(np.mean(nets)),
        "net_std": float(np.std(nets, ddof=1)) if len(nets) > 1 else 0.0,
        "reward": float(np.mean(rewards)),
        "nets": nets,
    }
