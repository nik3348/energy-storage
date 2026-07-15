"""Streaming evaluation for the adaptation experiment (docs/adaptation-design.md).

A "deployment" is one continuous market run driven day-by-day: `run_stream`
steps a policy through the hours of each day, then hands the settled day to
the arm's `end_of_day` hook — where adaptive arms append the observed day,
fine-tune the market model, and fine-tune SAC in imagination, all off-asset.
Prices are exogenous, so arms sharing a reset seed see identical markets
(paired comparison) regardless of how they act.

The harness itself needs only numpy; the arms that carry a SAC model import
stable-baselines3 (and the adaptive arm the torch market model) lazily, so
this module stays importable without the 'train' extra.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from energy_storage.env import BatteryArbitrageEnv, EnvConfig
from energy_storage.market_history import (
    MarketHistory,
    ReplayedMarket,
    build_dataset,
    calendar_features,
    norm_price,
    prev_day_stats,
)
from energy_storage.oracle import RollingHorizonOracle


@dataclass
class DayRecord:
    """One settled day of the deployment stream, handed to `end_of_day`."""

    index: int  # day within the stream (0-based); the shift begins at pre_days
    day_of_year: int
    is_weekend: bool
    is_holiday: bool
    prices: np.ndarray  # (24,) realized day-ahead prices
    profit: float
    degradation: float
    # Per-hour (obs, next_obs, action, reward, done, truncated) — the format
    # seed_replay_buffer takes, so model-free arms can refill their buffer.
    transitions: list[tuple]

    @property
    def net(self) -> float:
        return self.profit - self.degradation


@dataclass
class StreamResult:
    """Per-day series for one deployment; extras carries whatever the arm's
    end_of_day hook logged (NaN on days a key wasn't logged)."""

    profit: np.ndarray
    degradation: np.ndarray
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def net(self) -> np.ndarray:
        return self.profit - self.degradation


def run_stream(arm, config: EnvConfig, seed: int) -> StreamResult:
    """Drive one arm through a `config.episode_days`-day deployment.

    The arm needs `act(env, obs) -> action` and
    `end_of_day(record) -> dict | None`."""
    env = BatteryArbitrageEnv(config)
    obs, _ = env.reset(seed=seed)
    profits, degradations = [], []
    extras: dict[str, list[float]] = {}
    terminated = False
    for day_index in range(config.episode_days):
        meta = env.today
        prices, transitions = [], []
        profit = degradation = 0.0
        for _ in range(24):
            action = np.asarray(arm.act(env, obs), dtype=np.float32).reshape(-1)
            next_obs, reward, terminated, truncated, info = env.step(action)
            prices.append(info["price"])
            profit += info["profit"]
            degradation += info["degradation_cost"]
            transitions.append(
                (obs, next_obs, action, float(reward), bool(terminated or truncated), bool(truncated))
            )
            obs = next_obs
            if terminated:
                break
        profits.append(profit)
        degradations.append(degradation)
        record = DayRecord(
            index=day_index,
            day_of_year=int(meta.day_of_year),
            is_weekend=bool(meta.is_weekend),
            is_holiday=bool(meta.is_holiday),
            prices=np.asarray(prices),
            profit=profit,
            degradation=degradation,
            transitions=transitions,
        )
        logs = arm.end_of_day(record) or {}
        for key, value in logs.items():
            extras.setdefault(key, [np.nan] * day_index).append(float(value))
        for key, values in extras.items():
            if len(values) < day_index + 1:
                values.append(np.nan)
        if terminated:
            break
    return StreamResult(
        profit=np.asarray(profits),
        degradation=np.asarray(degradations),
        extras={key: np.asarray(values) for key, values in extras.items()},
    )


class PolicyArm:
    """A frozen policy_fn(env, obs) — baselines, the rolling-horizon oracle,
    or a loaded SAC via baselines.model_policy — with no nightly update."""

    def __init__(self, policy_fn):
        self._policy = policy_fn

    def act(self, env, obs):
        return self._policy(env, obs)

    def end_of_day(self, record: DayRecord) -> dict | None:
        return None


def planner_transitions(config: EnvConfig, days: int, seed0: int) -> list[tuple]:
    """Rolling-horizon planner episodes over a (replayed) market: the
    real-world-legal behavior policy providing demonstration transitions
    for a replay buffer."""
    env = BatteryArbitrageEnv(config)
    oracle = RollingHorizonOracle()
    episodes = max(1, days // config.episode_days)
    transitions = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        while not done:
            action = np.asarray(oracle(env, obs), dtype=np.float32).reshape(-1)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            transitions.append((obs, next_obs, action, reward, done, truncated))
            obs = next_obs
    return transitions


def seed_replay_buffer(model, transitions: list[tuple], n_envs: int) -> int:
    """Add (obs, next_obs, action, reward, done, truncated) tuples to an SB3
    off-policy model's replay buffer, in chunks matching its n_envs."""
    usable = len(transitions) - len(transitions) % n_envs
    for i in range(0, usable, n_envs):
        chunk = transitions[i : i + n_envs]
        model.replay_buffer.add(
            obs=np.stack([t[0] for t in chunk]),
            next_obs=np.stack([t[1] for t in chunk]),
            action=np.stack([t[2] for t in chunk]),
            reward=np.array([t[3] for t in chunk], dtype=np.float32),
            done=np.array([t[4] for t in chunk], dtype=np.float32),
            infos=[{"TimeLimit.truncated": bool(t[5])} for t in chunk],
        )
    return usable


class OnlineFinetuneArm:
    """Model-free adaptation baseline: a trained SAC keeps learning from the
    real stream. Its (fresh) buffer refills at 24 transitions/day and it takes
    gradient steps each night — training starts once the buffer holds a full
    batch (~day 11 at SAC's default batch size, still pre-shift). Acts
    stochastically: that is its only exploration."""

    def __init__(
        self,
        model_path: Path | str,
        env_config: EnvConfig,
        gradient_steps_per_night: int = 240,
        seed: int = 0,
    ):
        from stable_baselines3 import SAC
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.logger import configure

        # Loading with a 1-env vec env rebuilds an empty replay buffer with
        # n_envs=1, matching the one-transition-at-a-time refill below.
        vec1 = make_vec_env(
            lambda: BatteryArbitrageEnv(env_config), n_envs=1, seed=seed
        )
        self.model = SAC.load(model_path, env=vec1)
        self.model.set_logger(configure(None, []))  # .train() needs a logger
        self.model.set_random_seed(seed)
        self.gradient_steps = gradient_steps_per_night

    def act(self, env, obs):
        action, _ = self.model.predict(obs, deterministic=False)
        return action

    def end_of_day(self, record: DayRecord) -> dict:
        seed_replay_buffer(self.model, record.transitions, n_envs=1)
        buffer_size = self.model.replay_buffer.size()
        if buffer_size >= self.model.batch_size:
            self.model.train(
                gradient_steps=self.gradient_steps, batch_size=self.model.batch_size
            )
        return {"buffer_size": float(buffer_size)}


@dataclass
class AdaptiveDynaConfig:
    window_days: int = 90  # trailing real days each nightly model fine-tune sees
    anchor_days: int = 30  # pre-stream days mixed into every fine-tune (anti-forgetting)
    ft_epochs: int = 25
    ft_lr: float = 3e-4
    sac_steps: int = 10_000  # imagined SAC steps per adapting night
    demo_days: int = 7  # trailing real days replayed for nightly planner demos
    buffer_size: int = 100_000  # small buffer: stale imagination should age out
    imagination_episode_days: int = 14
    # Adapt only on nights where the observed day's ensemble NLL exceeds this;
    # None = adapt every night (the phase-1 protocol).
    nll_trigger: float | None = None


class AdaptiveDynaArm:
    """The adaptive arm (docs/adaptation-design.md): nightly, append the
    observed day to the history, fine-tune the ensemble on a trailing window
    (with pre-stream anchor days against forgetting), refresh the replay
    buffer with planner demos on recent real days, fine-tune SAC in
    imagination, and deploy the updated policy tomorrow. Entirely off-asset:
    the real battery only ever executes the current policy."""

    def __init__(
        self,
        sac_path: Path | str,
        ensemble_path: Path | str,
        history: MarketHistory,
        env_config: EnvConfig,
        config: AdaptiveDynaConfig | None = None,
        seed: int = 0,
    ):
        from stable_baselines3 import SAC
        from stable_baselines3.common.buffers import ReplayBuffer
        from stable_baselines3.common.env_util import make_vec_env

        from energy_storage.market_model import LearnedMarket, MarketModelEnsemble

        self.cfg = config or AdaptiveDynaConfig()
        self.env_config = env_config
        self.ensemble = MarketModelEnsemble.load(ensemble_path)
        self.history = history.tail(len(history))  # private copy, grown nightly
        self._seam = len(self.history)  # boundary: pre-collected vs streamed days
        self._prev_prices: np.ndarray | None = None
        self._night = 0

        # The imagination env reads the ensemble by reference, so every
        # episode reset (and every generated day) uses the latest fine-tuned
        # weights — no nightly env rebuild needed.
        imagination_config = self._derived_config(
            episode_days=self.cfg.imagination_episode_days,
            market_factory=lambda mc, s: LearnedMarket(self.ensemble, mc, seed=s),
        )
        vec1 = make_vec_env(
            lambda: BatteryArbitrageEnv(imagination_config), n_envs=1, seed=seed
        )
        self.model = SAC.load(sac_path, env=vec1)
        self.model.replay_buffer = ReplayBuffer(
            self.cfg.buffer_size,
            self.model.observation_space,
            self.model.action_space,
            device=self.model.device,
            n_envs=1,
        )
        self.model.verbose = 0
        self.model.tensorboard_log = None  # else every nightly learn() opens a tb dir
        self.model.set_random_seed(seed)

    def _derived_config(self, episode_days: int, market_factory) -> EnvConfig:
        """The streaming env's battery/market economics with a swapped market
        source and no scenario (shifts reach imagination only through the
        fine-tuned model weights)."""
        base = self.env_config
        return EnvConfig(
            battery=base.battery,
            market=base.market,
            episode_days=episode_days,
            replacement_cost_per_kwh=base.replacement_cost_per_kwh,
            reward_scale=base.reward_scale,
            initial_soc_range=base.initial_soc_range,
            initial_soh_range=base.initial_soh_range,
            burn_in_days=base.burn_in_days,
            market_factory=market_factory,
        )

    def act(self, env, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def end_of_day(self, record: DayRecord) -> dict:
        cap = self.env_config.market.price_cap
        prev = (
            prev_day_stats(self._prev_prices, cap)
            if self._prev_prices is not None
            else np.asarray(self.ensemble.prev_stats_mean, dtype=np.float64)
        )
        x = np.concatenate(
            [
                calendar_features(record.day_of_year, record.is_weekend, record.is_holiday),
                prev,
            ]
        )
        y = norm_price(record.prices, cap)
        logs = {
            "nll": self.ensemble.day_nll(x, y),
            "disagreement": self.ensemble.disagreement(x),
        }
        self._prev_prices = record.prices
        self.history.append_day(
            record.day_of_year, record.is_weekend, record.is_holiday, record.prices
        )

        adapt = self.cfg.nll_trigger is None or logs["nll"] > self.cfg.nll_trigger
        logs["adapted"] = float(adapt)
        if not adapt:
            return logs
        self._night += 1

        # 1. Fine-tune the ensemble: trailing window + pre-stream anchors.
        x_all, y_all = build_dataset(self.history, cap)
        rows = np.arange(len(x_all))
        rows = rows[rows != self._seam - 1]  # drop the pre-history/stream seam pair
        recent = rows[-self.cfg.window_days :]
        pool = rows[rows < self._seam - 1]
        rng = np.random.default_rng(1000 + self._night)
        n_anchor = min(self.cfg.anchor_days, len(pool))
        anchors = (
            rng.choice(pool, size=n_anchor, replace=False)
            if n_anchor
            else np.empty(0, dtype=int)
        )
        sel = np.unique(np.concatenate([recent, anchors]))
        self.ensemble.fine_tune(
            x_all[sel],
            y_all[sel],
            epochs=self.cfg.ft_epochs,
            lr=self.cfg.ft_lr,
            seed=self._night,
        )
        logs["nll_post"] = self.ensemble.day_nll(x, y)

        # 2. Planner demos on the most recent real days into the buffer.
        if self.cfg.demo_days > 1 and len(self.history) >= self.cfg.demo_days:
            tail = self.history.tail(self.cfg.demo_days)
            demo_config = self._derived_config(
                episode_days=self.cfg.demo_days,
                market_factory=lambda mc, s: ReplayedMarket(tail, mc),
            )
            demos = planner_transitions(
                demo_config, days=self.cfg.demo_days, seed0=50_000 + self._night
            )
            seed_replay_buffer(self.model, demos, n_envs=1)

        # 3. Fine-tune SAC in (now shift-aware) imagination; deploy tomorrow.
        self.model.learn(
            total_timesteps=self.cfg.sac_steps,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        return logs
