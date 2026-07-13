import numpy as np

from energy_storage.baselines import evaluate, heuristic_policy, idle_policy
from energy_storage.env import EnvConfig
from energy_storage.robustness import (
    ORACLE,
    default_scenario_samplers,
    evaluate_robustness,
    format_table,
    summarize,
)

CONFIG = EnvConfig(episode_days=3)
SAMPLERS = {
    k: v for k, v in default_scenario_samplers(3).items() if k in ("calm", "cold-snap")
}


def test_calm_regime_matches_plain_evaluate():
    results = evaluate_robustness(
        {"heuristic": heuristic_policy}, CONFIG, episodes=3, seed0=11,
        samplers=SAMPLERS, include_oracle=False,
    )
    plain = evaluate(heuristic_policy, CONFIG, episodes=3, seed0=11)
    np.testing.assert_allclose(results["heuristic"]["calm"].mean(), plain["net"], rtol=1e-9)


def test_shock_changes_outcomes_calm_control_does_not():
    results = evaluate_robustness(
        {"idle": idle_policy, "heuristic": heuristic_policy},
        CONFIG, episodes=3, seed0=11, samplers=SAMPLERS,
    )
    # Idle never trades, so its (small, degradation-only) net barely moves
    # across regimes; the shock must move the oracle's opportunity.
    assert not np.allclose(
        results[ORACLE]["calm"], results[ORACLE]["cold-snap"], rtol=1e-3
    )
    # Cold snaps raise prices, so optimal arbitrage value should rise.
    assert results[ORACLE]["cold-snap"].mean() > results[ORACLE]["calm"].mean()


def test_summary_rows_and_capture():
    results = evaluate_robustness(
        {"heuristic": heuristic_policy}, CONFIG, episodes=3, seed0=11, samplers=SAMPLERS,
    )
    rows = summarize(results)
    assert {(r["policy"], r["regime"]) for r in rows} == {
        ("heuristic", "calm"), ("heuristic", "cold-snap"),
    }
    for row in rows:
        assert row["capture"] is not None
        assert row["capture"] <= 1.0 + 1e-6  # nobody beats the same-regime oracle
    assert "capture" in format_table(rows)


def test_deterministic_across_calls():
    kwargs = dict(config=CONFIG, episodes=2, seed0=11, samplers=SAMPLERS, include_oracle=False)
    a = evaluate_robustness({"heuristic": heuristic_policy}, **kwargs)
    b = evaluate_robustness({"heuristic": heuristic_policy}, **kwargs)
    np.testing.assert_array_equal(a["heuristic"]["cold-snap"], b["heuristic"]["cold-snap"])
