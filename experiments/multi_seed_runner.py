"""
Multi-Seed Experiment Runner
============================
Runs any experiment configuration across multiple random seeds and saves
aggregated results with formal hypothesis testing via Welch's t-test.

Usage:
    python -m experiments.multi_seed_runner          # 5-seed test
    python -m experiments.multi_seed_runner 30       # full 30-seed run
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import time
from pathlib import Path
from config.simulation_config import SimulationConfig
from experiments.run_simulation import run_simulation


def run_single_seed(base_config_dict: dict, seed: int) -> dict:
    """
    Run the simulation once with the given seed and return key metrics.

    Args:
        base_config_dict: Base configuration dictionary; random_seed will be
                          overridden by *seed*.
        seed: The random seed for this run.

    Returns:
        A dict of key scalar metrics for this seed, including the seed value
        itself for provenance.
    """
    cfg_dict = {**base_config_dict, "random_seed": seed}
    config = SimulationConfig(**cfg_dict)
    state = run_simulation(config)

    store = state.measurement_store
    agents = state.agents

    final_mean = float(np.mean([a.belief_position for a in agents]))
    final_std = float(np.std([a.belief_position for a in agents]))

    tidx_800 = 799
    tidx_1000 = 999

    pre_bias = float(np.mean(
        store.exposure_bias[:len(agents), tidx_800]
    ))
    post_bias = float(np.mean(
        store.exposure_bias[:len(agents), tidx_1000]
    ))

    adapted_fraction = (
        state.intervention_manager
        .calculate_adaptation_rate()
        .get("adapted_fraction", 0.0)
    )

    return {
        "seed": seed,
        "final_mean_belief": final_mean,
        "final_std_belief": final_std,
        "pre_intervention_bias": pre_bias,
        "post_intervention_bias": post_bias,
        "intervention_effect": post_bias - pre_bias,
        "total_anomalies": len(store.anomaly_log),
        "anomalies_per_agent": len(store.anomaly_log) / len(agents),
        "adapted_fraction": adapted_fraction,
        "final_n_agents": len(agents),
        "drift_from_start": final_mean - 5.0,
    }


def run_multi_seed_experiment(
    experiment_name: str,
    base_config_dict: dict,
    n_seeds: int = 30,
    seeds: list = None,
) -> dict:
    """
    Run an experiment across multiple random seeds and aggregate results.

    Args:
        experiment_name: Human-readable label for this experiment condition.
        base_config_dict: Base configuration; random_seed is overridden per seed.
        n_seeds: Number of seeds to use (ignored if *seeds* is provided).
        seeds: Explicit list of seeds. Defaults to range(42, 42 + n_seeds).

    Returns:
        A dict containing per-seed results and cross-seed aggregate statistics
        (mean, std, SE, 95% CI) for every numeric metric.
    """
    if seeds is None:
        seeds = list(range(42, 42 + n_seeds))

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"Running {experiment_name} — seed {seed} ({i + 1}/{len(seeds)})")
        result = run_single_seed(base_config_dict, seed)
        all_results.append(result)

    # Aggregate statistics for every metric except "seed"
    metrics = [k for k in all_results[0].keys() if k != "seed"]
    aggregates = {}
    for metric in metrics:
        values = [r[metric] for r in all_results]
        mean = float(np.mean(values))
        std = float(np.std(values))
        se = float(np.std(values) / np.sqrt(len(values)))
        ci_95_low = float(np.percentile(values, 2.5))
        ci_95_high = float(np.percentile(values, 97.5))
        aggregates[metric] = {
            "mean": mean,
            "std": std,
            "se": se,
            "ci_95_low": ci_95_low,
            "ci_95_high": ci_95_high,
        }

    return {
        "experiment_name": experiment_name,
        "n_seeds": len(seeds),
        "seeds_used": seeds,
        "per_seed_results": all_results,
        "aggregates": aggregates,
    }


def compare_experiments(
    results_a: dict,
    results_b: dict,
    metric: str = "intervention_effect",
) -> dict:
    """
    Welch's t-test comparing two multi-seed experiment results on a metric.

    Args:
        results_a: Output of run_multi_seed_experiment() for condition A.
        results_b: Output of run_multi_seed_experiment() for condition B.
        metric: The per-seed metric key to compare. Defaults to
                "intervention_effect".

    Returns:
        A dict with t-statistic, p-value, Cohen's d effect size, and an
        interpretation of significance and effect size magnitude.
    """
    from scipy import stats

    values_a = [r[metric] for r in results_a["per_seed_results"]]
    values_b = [r[metric] for r in results_b["per_seed_results"]]

    t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)

    # Cohen's d effect size
    pooled_std = np.sqrt(
        (np.std(values_a) ** 2 + np.std(values_b) ** 2) / 2
    )
    cohens_d = (np.mean(values_a) - np.mean(values_b)) / (pooled_std + 1e-8)

    return {
        "metric": metric,
        "experiment_a": results_a["experiment_name"],
        "experiment_b": results_b["experiment_name"],
        "mean_a": float(np.mean(values_a)),
        "mean_b": float(np.mean(values_b)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant": bool(p_value < 0.05),
        "effect_size": (
            "small" if abs(cohens_d) < 0.5
            else "medium" if abs(cohens_d) < 0.8
            else "large"
        ),
    }


def run_all_multi_seed_experiments(n_seeds: int = 5) -> dict:
    """
    Run multi-seed experiments for all five conditions and save results.

    Produces one JSON file per condition in experiments/ and a combined
    statistical_comparisons.json with Welch's t-test results.

    Args:
        n_seeds: Seeds per condition. Use 5 for a quick test, 30 for the
                 final paper run.

    Returns:
        A dict with all per-condition results and pairwise comparisons.
    """
    # ── BASE CONFIG — shared across all experiments ──────────────────────────
    base = {
        "total_timesteps": 1000,
        "burnin_end": 100,
        "checkpoints": [100, 200, 400, 600, 800, 900, 1000],
        "sequential_llm_checkpoints": [1000],
        "use_llm": False,
        "n_agents": 300,
        "agent_distribution": {
            "heavy_imdb": 60,
            "heavy_reddit": 60,
            "heavy_youtube": 60,
            "balanced": 80,
            "cross_platform": 40,
        },
        "new_agent_entry_rate": 1,
    }

    # ── FIVE EXPERIMENT CONFIGURATIONS ───────────────────────────────────────
    run_000_cfg = {**base,
                   "run_id": "run_000_null",
                   "intervention_type": "null",
                   "intervention_start": 1001}

    run_001_cfg = {**base,
                   "run_id": "run_001_div_late",
                   "intervention_type": "diversity_injection",
                   "intervention_start": 800}

    run_002_cfg = {**base,
                   "run_id": "run_002_div_early",
                   "intervention_type": "diversity_injection",
                   "intervention_start": 400}

    run_003_cfg = {**base,
                   "run_id": "run_003_friction",
                   "intervention_type": "algorithm_friction",
                   "intervention_start": 800}

    run_004_cfg = {**base,
                   "run_id": "run_004_bridge",
                   "intervention_type": "social_bridge",
                   "intervention_start": 800}

    # ── RUN ALL CONDITIONS ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"MULTI-SEED EXPERIMENT RUNNER — {n_seeds} seeds per condition")
    print(f"{'='*60}\n")

    t0 = time.time()

    run_000_result = run_multi_seed_experiment(
        "run_000_null", run_000_cfg, n_seeds=n_seeds
    )
    _save_result(run_000_result, "experiments/multiseed_run_000_results.json")

    run_001_result = run_multi_seed_experiment(
        "run_001_div_late", run_001_cfg, n_seeds=n_seeds
    )
    _save_result(run_001_result, "experiments/multiseed_run_001_results.json")

    run_002_result = run_multi_seed_experiment(
        "run_002_div_early", run_002_cfg, n_seeds=n_seeds
    )
    _save_result(run_002_result, "experiments/multiseed_run_002_results.json")

    run_003_result = run_multi_seed_experiment(
        "run_003_friction", run_003_cfg, n_seeds=n_seeds
    )
    _save_result(run_003_result, "experiments/multiseed_run_003_results.json")

    run_004_result = run_multi_seed_experiment(
        "run_004_bridge", run_004_cfg, n_seeds=n_seeds
    )
    _save_result(run_004_result, "experiments/multiseed_run_004_results.json")

    # ── PAIRWISE COMPARISONS ──────────────────────────────────────────────────
    comparisons = []
    for run_result in [run_001_result, run_002_result,
                       run_003_result, run_004_result]:
        comparison = compare_experiments(
            run_000_result,
            run_result,
            metric="intervention_effect",
        )
        comparisons.append(comparison)

    # Also compare run_001 vs run_002 (late vs early timing)
    comparisons.append(compare_experiments(
        run_001_result, run_002_result,
        metric="intervention_effect",
    ))

    with open("experiments/statistical_comparisons.json", "w") as f:
        json.dump(comparisons, f, indent=2)
    print("\nStatistical comparisons saved to "
          "experiments/statistical_comparisons.json")

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PAIRWISE STATISTICAL COMPARISONS (metric: intervention_effect)")
    print(f"{'='*60}")
    for comp in comparisons:
        exp_a = comp["experiment_a"]
        exp_b = comp["experiment_b"]
        mean_a = comp["mean_a"]
        mean_b = comp["mean_b"]
        p_value = comp["p_value"]
        cohens_d = comp["cohens_d"]
        significant = comp["significant"]
        print(
            f"{exp_a} vs {exp_b}: "
            f"mean_a={mean_a:.4f}, mean_b={mean_b:.4f}, "
            f"p={p_value:.4f}, d={cohens_d:.3f}, "
            f"significant={significant}"
        )
    print(f"\nTotal wall-clock time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return {
        "results": {
            "run_000": run_000_result,
            "run_001": run_001_result,
            "run_002": run_002_result,
            "run_003": run_003_result,
            "run_004": run_004_result,
        },
        "comparisons": comparisons,
    }


def _save_result(result: dict, path: str) -> None:
    """Serialize a multi-seed result dict to JSON."""
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {path}")


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_all_multi_seed_experiments(n_seeds=n_seeds)
