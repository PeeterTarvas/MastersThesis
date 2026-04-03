from typing import Callable, Any, NamedTuple

import numpy as np
import pandas as pd

from evaluate import ClusteringResult, make_result, compute_pof, compute_group_costs, compute_cluster_costs


class TrialOutput(NamedTuple):
    """Everything run_trials needs from one completed trial."""
    fair_result: ClusteringResult
    unfair_result: ClusteringResult
    timing: dict[str, float]


def run_trials(algorithm_fn: Callable[..., Any],
                result_builder: Callable[[Any, int], TrialOutput],
               df: pd.DataFrame, n_runs: int, **kwargs):
    fair_costs: list[float] = []
    unfair_costs: list[float] = []
    pofs: list[float] = []
    all_timings: list[dict] = []
    all_cluster_fair_costs: list[dict] = []
    all_cluster_unfair_costs: list[dict] = []
    trial_outputs: list[TrialOutput] = []

    for run_id in range(n_runs):
        seed = 42 + run_id
        raw = algorithm_fn(df, random_seed=seed, **kwargs)
        trial: TrialOutput = result_builder(raw, seed)
        trial_outputs.append(trial)

        fc = trial.fair_result.fair_cost
        uc = trial.fair_result.unfair_cost
        pof = compute_pof(fc, uc)

        fair_costs.append(fc)
        unfair_costs.append(uc)
        pofs.append(pof)
        all_timings.append(trial.timing)
        all_cluster_fair_costs.append(compute_cluster_costs(trial.fair_result))
        all_cluster_unfair_costs.append(compute_cluster_costs(trial.unfair_result))
        print(f"  → fair_cost={fc:,.2f}  unfair_cost={uc:,.2f}  "
              f"PoF={pof:.4f}")

    # pick representitive result whos cost is closest to median fair cost
    median_cost = float(np.median(fair_costs))
    rep_idx = int(np.argmin(np.abs(np.array(fair_costs) - median_cost)))
    rep_trial = trial_outputs[rep_idx]

    all_keys = set().union(*all_timings)
    avg_timing: dict[str, float] = {
        k: float(np.mean([t.get(k, 0.0) for t in all_timings]))
        for k in all_keys
    }

    result = ClusteringResult(
        algorithm=rep_trial.fair_result.algorithm + f"_avg{n_runs}",
        centers=rep_trial.fair_result.centers,
        labels=rep_trial.fair_result.labels,
        fair_cost=rep_trial.fair_result.fair_cost,
        unfair_cost=rep_trial.fair_result.unfair_cost,
        X=rep_trial.fair_result.X,
        weights=rep_trial.fair_result.weights,
        group_codes=rep_trial.fair_result.group_codes,
        group_names=rep_trial.fair_result.group_names,
        timing=avg_timing,
    )

    # per cluster results
    per_cluster_fair_costs: dict[int, list[float]] = compute_cluster_costs(result)

    # have to ask supervisor what to look for here mby, not sure what to measure/how
    # mby just take the median result as a specific example?
    mean_cluster_unfair_costs: dict[str, float] = {}
    std_cluster_fair_costs: dict[str, float] = {}
    mean_cluster_fair_costs: dict[str, float] = {}
    for cluster_center_idx, points in per_cluster_fair_costs.items():
        vals_fair = [d.get(cluster_center_idx, 0.0) for d in all_cluster_fair_costs]
        vals_unfair = [d.get(cluster_center_idx, 0.0) for d in all_cluster_unfair_costs]
        mean_cluster_fair_costs[cluster_center_idx] = float(np.mean(vals_fair))
        std_cluster_fair_costs[cluster_center_idx] = float(np.std(vals_fair, ddof=1 if n_runs > 1 else 0))
        mean_cluster_unfair_costs[cluster_center_idx] = float(np.mean(vals_unfair))



    avg_summary: dict[str, Any] = {
        "Algorithm": result.algorithm,
        "number of runs": n_runs,
        "representative_trial": rep_idx + 1,
        "Fair Cost (mean)": float(np.mean(fair_costs)),
        "Fair Cost (std)": float(np.std(fair_costs, ddof=1 if n_runs > 1 else 0)),
        "Fair Cost (min)": float(np.min(fair_costs)),
        "Fair Cost (max)": float(np.max(fair_costs)),
        "Unfair Cost (mean)": float(np.mean(unfair_costs)),
        "Unfair Cost (std)": float(np.std(unfair_costs, ddof=1 if n_runs > 1 else 0)),
        "PoF (mean)": float(np.mean(pofs)),
        "PoF (std)": float(np.std(pofs, ddof=1 if n_runs > 1 else 0)),
        "PoF (min)": float(np.min(pofs)),
        "PoF (max)": float(np.max(pofs)),
        "Median Run Cluster Fair Costs (mean)": mean_group_fair,
        "Median Run Cluster Fair Costs (std)": std_group_fair,
        "Median Run Cluster Unfair Costs (mean)": mean_group_unfair,
        "Avg Timing": avg_timing,
        "_fair_costs": fair_costs,
        "_unfair_costs": unfair_costs,
        "_pofs": pofs,
        "_timings": all_timings,
    }
    _print_summary(avg_summary)

    return result, avg_summary


def _print_summary(s: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  TRIAL AVERAGE SUMMARY — {s['Algorithm']}")
    print(f"  Trials: {s['n_trials']}  |  "
          f"Representative: #{s['representative_trial']}")
    print(f"  Fair Cost   : {s['Fair Cost (mean)']:>12,.2f}  "
          f"± {s['Fair Cost (std)']:,.2f}  "
          f"[{s['Fair Cost (min)']:,.2f} – {s['Fair Cost (max)']:,.2f}]")
    print(f"  Unfair Cost : {s['Unfair Cost (mean)']:>12,.2f}  "
          f"± {s['Unfair Cost (std)']:,.2f}")
    print(f"  PoF         : {s['PoF (mean)']:>12.4f}  "
          f"± {s['PoF (std)']:.4f}  "
          f"[{s['PoF (min)']:.4f} – {s['PoF (max)']:.4f}]")
    if s["Group PoFs (mean)"]:
        print("  Group PoFs  :")
        for g, gpof in s["Group PoFs (mean)"].items():
            print(f"    {g:<20s}: {gpof:.4f}")
    print(f"{'='*60}")



def build_bera_result(raw) -> TrialOutput:
    """
    Builder for main_bera_fair_clustering.fair_clustering().

    Return signature:
        unfair_centers, unfair_labels, unfair_cost,
        fair_labels, fair_cost, timing, x, weights,
        group_codes, group_names, lower_bounds, upper_bounds
    """
    (unfair_centers, unfair_labels, unfair_cost,
     fair_labels, fair_cost, timing, x, weights,
     group_codes, group_names, lower_bounds, upper_bounds) = raw

    fair_result = make_result(
        algorithm="bera",
        centers=unfair_centers,
        labels=fair_labels,
        fair_cost=fair_cost,
        unfair_cost=unfair_cost,
        X=x,
        weights=weights,
        group_codes=group_codes,
        group_names=group_names,
        timing=timing,
    )
    unfair_result = make_result(
        algorithm="kmedian-unfair-baseline",
        centers=unfair_centers,
        labels=unfair_labels,
        fair_cost=unfair_cost,
        unfair_cost=unfair_cost,
        X=x,
        weights=weights,
        group_codes=group_codes,
        group_names=group_names,
    )
    return TrialOutput(fair_result=fair_result, unfair_result=unfair_result, timing=timing)


def build_bercea_result(raw) -> TrialOutput:
    """
    Builder for main_bercea_fair_clustering.fair_clustering().

    Return signature:
        centers, unfair_labels, unfair_cost, labels, cost,
        timing, x, weights, group_codes, group_names,
        lower_bounds, upper_bounds
    """
    (centers, unfair_labels, unfair_cost, labels, cost,
     timing, x, weights, group_codes, group_names,
     lower_bounds, upper_bounds) = raw

    fair_result = make_result(
        algorithm="bercea",
        centers=centers,
        labels=labels,
        fair_cost=cost,
        unfair_cost=unfair_cost,
        X=x,
        weights=weights,
        group_codes=group_codes,
        group_names=group_names,
        timing=timing,
    )
    unfair_result = make_result(
        algorithm="kmedian-unfair-baseline",
        centers=centers,
        labels=unfair_labels,
        fair_cost=unfair_cost,
        unfair_cost=unfair_cost,
        X=x,
        weights=weights,
        group_codes=group_codes,
        group_names=group_names,
    )
    return TrialOutput(fair_result=fair_result, unfair_result=unfair_result, timing=timing)


def build_boehm_result(raw) -> TrialOutput:
    """
    Builder for main_boehm_fair_clustering.fair_clustering().

    Return signature:
        fair_centers, fair_labels, fair_cost, timing,
        unfair_center, unfair_label, unfair_cost,
        size_pruned_to, x, group_codes, group_names, df_balanced
    """
    (fair_centers, fair_labels, fair_cost, timing,
     unfair_centers, unfair_labels, unfair_cost,
     size_pruned_to, x, group_codes, group_names, df_balanced) = raw

    weights = np.ones(len(x), dtype=np.float64)

    fair_result = make_result(
        algorithm="boehm",
        centers=fair_centers,
        labels=fair_labels,
        fair_cost=fair_cost,
        unfair_cost=unfair_cost,
        X=x,
        weights=weights,
        group_codes=group_codes,
        group_names=group_names,
        timing=timing,
    )
    unfair_result = make_result(
        algorithm="kmedian-unfair-baseline",
        centers=unfair_centers,
        labels=unfair_labels,
        fair_cost=unfair_cost,
        unfair_cost=unfair_cost,
        X=x,
        weights=weights,
        group_codes=group_codes,
        group_names=group_names,
    )
    return TrialOutput(fair_result=fair_result, unfair_result=unfair_result, timing=timing)