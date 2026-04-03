from typing import Callable, Any, NamedTuple

import numpy as np
import pandas as pd

import csv_loader
from evaluate import ClusteringResult, make_result, compute_pof, compute_cluster_costs, \
    compute_cluster_pof
from algorithms.main_boehm_fair_clustering import fair_clustering as bercea_fc
from algorithms.main_bera_fair_clustering import fair_clustering as bera_fc
from algorithms.main_boehm_fair_clustering import fair_clustering as boehm_fc

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
        trial: TrialOutput = result_builder(raw)
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
    rep_trial_timing = all_timings[rep_idx]
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
        timing=rep_trial_timing,
    )

    unfair_result = ClusteringResult(
        algorithm=rep_trial.unfair_result.algorithm + f"_avg{n_runs}",
        centers=rep_trial.unfair_result.centers,
        labels=rep_trial.unfair_result.labels,
        fair_cost=rep_trial.unfair_result.fair_cost,
        unfair_cost=rep_trial.unfair_result.unfair_cost,
        X=rep_trial.unfair_result.X,
        weights=rep_trial.unfair_result.weights,
        group_codes=rep_trial.unfair_result.group_codes,
        group_names=rep_trial.unfair_result.group_names,
        timing=rep_trial_timing,
    )

    # per cluster results
    per_cluster_fair_costs: dict[int, list[float]] = compute_cluster_costs(result)
    per_cluster_unfair_costs: dict[int, list[float]] = compute_cluster_costs(unfair_result)
    cluster_pof = compute_cluster_pof(result, unfair_result)

    valid_fair_cluster_cost = [v for v in per_cluster_fair_costs.values() if v != float('inf')] if per_cluster_fair_costs else []
    valid_unfair_cluster_cost = [v for v in per_cluster_unfair_costs.values() if v != float('inf')] if per_cluster_unfair_costs else []

    # have to ask supervisor what to look for here mby, not sure what to measure/how
    # mby just take the median result as a specific example?
    mean_fair_group_fair = float(np.mean(valid_fair_cluster_cost))
    std_fair_group_fair = float(np.std(valid_fair_cluster_cost))

    mean_unfair_group_fair = float(np.mean(valid_unfair_cluster_cost))
    std_unfair_group_fair = float(np.std(valid_unfair_cluster_cost))


    avg_summary: dict[str, Any] = {
        "Algorithm": result.algorithm,
        "number of runs": n_runs,
        "All results representative_trial": rep_idx + 1,
        "All results Fair Cost (mean)": float(np.mean(fair_costs)),
        "All results Fair Cost (std)": float(np.std(fair_costs, ddof=1 if n_runs > 1 else 0)),
        "All results Fair Cost (min)": float(np.min(fair_costs)),
        "All results Fair Cost (max)": float(np.max(fair_costs)),
        "All results Unfair Cost (mean)": float(np.mean(unfair_costs)),
        "All results Unfair Cost (std)": float(np.std(unfair_costs, ddof=1 if n_runs > 1 else 0)),
        "All results PoF (mean)": float(np.mean(pofs)),
        "All results PoF (std)": float(np.std(pofs, ddof=1 if n_runs > 1 else 0)),
        "All results PoF (min)": float(np.min(pofs)),
        "All results PoF (max)": float(np.max(pofs)),
        "Median Run Cluster Fair Costs (mean)": mean_fair_group_fair,
        "Median Run Cluster Fair Costs (std)": std_fair_group_fair,
        "Median Run Cluster Unfair Costs (mean)": mean_unfair_group_fair,
        "Median Run Cluster Unfair Costs (std)": std_unfair_group_fair,
        "Median Run Timing": rep_trial_timing,
        "Median Fair Trial Fair Cluster Price of Fairness": per_cluster_fair_costs,
        "Median Fair Trial Unfair Cluster Price of Fairness": per_cluster_unfair_costs,
        "Median Fair Trial Clustering Price of Fairness": cluster_pof,
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
    print(f"  TRIAL AVERAGE SUMMARY — {s['Algorithm']}\n")
    print(f"  Trials: {s['number of runs']}  |  "
          f"Representative: #{s['All results representative_trial']}")
    print(f"  Fair Cost   : {s['All results Fair Cost (mean)']:>12,.2f}  "
          f"± {s['All results Fair Cost (std)']:,.2f}  "
          f"[{s['All results Fair Cost (min)']:,.2f} – {s['All results Fair Cost (max)']:,.2f}]")
    print(f"  Unfair Cost : {s['All results Unfair Cost (mean)']:>12,.2f}  "
          f"± {s['All results Unfair Cost (std)']:,.2f}")
    print(f"  PoF         : {s['All results PoF (mean)']:>12.4f}  "
          f"± {s['All results PoF (std)']:.4f}  "
          f"[{s['All results PoF (min)']:.4f} – {s['All results PoF (max)']:.4f}]")
    print(f"  Median Run Cluster Fair Costs (mean): {s['Median Run Cluster Fair Costs (mean)']}\n")
    print(f"  Median Run Cluster Fair Costs (std): {s['Median Run Cluster Fair Costs (mean)']}\n")
    print(f"  Median Run Cluster Unfair Costs (mean): {s['Median Run Cluster Unfair Costs (mean)']}\n")
    print(f"  Median Run Cluster Unfair Costs (std): {s['Median Run Cluster Unfair Costs (std)']}\n")

    print(f" Median Trial Fair Cluster Price of Fairness: \n")
    for cluster_idx, cost in s["Median Fair Trial Fair Cluster Price of Fairness"].items():
        print(f"    {cluster_idx}: {cost}\n")

    print(f" Median Trial Unfair Cluster Price of Fairness: \n")
    for cluster_idx, cost in s["Median Fair Trial Unfair Cluster Price of Fairness"].items():
        print(f"    {cluster_idx}: {cost}\n")

    print(f" Median Fair Trial Clustering Price of Fairness: \n")
    for cluster_idx, cpof in s["Median Fair Trial Clustering Price of Fairness"].items():
        print(f"    {cluster_idx}: {cpof}\n")

    print(f"  Median Run Timing: {s['Median Run Timing']}\n")
    print(f"  Avg Timing: {s['Avg Timing']}\n")
    print(f"  Fair costs: {s['_fair_costs']}\n")
    print(f"  Unfari costs: {s['_unfair_costs']}\n")
    print(f"  All timings: {s['_timings']}\n")

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


if __name__ == "__main__":
    df = csv_loader.load_csv_chunked(
        "us_census_puma_data.csv",
        csv_loader.LOAD_COLS,
        csv_loader.LOAD_DTYPES,
        chunk_size=10_00,
        max_rows=10_00,
    )

    df_processed = csv_loader.preprocess_dataset(df)

    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"
    K = 4
    N_RUNS = 5


    print("\n" + "="*60)
    print("  RUNNING BÖHM ET AL. (Exact Balance / Assignment)")
    print("="*60)
    boehm_result, boehm_summary = run_trials(
        algorithm_fn=boehm_fc,
        result_builder=build_boehm_result,
        df=df_processed,
        n_runs=N_RUNS,
        # Böhm specific kwargs:
        feature_cols=FEATURE_COLS,
        protected_group_col=PROTECTED_COL,
        k=K,
        kmedian_trials=3,
        kmedian_max_iter=30
        # Note: No alpha or weight_col for Böhm
    )