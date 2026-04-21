import json
from typing import Callable, Any, NamedTuple

import numpy as np
import pandas as pd

import csv_loader
from evaluate import ClusteringResult, make_result, compute_pof, compute_cluster_costs, \
    compute_cluster_pof, compute_group_costs, compute_gpof
from algorithms.main_bercea_fair_clustering import fair_clustering as bercea_fc
from algorithms.main_bera_fair_clustering import fair_clustering as bera_fc
from algorithms.main_boehm_fair_clustering import fair_clustering as boehm_fc
from algorithms.main_backurs_fair_clustering import fair_clustering as backurs_fc


class TrialOutput(NamedTuple):
    """Everything run_trials needs from one completed trial."""
    fair_result: ClusteringResult
    unfair_result: ClusteringResult
    timing: dict[str, float]



def run_trials(max_rows, algorithm_fn: Callable[..., Any],
                result_builder: Callable[[Any, int], TrialOutput], group_id_features: list[str], n_runs: int, **kwargs):

    fair_costs: list[float] = []
    unfair_costs: list[float] = []
    pofs: list[float] = []
    all_timings: list[dict] = []
    all_cluster_fair_costs: list[dict] = []
    all_cluster_unfair_costs: list[dict] = []
    all_cpofs: list[dict] = []
    all_group_costs_fair: list[dict] = []
    all_group_costs_unfair: list[dict] = []
    all_gpofs: list[dict] = []
    trial_outputs: list[TrialOutput] = []

    master_ss = np.random.SeedSequence(12345)
    child_seeds = master_ss.spawn(n_runs)

    for run_id, child_ss in enumerate(child_seeds):
        seed = int(child_ss.generate_state(1)[0] % (2 ** 31))

        df = csv_loader.load_csv_chunked(
            "../us_census_puma_data.csv",
            csv_loader.LOAD_COLS,
            max_rows=max_rows,
            random_seed=seed
        )
        print(f"  Run {run_id + 1}/{n_runs}  |  shape={df.shape}")

        if df.empty:
            raise ValueError("Dataframe is empty immediately after loading. Check your CSV and load_csv_chunked logic.")
        df = csv_loader.preprocess_dataset(df, group_id_features)

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

        if trial.fair_result.algorithm.lower() in ("bera", "bercea"):
            cluster_fc = compute_cluster_costs(trial.fair_result)
            cluster_uc = compute_cluster_costs(trial.unfair_result)
            cpof = compute_cluster_pof(trial.fair_result, trial.unfair_result)
            all_cluster_fair_costs.append(cluster_fc)
            all_cluster_unfair_costs.append(cluster_uc)
            all_cpofs.append(cpof)

        group_fc = compute_group_costs(trial.fair_result)
        group_uc = compute_group_costs(trial.unfair_result)
        gpof = compute_gpof(group_fc, group_uc)
        all_group_costs_fair.append(group_fc)
        all_group_costs_unfair.append(group_uc)
        all_gpofs.append(gpof)

    all_keys = set().union(*all_timings)
    avg_timing: dict[str, float] = {
        k: float(np.mean([t.get(k, 0.0) for t in all_timings]))
        for k in sorted(all_keys)
    }

    all_group_names = list(all_gpofs[0].keys()) if all_gpofs else []
    gpof_means, gpof_stds = {}, {}
    for g in all_group_names:
        vals = [gp.get(g, float("inf")) for gp in all_gpofs]
        valid = [v for v in vals if v != float("inf")]
        gpof_means[g] = float(np.mean(valid)) if valid else float("inf")
        gpof_stds[g] = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0

    pooled_cpof_values = []
    for cpof_dict in all_cpofs:
        for v in cpof_dict.values():
            if v != float("inf"):
                pooled_cpof_values.append(v)

    cpof_spreads, cpof_ginis = [], []
    for cpof_dict in all_cpofs:
        vals = [v for v in cpof_dict.values() if v != float("inf")]
        if vals:
            cpof_spreads.append(max(vals) - min(vals))
            cpof_ginis.append(_gini(vals))
        else:
            cpof_spreads.append(0.0)
            cpof_ginis.append(0.0)

    gpof_spreads, gpof_ginis = [], []
    for gpof_dict in all_gpofs:
        vals = [v for v in gpof_dict.values() if v != float("inf")]
        if vals:
            gpof_spreads.append(max(vals) - min(vals))
            gpof_ginis.append(_gini(vals))
        else:
            gpof_spreads.append(0.0)
            gpof_ginis.append(0.0)

    avg_summary: dict[str, Any] = {
        "number of runs": n_runs,
        # Global cost / PoF
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
        "Avg Timing": avg_timing,
        # Per-group aggregated metrics
        "G-PoF means": gpof_means,
        "G-PoF stds": gpof_stds,
        "G-PoF spreads": gpof_spreads,
        "G-PoF Ginis": gpof_ginis,
        # Per-cluster aggregated metrics
        "Pooled C-PoF values": pooled_cpof_values,
        "C-PoF spreads": cpof_spreads,
        "C-PoF Ginis": cpof_ginis,
        # Raw per-trial data (for downstream plots)
        "_fair_costs": fair_costs,
        "_unfair_costs": unfair_costs,
        "_pofs": pofs,
        "_timings": all_timings,
        "_all_cpofs": all_cpofs,
        "_all_gpofs": all_gpofs,
        "_all_group_costs_fair": all_group_costs_fair,
        "_all_group_costs_unfair": all_group_costs_unfair,
        "_all_cluster_fair_costs": all_cluster_fair_costs,
        "_all_cluster_unfair_costs": all_cluster_unfair_costs,
    }

    _print_summary(avg_summary)
    return avg_summary

def _gini(values: list[float]) -> float:
    """Gini coefficient for a list of positive values (0 = equal, 1 = max inequality)."""
    arr = np.array(values, dtype=float)
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


def _print_summary(s: dict) -> None:
    n = s["number of runs"]
    print(f"\n{'=' * 60}")
    print(f"  TRIAL SUMMARY  ({n} runs)")
    print(f"{'=' * 60}")
    print(f"  Fair Cost   : {s['All results Fair Cost (mean)']:>12,.2f}  "
          f"± {s['All results Fair Cost (std)']:,.2f}  "
          f"[{s['All results Fair Cost (min)']:,.2f} – {s['All results Fair Cost (max)']:,.2f}]")
    print(f"  Unfair Cost : {s['All results Unfair Cost (mean)']:>12,.2f}  "
          f"± {s['All results Unfair Cost (std)']:,.2f}")
    print(f"  PoF         : {s['All results PoF (mean)']:>12.4f}  "
          f"± {s['All results PoF (std)']:.4f}  "
          f"[{s['All results PoF (min)']:.4f} – {s['All results PoF (max)']:.4f}]")

    if s.get("G-PoF means"):
        print(f"\n  Per-Group G-PoF:")
        for g in s["G-PoF means"]:
            m = s["G-PoF means"][g]
            sd = s["G-PoF stds"][g]
            print(f"    {str(g):20s}: {m:.4f} ± {sd:.4f}")
        gs = s["G-PoF spreads"]
        gg = s["G-PoF Ginis"]
        print(f"  G-PoF Spread: {np.mean(gs):.4f} ± {np.std(gs, ddof=1) if len(gs) > 1 else 0:.4f}")
        print(f"  G-PoF Gini:   {np.mean(gg):.4f} ± {np.std(gg, ddof=1) if len(gg) > 1 else 0:.4f}")

    cs = s.get("C-PoF spreads", [])
    cg = s.get("C-PoF Ginis", [])
    if cs:
        print(f"\n  C-PoF Spread: {np.mean(cs):.4f} ± {np.std(cs, ddof=1) if len(cs) > 1 else 0:.4f}")
        print(f"  C-PoF Gini:   {np.mean(cg):.4f} ± {np.std(cg, ddof=1) if len(cg) > 1 else 0:.4f}")

    print(f"\n  Avg Timing: {s['Avg Timing']}")
    print(f"{'=' * 60}")



def build_bera_result(raw) -> TrialOutput:
    """
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
        size_pruned_to, x, group_codes, group_names, df_balanced, weights
    """
    (fair_centers, fair_labels, fair_cost, timing,
     unfair_centers, unfair_labels, unfair_cost,
     size_pruned_to, x, group_codes, group_names, df_balanced, weights) = raw

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

def build_backurs_result(raw) -> TrialOutput:
    """
    Builder for main_backurs_fair_clustering.fair_clustering().

    Return signature:
        unfair_centers, unfair_labels, unfair_cost,
        fair_centers, fair_labels, fair_cost, timing,
        x, group_codes, group_names,
        lower_bounds, upper_bounds
    """
    (unfair_centers, unfair_labels, unfair_cost,
     fair_centers, fair_labels, fair_cost, timing,
     x, group_codes, group_names,
     lower_bounds, upper_bounds) = raw

    weights = np.ones(len(fair_labels))

    fair_result = make_result(
        algorithm="backurs",
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

    ROW_SIZE = 10_000

    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"
    K = 10
    N_RUNS = 10
    ALPHA = 0.05
    GROUP_ID_FEATURES = ['RAC1P', 'SEX', 'AGE_BIN', 'INC_BIN']

    #print("\n" + "="*60)
    #print("  RUNNING BÖHM ET AL. (Exact Balance / Assignment)")
    #print("="*60)
    #boehm_result, boehm_summary = run_trials(
    #    max_rows=ROW_SIZE,
    #    algorithm_fn=boehm_fc,
    #    result_builder=build_boehm_result,
    #    n_runs=N_RUNS,
    #    # Böhm specific kwargs:
    #    feature_cols=FEATURE_COLS,
    #    protected_group_col=PROTECTED_COL,
    #    k=K,
    #    kmedian_trials=3,
    #    kmedian_max_iter=30
    #)

    print("  RUNNING BERCEA ET AL. (Proportional Bounds)")
    print("=" * 60)
    bera_summary = run_trials(
        max_rows=ROW_SIZE,
        algorithm_fn=bera_fc,
        result_builder=build_bera_result,
        group_id_features=GROUP_ID_FEATURES,
        n_runs=N_RUNS,
        # Bera specific kwargs:
        feature_cols=FEATURE_COLS,
        protected_group_col=PROTECTED_COL,
        k_centers=K,
        alpha=ALPHA,
        weight_col=None
    )

    print("  RUNNING BERA ET AL. (Iterative Rounding)")
    print("=" * 60)
    bercea_summary = run_trials(
        max_rows=ROW_SIZE,
        algorithm_fn=bercea_fc,
        result_builder=build_bercea_result,
        group_id_features=GROUP_ID_FEATURES,
        n_runs=N_RUNS,
        # Bercea specific kwargs:
        feature_cols=FEATURE_COLS,
        protected_group_col=PROTECTED_COL,
        k_cluster=K,
        alpha=ALPHA,
        weight_col=None
    )

