from typing import Callable, Any, NamedTuple

import numpy as np
from collections import defaultdict
import pandas as pd

from evaluate import ClusteringResult, make_result, compute_pof, compute_group_costs, compute_cluster_costs


class TrialOutput(NamedTuple):
    """Everything run_trials needs from one completed trial."""
    fair_result: ClusteringResult       # result from the fair algorithm
    unfair_result: ClusteringResult     # vanilla k-median baseline
    timing: dict[str, float]           # step → seconds


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


    return final_results



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