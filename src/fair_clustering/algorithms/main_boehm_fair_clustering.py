from __future__ import annotations

from typing import Any

import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import time
from fair_clustering.kmedian import kmedian, pairwise_l1

import numpy as np
from fair_clustering import csv_loader
from fair_clustering.evaluate import make_result, evaluate, audit_fairness_exact_balance, plot_pof_comparison, \
    plot_group_pof, \
    plot_cost_breakdown
from fair_clustering.evaluate import plot_execution_times, plot_spatial_clusters, plot_cluster_pof


def encode_groups_to_int(group_series: pd.Series) -> tuple[np.ndarray, list]:
    cats = pd.Categorical(group_series)
    return cats.codes.astype(np.int32), list(cats.categories)


def balance_dataset_for_boehm(df: pd.DataFrame, group_col: str, random_seed) -> tuple[pd.DataFrame, int]:
    group_counts = df[group_col].value_counts()
    size_for_all_groups = group_counts.max()
    rng = np.random.default_rng(random_seed)
    dfs = []
    for group in group_counts.index:
        df_for_group = df[df[group_col] == group].copy()
        df_for_group['_orig_idx'] = df_for_group.index
        original_size = len(df_for_group)

        if original_size < size_for_all_groups:
            base_reps = size_for_all_groups // original_size
            remainder = size_for_all_groups - base_reps * original_size
            repeated = pd.concat([df_for_group] * base_reps, ignore_index=False)
            if remainder > 0:
                extra_positions = rng.choice(original_size, size=remainder, replace=False)
                extra = df_for_group.iloc[extra_positions]
                df_for_group = pd.concat([repeated, extra], ignore_index=False)
            else:
                df_for_group = repeated

        df_for_group['_boehm_weight'] = 1.0
        dfs.append(df_for_group)
    balanced_df = pd.concat(dfs).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return balanced_df, size_for_all_groups


def boehm_fair_clustering(
        x: np.ndarray,
        group_codes,
        k_centers,
        weights: np.ndarray,
        n_trials: int = 5,
        max_iter: int = 30,
        random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray, float]:
    unique_groups, counts = np.unique(group_codes, return_counts=True)
    total_points = len(x)
    group_indices = {g: np.where(group_codes == g)[0] for g in unique_groups}
    x_groups = {g: x[group_indices[g]] for g in unique_groups}
    w_groups = {g: weights[group_indices[g]] for g in unique_groups}

    best_overall_cost = np.inf
    best_overall_labels = None
    best_overall_centers = None

    for baseline_color in unique_groups:
        # to store this trials assignments back to original
        trial_labels = np.full(total_points, -1, dtype=np.int32)
        centers, base_labels, _ = kmedian(
            x=x_groups[baseline_color],
            k=k_centers,
            _weights=w_groups[baseline_color],
            n_trials=n_trials,
            max_iter=max_iter,
            random_seed=random_seed
        )
        trial_labels[group_indices[baseline_color]] = base_labels
        for other_color in unique_groups:
            if other_color == baseline_color:
                continue
            cost_matrix = cdist(x_groups[other_color], x_groups[baseline_color],
                                metric='cityblock')  # cityblock == manhattan
            cost_matrix = cost_matrix * w_groups[other_color][:, np.newaxis]
            row_ind, matched_base_ind = linear_sum_assignment(cost_matrix)
            mapped_labels = base_labels[matched_base_ind]
            trial_labels[group_indices[other_color]] = mapped_labels

        distance_to_all = pairwise_l1(x, centers)
        trial_cost = float((weights * distance_to_all[np.arange(total_points), trial_labels]).sum())
        if trial_cost < best_overall_cost:
            best_overall_cost = trial_cost
            best_overall_centers = centers
            best_overall_labels = trial_labels
    return best_overall_centers, best_overall_labels, best_overall_cost


def evaluate_fairness(
        labels: np.ndarray,
        group_codes: np.ndarray,
        group_names: list,
        k: int,
):
    H = len(group_names)

    violations = 0
    for j in range(k):
        at_j = (labels == j)
        total_j = at_j.sum()
        if total_j == 0:
            continue

        for h, name in enumerate(group_names):
            mass_h = (at_j & (group_codes == h)).sum()
            frac = mass_h / total_j
            expected_frac = 1.0 / H

            # Allow tiny floating point tolerance, though it should be exact integers
            if abs(frac - expected_frac) > 1e-4:
                violations += 1
                status = "⚠ VIOLATION"
            else:
                status = "✓"



def fair_clustering(
        df: pd.DataFrame,
        feature_cols: list[str],
        protected_group_col: str,
        k: int,
        kmedian_trials,
        kmedian_max_iter,
        random_seed
) -> tuple[ndarray, ndarray, float, dict[Any, Any], ndarray, ndarray, float, int, Any, ndarray, list, DataFrame, Any]:
    timing = {}
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    df_balanced, size_pruned_to = balance_dataset_for_boehm(df, group_col=protected_group_col, random_seed=random_seed)
    weights = df_balanced['_boehm_weight'].to_numpy(dtype=np.float64)
    timing["Balance Dataset"] = time.perf_counter() - t0

    x = df_balanced[feature_cols].to_numpy(dtype=np.float64)
    timing["Balance Dataset"] = time.perf_counter() - t0
    x_orig = df[feature_cols].to_numpy(dtype=np.float64)
    w_orig = (df['Weight'].to_numpy(dtype=np.float64)
              if 'Weight' in df.columns
              else np.ones(len(df), dtype=np.float64))
    t0 = time.perf_counter()
    unfair_center, unfair_label, unfair_cost = kmedian(
        x=x_orig,
        k=k,
        _weights=w_orig,
        n_trials=kmedian_trials,
        max_iter=kmedian_max_iter,
        random_seed=random_seed
    )
    timing["Vanilla k-Median"] = time.perf_counter() - t0

    group_codes, group_names = encode_groups_to_int(df_balanced[protected_group_col])
    group_codes_orig, _ = encode_groups_to_int(df[protected_group_col])
    t0 = time.perf_counter()
    fair_centers, fair_labels, fair_cost = boehm_fair_clustering(x, group_codes,
                                                                 k_centers=k,
                                                                 n_trials=kmedian_trials,
                                                                 weights=weights,
                                                                 max_iter=kmedian_max_iter,
                                                                 random_seed=random_seed)
    # majority-vote each original row's label across its replicated copies
    labels_series = pd.Series(fair_labels, index=df_balanced['_orig_idx'].to_numpy())
    fair_labels_orig = (labels_series
                        .groupby(level=0)
                        .agg(lambda s: s.value_counts().idxmax())
                        .reindex(df.index)
                        .to_numpy(dtype=np.int32))
    # Fair cost on the original dataset (apples-to-apples with Bera/Backurs/Bercea)
    D_orig = pairwise_l1(x_orig, fair_centers)
    fair_cost_orig = float((w_orig * D_orig[np.arange(len(df)), fair_labels_orig]).sum())
    timing["Böhm Fair Clustering"] = time.perf_counter() - t0

    evaluate_fairness(fair_labels, group_codes, group_names, k)
    timing["Total Time"] = time.perf_counter() - t_start
    return (fair_centers,
            fair_labels_orig,
            fair_cost_orig,
            timing,
            unfair_center,
            unfair_label,
            unfair_cost,
            size_pruned_to,
            x_orig,
            group_codes_orig,
            group_names,
            df_balanced,
            w_orig
            )


if __name__ == "__main__":
    df = csv_loader.load_csv_chunked(
        "../us_census_puma_data.csv",
        csv_loader.LOAD_COLS,
        max_rows=10_000,
    )

    df_processed = csv_loader.preprocess_dataset(df, ['RACE_BINARY'])

    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"
    K = 10

    (fair_centers, fair_labels, fair_cost, timing, unfair_centers,
     unfair_labels, unfair_cost, size_pruned_to, x, group_codes, group_names, df_balanced, weights) = fair_clustering(
        df_processed,
        feature_cols=FEATURE_COLS,
        protected_group_col=PROTECTED_COL,
        k=K,
        kmedian_trials=5,
        kmedian_max_iter=30,
        random_seed=42,
    )

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

    summary = evaluate(fair_result, unfair_result=unfair_result)

    audit_fairness_exact_balance(fair_result)

    plot_execution_times(fair_result, timing, title="Essential k-Median — Run Time")
    plot_spatial_clusters(df_processed, fair_result, unfair_result, feature_cols=FEATURE_COLS)
    plot_cluster_pof(fair_result, [summary])
    plot_pof_comparison(fair_result, [summary])
    plot_group_pof(fair_result, [summary])
    plot_cost_breakdown(fair_result, [summary])
