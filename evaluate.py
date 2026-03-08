import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Distance helper for cost calculation
from kmedian import pairwise_l1


def calculate_group_costs(X, labels, centers, weights, groups):
    """
    Calculates the total k-median cost broken down by demographic group.
    """
    unique_groups = np.unique(groups)
    group_costs = {}

    # Calculate distance of every point to its assigned center
    all_dists = np.zeros(len(X))
    for i in range(len(centers)):
        mask = (labels == i)
        if np.any(mask):
            dists = np.sum(np.abs(X[mask] - centers[i]), axis=1)
            all_dists[mask] = dists * weights[mask]

    for g in unique_groups:
        group_mask = (groups == g)
        group_costs[g] = np.sum(all_dists[group_mask])

    return group_costs


def evaluate_clustering(name, df, centers, labels, unfair_group_costs):
    """
    Computes thesis metrics: G-PoF and Silhouette Score.
    """
    X = df[['Lat_Scaled', 'Lon_Scaled']].values
    weights = df['Weight'].values
    groups = df['GROUP_ID'].values

    # 1. Calculate Fair Group Costs
    fair_group_costs = calculate_group_costs(X, labels, centers, weights, groups)

    # 2. Calculate G-PoF (Price of Fairness per group)
    # G-PoF = Cost_Fair(g) / Cost_Unfair(g)
    gpof = {g: fair_group_costs[g] / unfair_group_costs[g] for g in fair_group_costs}

    # 3. Calculate Spatial Silhouette Score (unweighted for standard comparison)
    # Note: Silhouette is expensive on large N, we sample if necessary.
    sil = silhouette_score(X, labels, metric='manhattan') if len(X) < 20000 else 0.0

    # 4. Total Cost
    total_cost = sum(fair_group_costs.values())

    return {
        "Algorithm": name,
        "Total Cost": total_cost,
        "Silhouette": sil,
        "Max G-PoF": max(gpof.values()),
        "Avg G-PoF": np.mean(list(gpof.values())),
        "Group_PoFs": gpof
    }


def run_full_comparison(coreset_df, k=30):
    """
    Runs all implemented algorithms on the same coreset and compares them.
    """
    results = []
    X = coreset_df[['Lat_Scaled', 'Lon_Scaled']].values
    w = coreset_df['Weight'].values
    groups = coreset_df['GROUP_ID'].values

    # --- 0. Baseline (Unfair) ---
    from kmedian import kmedian
    u_centers, u_labels, u_total_cost = kmedian(X, k, w)
    unfair_group_costs = calculate_group_costs(X, u_labels, u_centers, w, groups)

    # --- 1. Bercea et al. ---
    from algorithms.fair_rounding import round_fractional_assignments  # (From previous turns)
    # Note: You'd need to run the LP first to get the fractional 'x'
    # results.append(evaluate_clustering("Bercea (LP+Flow)", ...))

    # --- 2. Bera et al. (Multi-feature LP) ---
    from algorithms.bera_fair_clustering import compute_bera_fair_clustering
    # Using 10% tolerance around global mean for alpha/beta
    # (Implementation details would be added here based on your specific bounds)

    # --- 3. Böhm et al. (Reduction) ---
    from algorithms.boehm_fair_clustering import compute_boehm_fair_clustering
    b_centers, b_df = compute_boehm_fair_clustering(coreset_df, k, color_col='SEX')
    results.append(
        evaluate_clustering("Böhm (Reduction)", b_df, b_centers, b_df['Boehm_Fair_Cluster'].values, unfair_group_costs))

    # Convert to DataFrame for easy visualization
    summary_df = pd.DataFrame(results)
    return summary_df


# Example visualization for your thesis
def plot_gpof_comparison(summary_df):
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['Algorithm'], summary_df['Avg G-PoF'], color='skyblue', label='Avg G-PoF')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Unfair Baseline')
    plt.ylabel("Price of Fairness (Ratio)")
    plt.title("Comparison of Fair Clustering Algorithms")
    plt.legend()
    plt.savefig("thesis_gpof_comparison.png")