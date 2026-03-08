import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

# Import your existing baseline
from kmedian import kmedian, pairwise_l1


def compute_optimal_transport(
        X_j: np.ndarray, w_j: np.ndarray,
        X_i: np.ndarray, w_i: np.ndarray
) -> np.ndarray:
    """
    Computes the Earth Mover's Distance (Min-Cost Perfect Matching) between two
    continuous distributions using Linear Programming, as required by Böhm et al.
    """
    N_j = len(X_j)
    N_i = len(X_i)

    # Cost matrix: L1 distance between points in group j and group i
    D = pairwise_l1(X_j, X_i)
    c = D.flatten()

    # Equality constraints: Supply from j must equal demand at i
    # 1. Supply constraints (sum over i for each j = w_j)
    A_eq_j = lil_matrix((N_j, N_j * N_i))
    for j_idx in range(N_j):
        A_eq_j[j_idx, j_idx * N_i: (j_idx + 1) * N_i] = 1

    # 2. Demand constraints (sum over j for each i = w_i)
    A_eq_i = lil_matrix((N_i, N_j * N_i))
    for i_idx in range(N_i):
        A_eq_i[i_idx, i_idx::N_i] = 1

    from scipy.sparse import vstack
    A_eq = vstack([A_eq_j, A_eq_i])
    b_eq = np.concatenate([w_j, w_i])

    # Solve the transport problem
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')

    if not res.success:
        raise ValueError(f"Optimal Transport failed: {res.message}")

    transport_plan = res.x.reshape((N_j, N_i))
    return transport_plan


def compute_boehm_fair_clustering(df_core: pd.DataFrame, k: int, color_col: str):
    """
    Implements Algorithm 1: Fair to Unfair Reduction for (k, p, q)-Clustering.
    """
    groups = df_core[color_col].unique()
    l_colors = len(groups)
    print(f"Running Böhm Reduction across {l_colors} groups: {groups}")

    # Normalize weights so each group has a total weight of exactly 1.0 (Balanced Requirement)
    df_core = df_core.copy()
    for g in groups:
        mask = df_core[color_col] == g
        total_w = df_core.loc[mask, 'Weight'].sum()
        df_core.loc[mask, 'Normalized_Weight'] = df_core.loc[mask, 'Weight'] / total_w

    best_cost = np.inf
    best_centers = None
    best_assignments = None
    best_baseline_group = None

    # Loop over every possible baseline color
    for baseline_g in groups:
        print(f"\n--- Testing Baseline Group: {baseline_g} ---")

        # 1. Run unconstrained clustering on the baseline group ONLY
        mask_i = df_core[color_col] == baseline_g
        X_i = df_core.loc[mask_i, ['Lat_Scaled', 'Lon_Scaled']].values
        w_i = df_core.loc[mask_i, 'Normalized_Weight'].values

        centers_i, labels_i, cost_i = kmedian(X_i, k, w_i)

        total_fair_cost = cost_i
        current_assignments = {baseline_g: labels_i}

        # 2. Map every other group to the baseline group
        for other_g in groups:
            if other_g == baseline_g:
                continue

            mask_j = df_core[color_col] == other_g
            X_j = df_core.loc[mask_j, ['Lat_Scaled', 'Lon_Scaled']].values
            w_j = df_core.loc[mask_j, 'Normalized_Weight'].values

            # Compute Min-Cost Perfect Matching (Earth Mover's Distance)
            transport_plan = compute_optimal_transport(X_j, w_j, X_i, w_i)

            # Assign points in j to the center of their matched point in i
            # Because of fractional transport, a point in j maps to a distribution over centers
            # For strict cost calculation, we multiply transport mass by the distance to the assigned center
            D_j_to_centers = pairwise_l1(X_j, centers_i)

            # transport_plan has shape (N_j, N_i). labels_i maps N_i -> K
            # We route the mass from j -> i -> K
            mass_to_centers = np.zeros((len(X_j), k))
            for j_idx in range(len(X_j)):
                for i_idx in range(len(X_i)):
                    mass = transport_plan[j_idx, i_idx]
                    if mass > 0:
                        center_idx = labels_i[i_idx]
                        mass_to_centers[j_idx, center_idx] += mass

            # Add assignment cost for group j
            j_cost = np.sum(mass_to_centers * D_j_to_centers)
            total_fair_cost += j_cost

            # Save assignments (using greedy argmax for final hard labels)
            current_assignments[other_g] = np.argmax(mass_to_centers, axis=1)

        print(f"Total Fair Cost with baseline {baseline_g}: {total_fair_cost:,.4f}")

        # 3. Output the best fair clustering among all baseline candidates
        if total_fair_cost < best_cost:
            best_cost = total_fair_cost
            best_centers = centers_i
            best_assignments = current_assignments
            best_baseline_group = baseline_g

    print(f"\nOptimal Baseline Color: {best_baseline_group} (Cost: {best_cost:,.4f})")

    # Reassemble the final labels into the dataframe order
    final_labels = np.zeros(len(df_core), dtype=int)
    for g in groups:
        mask = df_core[color_col] == g
        final_labels[mask] = best_assignments[g]

    df_result = df_core.copy()
    df_result['Boehm_Fair_Cluster'] = final_labels

    return best_centers, df_result


if __name__ == "__main__":
    import csv_loader
    from coreset import compute_fair_coreset

    df = csv_loader.load_csv_chunked("us_census_puma_data.csv", max_rows=10_000)
    coreset_df = compute_fair_coreset(df, n_locations=300, random_seed=42)

    # Run the Böhm reduction using 'AGE_BIN' or 'SEX' as the protected groups
    centers, result_df = compute_boehm_fair_clustering(coreset_df, k=30, color_col='AGE_BIN')

    print(result_df[['AGE_BIN', 'Weight', 'Boehm_Fair_Cluster']].head())