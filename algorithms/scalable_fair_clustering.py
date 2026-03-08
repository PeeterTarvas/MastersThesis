import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from collections import defaultdict

# Import your existing baseline
from kmedian import kmedian


def extract_binary_colors(df: pd.DataFrame, target_col: str) -> np.ndarray:
    """
    Converts a target column into a 0/1 binary color array for the algorithm.
    Example: target_col='SEX' (1=Male, 2=Female) -> colors (0, 1)
    """
    unique_vals = df[target_col].dropna().unique()
    if len(unique_vals) != 2:
        # If not naturally binary, we force a binary split (e.g., median split for income)
        print(f"Column {target_col} has >2 values. Splitting by median to create binary classes.")
        median_val = df[target_col].median()
        colors = (df[target_col] > median_val).astype(int).values
    else:
        # Map existing two values to 0 and 1
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        colors = df[target_col].map(mapping).values

    return colors


def tree_based_fairlet_decomposition(X: np.ndarray, colors: np.ndarray) -> tuple[list, np.ndarray]:
    """
    Phase 1: Computes a fairlet decomposition using a spatial tree[cite: 883].

    Instead of building a complex quadtree from scratch, we use hierarchical
    agglomerative clustering (Ward linkage) to generate an exact binary spatial tree.
    We then traverse bottom-up, matching red (0) and blue (1) points at the lowest
    possible ancestor to minimize spatial distortion.
    """
    n_points = len(X)
    print(f"Building spatial tree for {n_points} points...")

    # Build a binary spatial tree (Z is the linkage matrix)
    Z = linkage(X, method='ward')

    # Keep track of unmatched points at each node
    # nodes 0 to n_points-1 are the original leaves
    unmatched = {i: {'red': [], 'blue': []} for i in range(n_points)}
    for i in range(n_points):
        if colors[i] == 0:
            unmatched[i]['red'].append(i)
        else:
            unmatched[i]['blue'].append(i)

    fairlets = []

    print("Traversing tree bottom-up to extract fairlets...")
    # Traverse the tree bottom-up
    for i, step in enumerate(Z):
        current_node = n_points + i
        left_child = int(step[0])
        right_child = int(step[1])

        # Merge unmatched points from children
        reds = unmatched[left_child]['red'] + unmatched[right_child]['red']
        blues = unmatched[left_child]['blue'] + unmatched[right_child]['blue']

        # Greedily form 1:1 fairlets at this node
        while len(reds) > 0 and len(blues) > 0:
            r_idx = reds.pop()
            b_idx = blues.pop()
            fairlets.append([r_idx, b_idx])

        # Store remaining unmatched points for the parent
        unmatched[current_node] = {'red': reds, 'blue': blues}

        # Free up memory
        del unmatched[left_child]
        del unmatched[right_child]

    # Any remaining points in the root are dropped (or assigned to nearest fairlet)
    # if the dataset is not perfectly 1:1 balanced.

    print(f"Formed {len(fairlets)} exact 1:1 fairlets.")

    # Calculate geometric centers of each fairlet
    fairlet_centers = np.zeros((len(fairlets), X.shape[1]))
    for idx, f in enumerate(fairlets):
        fairlet_centers[idx] = X[f].mean(axis=0)

    return fairlets, fairlet_centers


def compute_scalable_fair_clustering(df: pd.DataFrame, k: int, color_col: str):
    """
    Executes the full two-phase Scalable Fair Clustering algorithm[cite: 882].
    """
    X = df[['Lat_Scaled', 'Lon_Scaled']].values
    colors = extract_binary_colors(df, color_col)

    # --- PHASE 1: Fairlet Decomposition ---
    fairlets, fairlet_centers = tree_based_fairlet_decomposition(X, colors)

    # --- PHASE 2: Cluster the Fairlets ---
    print(f"Phase 2: Running k-median on {len(fairlet_centers)} fairlet centers... ")
    # We assign a weight of 2 to each center since each 1:1 fairlet contains exactly 2 points.
    weights = np.full(len(fairlet_centers), 2.0)

    final_centers, fairlet_labels, fairlet_cost = kmedian(
        X=fairlet_centers,
        k=k,
        _weights=weights
    )

    # --- MAP BACK TO ORIGINAL POINTS ---
    final_point_labels = np.full(len(X), -1)
    for fairlet_idx, fairlet_points in enumerate(fairlets):
        assigned_cluster = fairlet_labels[fairlet_idx]
        for p_idx in fairlet_points:
            final_point_labels[p_idx] = assigned_cluster

    # Calculate total unfair cost for comparison
    _, baseline_labels, baseline_cost = kmedian(X, k, _weights=np.ones(len(X)))

    print(f"\nResults:")
    print(f"  Unfair Baseline Cost: {baseline_cost:,.2f}")
    print(f"  Fair Clustering Cost: {fairlet_cost:,.2f} (approximate)")

    df_result = df.copy()
    df_result['Scalable_Fair_Cluster'] = final_point_labels

    # Filter out any points that couldn't be paired (due to global imbalance)
    df_result = df_result[df_result['Scalable_Fair_Cluster'] != -1]

    return final_centers, df_result


if __name__ == "__main__":
    import csv_loader

    # Note: Because tree building (Phase 1) is O(n log n), you can run this on larger chunks
    # than the LP approach. We load 10,000 points here for a fast test.
    df = csv_loader.load_csv_chunked("../us_census_puma_data.csv", max_rows=10_000)

    # We use SEX as our binary feature
    final_centers, result_df = compute_scalable_fair_clustering(df, k=30, color_col='SEX')

    print(result_df[['SEX', 'Scalable_Fair_Cluster']].head())