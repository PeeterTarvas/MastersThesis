import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.optimize import linprog
import math

# Import your existing baseline
from kmedian import kmedian, pairwise_l1
import networkx as nx

def fair_assignment_lp(
    X: np.ndarray, 
    weights: np.ndarray, 
    groups: np.ndarray, 
    centers: np.ndarray, 
    margin: float = 0.1
) -> np.ndarray:
    """
    Solves the Fair Assignment LP for fixed centers.
    
    Parameters
    ----------
    X       : (n, d) array of coreset coordinates.
    weights : (n,) array of coreset weights.
    groups  : (n,) array of demographic group labels.
    centers : (k, d) array of baseline center coordinates.
    margin  : Allowed +/- deviation from the global demographic proportion.
    
    Returns
    -------
    assignment_matrix : (n, k) matrix of fractional assignments in [0, 1].
    """
    N = len(X)
    K = len(centers)
    unique_groups = np.unique(groups)
    
    print(f"Setting up Fair Assignment LP for {N} points, {K} centers, {len(unique_groups)} groups...")
    
    # 1. Global Proportions
    total_weight = weights.sum()
    props = {g: weights[groups == g].sum() / total_weight for g in unique_groups}
    
    # 2. Objective Function: minimize distance * weight
    # Flattening order: point 0 to all centers, point 1 to all centers, ...
    D = pairwise_l1(X, centers)  # shape (N, K)
    c = (D * weights[:, None]).flatten()
    
    # 3. Equality Constraints: Every point must be fully assigned (sum over K = 1)
    A_eq = lil_matrix((N, N * K))
    b_eq = np.ones(N)
    for j in range(N):
        A_eq[j, j*K : (j+1)*K] = 1
        
    # 4. Inequality Constraints: Fairness bounds for each center and group
    num_ub_constraints = K * len(unique_groups) * 2
    A_ub = lil_matrix((num_ub_constraints, N * K))
    b_ub = np.zeros(num_ub_constraints)
    
    row_idx = 0
    for i in range(K):
        for g in unique_groups:
            # Relaxed bounds [l_h, u_h]
            l_h = max(0.0, props[g] - margin)
            u_h = min(1.0, props[g] + margin)
            
            # Vectorized constraint generation for the current center and group
            is_g = (groups == g).astype(float)
            
            # Lower bound: sum_{j in g} x_{j,i} w_j >= l_h sum_j x_{j,i} w_j
            # => sum_j (l_h * w_j - I(j in g) * w_j) x_{j,i} <= 0
            lower_coeffs = l_h * weights - is_g * weights
            
            # Upper bound: sum_{j in g} x_{j,i} w_j <= u_h sum_j x_{j,i} w_j
            # => sum_j (I(j in g) * w_j - u_h * w_j) x_{j,i} <= 0
            upper_coeffs = is_g * weights - u_h * weights
            
            # Map into the sparse matrix
            for j in range(N):
                var_idx = j * K + i
                if lower_coeffs[j] != 0:
                    A_ub[row_idx, var_idx] = lower_coeffs[j]
                if upper_coeffs[j] != 0:
                    A_ub[row_idx + 1, var_idx] = upper_coeffs[j]
                    
            row_idx += 2
            
    print("Solving LP with HiGHS interior-point method...")
    # 'highs' is highly optimized for large sparse LPs
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method='highs')
    
    if not res.success:
        raise ValueError(f"LP failed to converge: {res.message}. Try increasing the margin.")
        
    # Reshape back to (N, K)
    assignment_matrix = res.x.reshape((N, K))
    return assignment_matrix

def compute_fair_kmedian(df_core: pd.DataFrame, k: int, margin: float = 0.1):
    """
    Executes the black-box framework for fair k-median clustering.
    """
    X = df_core[['Lat_Scaled', 'Lon_Scaled']].values
    weights = df_core['Weight'].values.astype(float)
    groups = df_core['GROUP_ID'].values
    
    print("Running baseline k-median to discover geometric centers...")
    _centers, unfair_labels, unfair_cost = kmedian(X, k, weights)
    
    assignment_matrix = fair_assignment_lp(X, weights, groups, _centers, margin=margin)
    
    D = pairwise_l1(X, _centers)
    fair_cost = np.sum(assignment_matrix * D * weights[:, None])
    
    print(f"\nResults:")
    print(f"  Unfair Cost : {unfair_cost:,.2f}")
    print(f"  Fair Cost   : {fair_cost:,.2f}")
    
    # For visualization/downstream tasks, we assign each coreset point to the cluster 
    # taking the largest fraction of its weight.
    fair_labels = np.argmax(assignment_matrix, axis=1)
    
    return _centers, fair_labels, assignment_matrix, unfair_cost, fair_cost


def round_fractional_assignments(
        X: np.ndarray,
        centers: np.ndarray,
        fractional_assignments: np.ndarray,
        groups: np.ndarray
) -> np.ndarray:
    """
    Rounds fractional LP assignments to an integral solution using Min-Cost Flow.
    Implements Lemma 8 from Bercea et al. for separable objectives (k-median).

    Parameters
    ----------
    X                      : (n, d) array of data points.
    centers                : (k, d) array of cluster centers.
    fractional_assignments : (n, k) matrix of fractional assignments (the x-variables).
    groups                 : (n,) array of demographic group labels.

    Returns
    -------
    integral_labels : (n,) integer array of final cluster assignments.
    """
    N = len(X)
    K = len(centers)
    unique_groups = np.unique(groups)

    print(f"Building Min-Cost Flow graph for {N} points...")
    G = nx.DiGraph()

    # Precompute pairwise L1 distances for the costs
    distances = pairwise_l1(X, centers)

    # Calculate masses based on fractional assignments
    # mass_h(x, i) = sum of x_ji for points j of color h assigned to center i
    # mass(x, i) = sum of x_ji for all points j assigned to center i
    mass_h = np.zeros((K, len(unique_groups)))
    mass = np.zeros(K)

    for g_idx, g in enumerate(unique_groups):
        mask = (groups == g)
        mass_h[:, g_idx] = fractional_assignments[mask, :].sum(axis=0)

    mass = fractional_assignments.sum(axis=0)

    # --- 1. Add Nodes with Demands (Balances) ---
    # In NetworkX: demand < 0 means supply, demand > 0 means it needs flow.
    # The paper uses b_v = 1 for point nodes, and negative values for sinks.
    # Therefore, P nodes supply -1, and S nodes demand positive amounts.

    # Point nodes (V_P)
    for j in range(N):
        G.add_node(f"P_{j}", demand=-1)

    # Center-Color nodes (V_S^h)
    for i in range(K):
        for g_idx, g in enumerate(unique_groups):
            # Demand is floor(mass_h(x, i))
            d = math.floor(mass_h[i, g_idx])
            G.add_node(f"S_{i}_{g}", demand=d)

    # Center nodes (V_S)
    for i in range(K):
        # Demand is B_i = floor(mass(x, i)) - sum_h floor(mass_h(x, i))
        sum_floor_h = sum(math.floor(mass_h[i, g_idx]) for g_idx in range(len(unique_groups)))
        b_i = math.floor(mass[i]) - sum_floor_h
        G.add_node(f"S_{i}", demand=b_i)

    # Global Sink node (t)
    sum_floor_mass = sum(math.floor(mass[i]) for i in range(K))
    b_t = N - sum_floor_mass
    G.add_node("T", demand=b_t)

    # --- 2. Add Edges with Capacities and Costs ---

    # Edges from Points to Center-Color nodes
    for j in range(N):
        g = groups[j]
        for i in range(K):
            if fractional_assignments[j, i] > 0:
                cost = int(distances[j, i] * 10000)  # Scale float to int for stability
                G.add_edge(f"P_{j}", f"S_{i}_{g}", weight=cost, capacity=1)

    # Edges from Center-Color nodes to Center nodes
    for i in range(K):
        for g_idx, g in enumerate(unique_groups):
            rem = mass_h[i, g_idx] - math.floor(mass_h[i, g_idx])
            if rem > 0:
                # capacity 1, cost 0
                G.add_edge(f"S_{i}_{g}", f"S_{i}", weight=0, capacity=1)

    # Edges from Center nodes to Global Sink
    for i in range(K):
        rem = mass[i] - math.floor(mass[i])
        if rem > 0:
            # capacity 1, cost 0
            G.add_edge(f"S_{i}", "T", weight=0, capacity=1)

    # --- 3. Solve Min-Cost Flow ---
    print("Solving Min-Cost Flow to round assignments...")
    try:
        flow_dict = nx.min_cost_flow(G)
    except nx.NetworkXUnfeasible:
        raise ValueError("Min-Cost flow formulation is unfeasible. Check LP constraints.")

    # --- 4. Extract Integral Assignments ---
    integral_labels = np.zeros(N, dtype=int)
    for j in range(N):
        assigned = False
        # Find where the 1 unit of flow went from this point
        for target, flow in flow_dict[f"P_{j}"].items():
            if flow == 1:
                # Target format is "S_i_g", extract the center index 'i'
                center_idx = int(target.split('_')[1])
                integral_labels[j] = center_idx
                assigned = True
                break
        if not assigned:
            print(f"Warning: Point {j} was not assigned in the flow.")

    return integral_labels

if __name__ == "__main__":
    import csv_loader
    from coreset import compute_fair_coreset
    
    df = csv_loader.load_csv_chunked("../us_census_puma_data.csv", max_rows=100_000)
    coreset_df = compute_fair_coreset(df, n_locations=300, random_seed=42)
    
    # Run the fair pipeline
    centers, fair_labels, assignments, u_cost, f_cost = compute_fair_kmedian(coreset_df, k=30, margin=0.15)
    
    coreset_df['Fair_Cluster'] = fair_labels
    # You can now plot this exactly as you plotted the baseline in kmedian.py