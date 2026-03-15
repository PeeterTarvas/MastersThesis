import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess_dataset(df: pd.DataFrame):
    return None

def compute_fair_coreset(
        df: pd.DataFrame,
        n_locations: int = 1000,
        random_seed: int = 42
) -> pd.DataFrame:
    """
    Computes a Fair Coreset by moving points to common spatial locations
    and consolidating weights by demographic 'color'.


    """
    print("Discretizing continuous features into categories...")
    df_core = df.copy()

    if n_locations >= len(df):
        n_locations = len(df)

    #  0-18, 19-35, 36-55, 56+
    df_core['AGE_BIN'] = pd.cut(df_core['AGEP'], bins=[0, 18, 35, 55, 120],
                                labels=['Youth', 'YoungAdult', 'Adult', 'Senior'])

    df_core['INC_BIN'] = pd.cut(df_core['PINCP'], bins=[-np.inf, 35000, 75000, np.inf], labels=['Low', 'Med', 'High'])

    df_core['AGE_BIN'] = df_core['AGE_BIN'].cat.add_categories('Unknown').fillna('Unknown')
    df_core['INC_BIN'] = df_core['INC_BIN'].cat.add_categories('Unknown').fillna('Unknown')

    print("Generating unique 'groups' for intersectional fairness...")
    df_core['GROUP_ID'] = (
            df_core['RAC1P'].astype(str) + "_" +
            df_core['SEX'].astype(str) + "_" +
            df_core['AGE_BIN'].astype(str) + "_" +
            df_core['INC_BIN'].astype(str)
    )

    print("3. Extracting and scaling spatial coordinates...")
    scaler = MinMaxScaler()
    spatial_coords = scaler.fit_transform(df_core[['Latitude', 'Longitude']])

    df_core['Lat_Scaled'] = spatial_coords[:, 0]
    df_core['Lon_Scaled'] = spatial_coords[:, 1]

    print(f"Generating {n_locations} common spatial locations (D1 Sampling)...")
    # We use uniform random sampling for speed on 3M rows to establish the base locations.
    # For a stricter coreset, you can plug in your kmedian_plus_plus_seed here.
    rng = np.random.default_rng(random_seed)
    location_indices = rng.choice(len(df_core), size=n_locations, replace=False)
    centers = spatial_coords[location_indices]
    orig_lats = df_core['Latitude'].values[location_indices]
    orig_lons = df_core['Longitude'].values[location_indices]

    #print("Mapping all points to their nearest spatial location...")
    ## Vectorized L1 distance computation (Manhattan) to match your thesis metric
    ## We process in chunks to prevent RAM overflow on 3M rows
    #chunk_size = 100000
    #labels = np.zeros(len(df_core), dtype=int)

    #for i in range(0, len(df_core), chunk_size):
    #    chunk = spatial_coords[i:i + chunk_size]
    #    dists = np.sum(np.abs(chunk[:, np.newaxis, :] - centers[np.newaxis, :, :]), axis=2)
    #    labels[i:i + chunk_size] = np.argmin(dists, axis=1)

    # --- Step 5: Assign every point to its nearest reference location (L1) -----
    # Memory note: naively broadcasting (n_points, n_locations, 2) at once
    # costs n_points × n_locations × 2 × 4 bytes.  With 100k points and 30k
    # locations that is ~22 GB — way over budget.
    #
    # Fix: tile both axes.  For each points-tile we iterate over location-tiles,
    # tracking only the running best (min_dist, argmin) — O(points_tile ×
    # loc_tile × 2) peak, which we keep to ~256 MB by choosing tile sizes below.
    #
    # tile_points × tile_locs × 2 × 4 bytes ≤ target_bytes
    # With tile_points=5_000 and tile_locs=5_000: 5k×5k×2×4 = 200 MB  ✓
    tile_points = 5_000
    tile_locs   = 5_000
    n_pts = len(spatial_coords)

    point_labels   = np.empty(n_pts, dtype=np.int32)
    for p_start in range(0, n_pts, tile_points):
        p_end  = min(p_start + tile_points, n_pts)
        pts    = spatial_coords[p_start:p_end]          # (tp, 2)
        tp     = len(pts)

        best_dists  = np.full(tp, np.inf, dtype=np.float32)
        best_labels = np.zeros(tp, dtype=np.int32)

        for l_start in range(0, len(centers), tile_locs):
            l_end  = min(l_start + tile_locs, len(centers))
            locs   = centers[l_start:l_end]             # (tl, 2)

            # (tp, tl) — peak allocation: tp × tl × 4 bytes per dim, summed over 2 dims
            d = (np.abs(pts[:, 0:1] - locs[:, 0])      # lat component  (tp, tl)
               + np.abs(pts[:, 1:2] - locs[:, 1]))      # lon component  (tp, tl)

            loc_best_dist  = d.min(axis=1)              # (tp,)
            loc_best_label = d.argmin(axis=1) + l_start # global location index

            improved = loc_best_dist < best_dists
            best_dists[improved]  = loc_best_dist[improved]
            best_labels[improved] = loc_best_label[improved]

        point_labels[p_start:p_end] = best_labels

        if (p_start // tile_points) % 20 == 0:
            print(f"  [Coreset] assigned {min(p_end, n_pts):,} / {n_pts:,} points...", end="\r")

    df_core['Assigned_Center_Idx'] = point_labels

    print("Consolidating points by Center and Group to generate weights...")
    # This matches the paper: "we put p into S' with color i and weight n_{p,i}"
    coreset = df_core.groupby(['Assigned_Center_Idx', 'GROUP_ID']).size().reset_index(name='Weight')
    center_meta = pd.DataFrame({
        'Assigned_Center_Idx': np.arange(n_locations),
        'Lat_Scaled':          centers[:, 0],
        'Lon_Scaled':          centers[:, 1],
        'Latitude':            orig_lats,
        'Longitude':           orig_lons,
    })

    fair_coreset = pd.merge(coreset, center_meta, on='Assigned_Center_Idx', how='left')

    print(
        f"[Coreset] {len(df):,} rows → {len(fair_coreset):,} weighted points "
        f"({fair_coreset['GROUP_ID'].nunique()} unique groups, "
        f"{n_locations:,} locations)"
    )
    return fair_coreset

# --- Execution ---
# final_coreset_df = compute_fair_coreset(final_us_dataset, n_locations=2000)

