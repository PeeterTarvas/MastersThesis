import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence
import polars as pl

from sklearn.preprocessing import MinMaxScaler

LOAD_COLS = ["Longitude", "Latitude", "PINCP", "AGEP", "SEX", "RAC1P"]

LOAD_DTYPES = {
    "Longitude": "float32",
    "Latitude": "float32",
    "PINCP": "float32",
    "AGEP": "Int16",
    "SEX": "Int8",
    "RAC1P": "Int16",
}

_POLARS_SCHEMA = {
    "Longitude": pl.Float32, "Latitude": pl.Float32, "PINCP": pl.Float32,
    "AGEP":      pl.Int16,   "SEX":      pl.Int8,    "RAC1P": pl.Int16,
}


def load_csv_chunked(
        csv_path: str,
        cols: Sequence[str] = LOAD_COLS,
        max_rows: Optional[int] = None,
        random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load CSV in chunks, keeping memory usage low.
    Drops rows with any NaN in the requested columns.

    Parameters
    ----------
    csv_path    : path to us_census_puma_data.csv
    cols        : columns to load (others are skipped)
    dtypes      : dtype per column (use float32 to halve memory vs float64)
    chunk_size  : rows per chunk (tune down if RAM is tight)
    max_rows    : desired sample size; None → return all clean rows.
    random_seed : If provided, performs vectorised reservoir sampling so that
                  every row in the file has equal probability of appearing in
                  the final sample.  Different seeds produce independent
                  samples; the same seed reproduces the same sample.
                  If None and max_rows is set, returns the first max_rows
                  clean rows (deterministic, fast early-exit).

    Returns
    -------
    pd.DataFrame with up to max_rows rows, all NaN/invalid rows removed.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    schema = {c: _POLARS_SCHEMA[c] for c in cols if c in _POLARS_SCHEMA}
    lf = (
        pl.scan_csv(csv_path, schema_overrides=schema)
        .select(list(cols))
        .drop_nulls()
    )

    df_pl = lf.collect()
    total_clean = len(df_pl)

    rng = np.random.default_rng(random_seed)
    weights = rng.random(total_clean)
    idx = np.argpartition(weights, max_rows)[:max_rows]
    idx = idx[np.argsort(weights[idx])]
    df_pl = df_pl[rng.permutation(idx)]
    df = df_pl.to_pandas()
    return df


GROUP_ID_FEATURES = ['RAC1P', 'SEX', 'AGE_BIN', 'INC_BIN']

def merge_race_6(r):
    if r == 1: return 'White'
    elif r == 2: return 'Black'
    elif r == 6: return 'Asian'
    elif r in [3, 4, 5]: return 'Native/AI'
    elif r == 8: return 'Other'
    elif r in [7, 9]: return 'Multi/PI'
    else: return 'Other'

def preprocess_dataset(df: pd.DataFrame, group_id_features: list[str]):
    df_core = df.copy()
    age_bins = [0, 25, 45, 65, 98]
    age_labels = ['Youth (0-24)', 'Young Adult (25-44)', 'Adult (45-64)', 'Senior (65+)']
    df_core['AGE_BIN'] = pd.cut(df_core['AGEP'], bins=age_bins, labels=age_labels, right=False)

    df_core['INC_BIN'] = pd.cut(df_core['PINCP'], bins=[-np.inf, 15_000, 50_000, 150_000, np.inf], labels=['Low', 'Mid-Low',  'Mid-High', 'High'])

    df_core['AGE_BIN'] = df_core['AGE_BIN'].cat.add_categories('Unknown').fillna('Unknown')
    df_core['INC_BIN'] = df_core['INC_BIN'].cat.add_categories('Unknown').fillna('Unknown')

    df_core['RACE_BINARY'] = df_core['RAC1P'].apply(lambda x: 'White' if x == 1 else 'Non-White')
    df_core['RACE_6'] = df_core['RAC1P'].apply(merge_race_6)

    print("Generating unique 'groups' for intersectional fairness...")

    df_core['GROUP_ID'] = df_core[group_id_features].astype(str).agg('_'.join, axis=1)

    print("3. Extracting and scaling spatial coordinates...")
    scaler = MinMaxScaler()
    spatial_coords = scaler.fit_transform(df_core[['Latitude', 'Longitude']])

    df_core['Lat_Scaled'] = spatial_coords[:, 0]
    df_core['Lon_Scaled'] = spatial_coords[:, 1]
    return df_core
