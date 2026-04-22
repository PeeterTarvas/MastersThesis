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
    "AGEP": pl.Int16, "SEX": pl.Int8, "RAC1P": pl.Int16,
}

AGE_BINS = [0, 25, 45, 65, 98]
AGE_LABELS = ['Youth (0-24)', 'Young Adult (25-44)', 'Adult (45-64)', 'Senior (65+)']

INC_BINS = [-np.inf, 15_000, 50_000, 150_000, np.inf]
INC_LABELS = ['Low', 'Mid-Low', 'Mid-High', 'High']

SEX_LABELS = {1: 'Male', 2: 'Female'}


def merge_race_6(r):
    """
    Collapse 9 RAC1P codes into 6 categories (matches thesis §0.3.5 merging rule).

    RAC1P code                                             -> merged label
    1  White                                               -> White
    2  Black or African American                           -> Black
    3  American Indian                                     -> Native
    4  Alaska Native                                       -> Native
    5  American Indian and Alaska Native Tribes            -> Native
    6  Asian                                               -> Asian
    7  Native Hawaiian and Other Pacific Islander          -> Native
    8  Some Other Race                                     -> Other
    9  Two or More Races                                   -> Multi
    """
    if r == 1:
        return 'White'
    elif r == 2:
        return 'Black'
    elif r == 6:
        return 'Asian'
    elif r in [3, 4, 5, 7]:
        return 'Native'
    elif r == 8:
        return 'Other'
    elif r == 9:
        return 'Multi'
    else:
        return 'Other'

def load_csv_chunked(
        csv_path: str,
        cols: Sequence[str] = LOAD_COLS,
        max_rows: Optional[int] = None,
        random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load CSV in chunks, keeping memory usage low.
    Drops rows with any NaN in the requested columns.

    If max_rows is None, returns ALL clean rows (no sampling).
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

    if max_rows is None or max_rows >= total_clean:
        return df_pl.to_pandas()

    rng = np.random.default_rng(random_seed)
    weights = rng.random(total_clean)
    idx = np.argpartition(weights, max_rows)[:max_rows]
    idx = idx[np.argsort(weights[idx])]
    df_pl = df_pl[rng.permutation(idx)]
    return df_pl.to_pandas()


GROUP_ID_FEATURES = ['RAC1P', 'SEX', 'AGE_BIN', 'INC_BIN']


def preprocess_dataset(df: pd.DataFrame,
                       group_id_features: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Create all grouping columns used across the thesis, including intersectional
    attributes for Evaluation 3 (Age x Sex, Race6 x Sex). Produces:

      Marginal attributes:   AGE_BIN, INC_BIN, SEX_LABEL, RACE_BINARY, RACE_6
      Intersectional:        AGE_SEX (L=8), RACE6_SEX (L=12)
      Clustering features:   Lat_Scaled, Lon_Scaled (min-max to [0,1])
      Fair-clustering ID:    GROUP_ID = concatenation of the columns in
                             `group_id_features` (default: ['RACE_6']).

    Any column listed in `group_id_features` must be one of the attributes
    created above (e.g. 'RACE_6', 'RACE_BINARY', 'AGE_SEX', 'RACE6_SEX',
    'INC_BIN', 'AGE_BIN', 'SEX_LABEL'), OR a plain marginal you want to join
    intersectionally (e.g. ['AGE_BIN', 'INC_BIN', 'SEX_LABEL']).
    """
    if group_id_features is None:
        group_id_features = ['RACE_6']

    df_core = df.copy()

    df_core['AGE_BIN'] = pd.cut(df_core['AGEP'], bins=AGE_BINS,
                                labels=AGE_LABELS, right=False)
    df_core['INC_BIN'] = pd.cut(df_core['PINCP'], bins=INC_BINS,
                                labels=INC_LABELS)

    df_core['AGE_BIN'] = df_core['AGE_BIN'].cat.add_categories('Unknown').fillna('Unknown')
    df_core['INC_BIN'] = df_core['INC_BIN'].cat.add_categories('Unknown').fillna('Unknown')

    df_core['SEX_LABEL'] = df_core['SEX'].map(SEX_LABELS).fillna('Unknown')
    df_core['RACE_BINARY'] = df_core['RAC1P'].apply(lambda x: 'White' if x == 1 else 'Non-White')
    df_core['RACE_6'] = df_core['RAC1P'].apply(merge_race_6)

    # ---- intersectional attributes
    # Using '|' as separator because '_' already appears inside AGE_BIN labels,
    # which would make the string ambiguous when split later.
    df_core['AGE_SEX'] = df_core['AGE_BIN'].astype(str) + '|' + df_core['SEX_LABEL'].astype(str)
    df_core['RACE6_SEX'] = df_core['RACE_6'].astype(str) + '|' + df_core['SEX_LABEL'].astype(str)

    # ---- fair-clustering group id -------------------------------------------
    df_core['GROUP_ID'] = df_core[group_id_features].astype(str).agg('|'.join, axis=1)

    # ---- spatial clustering features ----------------------------------------
    scaler = MinMaxScaler()
    spatial_coords = scaler.fit_transform(df_core[['Latitude', 'Longitude']])
    df_core['Lat_Scaled'] = spatial_coords[:, 0]
    df_core['Lon_Scaled'] = spatial_coords[:, 1]

    return df_core
