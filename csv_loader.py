import numpy as np
import pandas as pd
import gc
import time
from pathlib import Path
from typing import Optional, Sequence

LOAD_COLS = ["Longitude", "Latitude", "PINCP", "AGEP", "SEX", "RAC1P"]

LOAD_DTYPES = {
    "Longitude": "float32",
    "Latitude":  "float32",
    "PINCP":     "float32",
    "AGEP":      "Int16",
    "SEX":       "Int8",
    "RAC1P":     "Int16",
}

def load_csv_chunked(
    csv_path: str,
    cols: Sequence[str] = LOAD_COLS,
    dtypes: dict = LOAD_DTYPES,
    chunk_size: int = 200_000,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load CSV in chunks, keeping memory usage low.
    Drops rows with any NaN in the requested columns.

    Parameters
    ----------
    csv_path   : path to us_census_puma_data.csv
    cols       : columns to load (others are skipped)
    dtypes     : dtype per column (use float32 to halve memory vs float64)
    chunk_size : rows per chunk (tune down if RAM is tight)
    max_rows   : stop after this many rows (useful for quick experiments)

    Returns
    -------
    df : concatenated DataFrame, all float32
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    chunks = []
    total_loaded = 0
    t0 = time.time()

    reader = pd.read_csv(
        csv_path,
        usecols=cols,
        dtype=dtypes,
        chunksize=chunk_size,
        engine='c'         # faster than pyarrow for chunked reads
    )

    for i, chunk in enumerate(reader):
        chunk = chunk.dropna(subset=cols)

        # Filter out clearly invalid values
        if "Longitude" in chunk.columns:
            chunk = chunk[(chunk["Longitude"] >= -180) & (chunk["Longitude"] <= 180)]
        if "Latitude" in chunk.columns:
            chunk = chunk[(chunk["Latitude"] >= -90) & (chunk["Latitude"] <= 90)]
        if "PINCP" in chunk.columns:
            chunk = chunk[chunk["PINCP"] >= 0]  # drop negative income

        chunks.append(chunk)
        total_loaded += len(chunk)

        elapsed = time.time() - t0
        print(f"  Chunk {i+1}: loaded {total_loaded:,} rows ({elapsed:.1f}s)", end="\r")

        if max_rows is not None and total_loaded >= max_rows:
            break

    print(f"\n  Done: {total_loaded:,} clean rows loaded in {time.time()-t0:.1f}s")

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    if max_rows is not None:
        df = df.iloc[:max_rows].copy()

    return df

def preprocess_dataset(df: pd.DataFrame):
    df_core = df.copy()
    df_core['AGE_BIN'] = pd.cut(df_core['AGEP'], bins=[0, 18, 35, 55, 120],
                                labels=['Youth', 'YoungAdult', 'Adult', 'Senior'])

    df_core['INC_BIN'] = pd.cut(df_core['PINCP'], bins=[-np.inf, 35000, 75000, np.inf], labels=['Low', 'Med', 'High'])

    df_core['AGE_BIN'] = df_core['AGE_BIN'].cat.add_categories('Unknown').fillna('Unknown')
    df_core['INC_BIN'] = df_core['INC_BIN'].cat.add_categories('Unknown').fillna('Unknown')

    print("Generating unique 'groups' for intersectional fairness...")
    df_core['GROUP_ID'] = (
            ##df_core['RAC1P'].astype(str) + "_"# +
            ##df_core['SEX'].astype(str) + "_" +
            df_core['AGE_BIN'].astype(str) + "_"# +
            ##df_core['INC_BIN'].astype(str)
    )

    print("3. Extracting and scaling spatial coordinates...")
    scaler = MinMaxScaler()
    spatial_coords = scaler.fit_transform(df_core[['Latitude', 'Longitude']])

    df_core['Lat_Scaled'] = spatial_coords[:, 0]
    df_core['Lon_Scaled'] = spatial_coords[:, 1]
    return df_core