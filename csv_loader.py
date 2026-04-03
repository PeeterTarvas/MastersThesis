import numpy as np
import pandas as pd
import gc
import time
from pathlib import Path
from typing import Optional, Sequence

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


def load_csv_chunked(
        csv_path: str,
        cols: Sequence[str] = LOAD_COLS,
        dtypes: dict = LOAD_DTYPES,
        chunk_size: int = 200_000,
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

    do_random = (random_seed is not None) and (max_rows is not None)
    rng = np.random.default_rng(random_seed) if do_random else None

    reservoir: Optional[pd.DataFrame] = None
    chunks: list[pd.DataFrame] = []   # used only in deterministic path
    total_clean = 0
    t0 = time.time()

    reader = pd.read_csv(
        csv_path,
        usecols=cols,
        dtype=dtypes,
        chunksize=chunk_size,
        engine="c",
    )

    for i, chunk in enumerate(reader):
        # ---- clean -------------------------------------------------------
        chunk = chunk.dropna(subset=cols)
        if "Longitude" in chunk.columns:
            chunk = chunk[(chunk["Longitude"] >= -180) & (chunk["Longitude"] <= 180)]
        if "Latitude" in chunk.columns:
            chunk = chunk[(chunk["Latitude"] >= -90)  & (chunk["Latitude"] <= 90)]
        if "PINCP" in chunk.columns:
            chunk = chunk[chunk["PINCP"] >= 0]
        if len(chunk) == 0:
            continue

        total_clean += len(chunk)

        # ---- random reservoir sampling -----------------------------------
        if do_random:
            chunk = chunk.copy()                        # avoid SettingWithCopyWarning
            chunk["_w"] = rng.random(size=len(chunk))  # uniform weight per row
            if reservoir is None:
                reservoir = chunk
            else:
                reservoir = pd.concat([reservoir, chunk], ignore_index=True)
            if len(reservoir) > max_rows:
                reservoir = reservoir.nsmallest(max_rows, "_w")

        # ---- deterministic path: take first max_rows rows ----------------
        else:
            if max_rows is not None:
                remaining = max_rows - sum(len(c) for c in chunks)
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]
            chunks.append(chunk)

        print(
            f"  Chunk {i+1:>3}: {total_clean:>10,} clean rows  ({time.time()-t0:.1f}s)",
            end="\r",
        )

    print(f"\n  Done: {total_clean:,} clean rows in {time.time()-t0:.1f}s")

    # ---- assemble result -------------------------------------------------
    if do_random:
        if reservoir is None:
            return pd.DataFrame(columns=list(cols))
        df = reservoir.drop(columns=["_w"])
        # Shuffle to remove the weight-sorted order nsmallest leaves behind
        df = df.sample(frac=1.0, random_state=int(rng.integers(2**31))).reset_index(drop=True)
    else:
        if not chunks:
            return pd.DataFrame(columns=list(cols))
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()

    print(f"  Final sample: {len(df):,} rows")
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
            df_core['AGE_BIN'].astype(str) + "_"  # +
        ##df_core['INC_BIN'].astype(str)
    )

    print("3. Extracting and scaling spatial coordinates...")
    scaler = MinMaxScaler()
    spatial_coords = scaler.fit_transform(df_core[['Latitude', 'Longitude']])

    df_core['Lat_Scaled'] = spatial_coords[:, 0]
    df_core['Lon_Scaled'] = spatial_coords[:, 1]
    return df_core
