import pandas as pd
import numpy as np
import json
from csv_loader import load_csv_chunked

BIN_EDGES = [0, 25, 45, 65, 120]
BIN_LABELS = ['Youth', 'YoungAdult', 'Adult', 'Senior']
INCOME_BIN_EDGES = [0, 15_000, 50_000, 150_000, float('inf')]
INCOME_BIN_LABELS = ['Low Income', 'Working Class', 'Middle Class', 'High Income']
INCOME_BIN_DESCRIPTIONS = {
    'Low Income':     '$0 – $15,000   (at/below federal poverty guideline)',
    'Working Class':  '$15k – $50k    (below U.S. median household income)',
    'Middle Class':   '$50k – $150k   (Pew Research middle-income tier)',
    'High Income':    '$150k+         (Pew Research upper-income tier, ~top 5%)',
}

def analyze_data():
    print("Loading data for analysis...")
    df = load_csv_chunked("us_census_puma_data.csv", max_rows=3000000)

    stats = {}

    print("Calculating statistics...")
    # Age stats
    stats["AGEP_percentiles"] = {str(k): float(v) for k, v in df["AGEP"].quantile(
        [0.1, 0.2, 0.25, 0.33, 0.5, 0.66, 0.75, 0.8, 0.9, 0.95, 0.99]).items()}
    stats["AGEP_min"] = float(df["AGEP"].min())
    stats["AGEP_max"] = float(df["AGEP"].max())

    # Age binning
    df["AGE_BIN"] = pd.cut(df["AGEP"], bins=BIN_EDGES, labels=BIN_LABELS, right=False)
    total = len(df)
    age_bin_stats = {}
    for label in BIN_LABELS:
        count = int((df["AGE_BIN"] == label).sum())
        age_bin_stats[label] = {
            "count": count,
            "percentage": round(count / total * 100, 2)
        }
    stats["AGE_BIN_stats"] = age_bin_stats

    stats["PINCP_percentiles"] = {str(k): float(v) for k, v in df["PINCP"].quantile(
        [0.1, 0.2, 0.25, 0.33, 0.5, 0.66, 0.75, 0.8, 0.9, 0.95, 0.99]).items()}
    stats["PINCP_min"] = float(df["PINCP"].min())
    stats["PINCP_max"] = float(df["PINCP"].max())

    df["INCOME_BIN"] = pd.cut(
        df["PINCP"],
        bins=INCOME_BIN_EDGES,
        labels=INCOME_BIN_LABELS,
        right=False,
        include_lowest=True,
    )
    income_bin_stats = {}
    for label in INCOME_BIN_LABELS:
        count = int((df["INCOME_BIN"] == label).sum())
        income_bin_stats[label] = {
            "count": count,
            "percentage": round(count / total * 100, 2),
            "description": INCOME_BIN_DESCRIPTIONS[label],
        }
    stats["INCOME_BIN_stats"] = income_bin_stats

    stats["SEX_counts"] = {str(k): int(v) for k, v in df["SEX"].value_counts().items()}

    stats["RAC1P_counts"] = {str(k): int(v) for k, v in df["RAC1P"].value_counts().items()}

    stats["Lat_bounds"] = {"min": float(df["Latitude"].min()), "max": float(df["Latitude"].max())}
    stats["Lon_bounds"] = {"min": float(df["Longitude"].min()), "max": float(df["Longitude"].max())}

    with open("data_summary.json", "w") as f:
        json.dump(stats, f, indent=4)

    print("\n=== Age Bin Summary ===")
    for label, s in age_bin_stats.items():
        print(f"  {label:12s}: {s['count']:>8,}  ({s['percentage']:.2f}%)")
    print(f"  {'TOTAL':12s}: {total:>8,}")
    print("\nData analysis complete. Results saved to data_summary.json.")


if __name__ == "__main__":
    analyze_data()