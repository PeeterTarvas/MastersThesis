import pandas as pd
import numpy as np
import json
from csv_loader import load_csv_chunked

def analyze_data():
    print("Loading data for analysis...")
    df = load_csv_chunked("us_census_puma_data.csv", max_rows=3000000)
    
    stats = {}
    
    print("Calculating statistics...")
    # Age stats
    stats["AGEP_percentiles"] = {str(k): float(v) for k, v in df["AGEP"].quantile([0.1, 0.2, 0.25, 0.33, 0.5, 0.66, 0.75, 0.8, 0.9, 0.95, 0.99]).items()}
    stats["AGEP_min"] = float(df["AGEP"].min())
    stats["AGEP_max"] = float(df["AGEP"].max())
    
    stats["PINCP_percentiles"] = {str(k): float(v) for k, v in df["PINCP"].quantile([0.1, 0.2, 0.25, 0.33, 0.5, 0.66, 0.75, 0.8, 0.9, 0.95, 0.99]).items()}
    stats["PINCP_min"] = float(df["PINCP"].min())
    stats["PINCP_max"] = float(df["PINCP"].max())
    
    stats["SEX_counts"] = {str(k): int(v) for k, v in df["SEX"].value_counts().items()}
    
    stats["RAC1P_counts"] = {str(k): int(v) for k, v in df["RAC1P"].value_counts().items()}
    
    stats["Lat_bounds"] = {"min": float(df["Latitude"].min()), "max": float(df["Latitude"].max())}
    stats["Lon_bounds"] = {"min": float(df["Longitude"].min()), "max": float(df["Longitude"].max())}

    with open("data_summary.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print("Data analysis complete. Results saved to data_summary.json.")

if __name__ == "__main__":
    analyze_data()
