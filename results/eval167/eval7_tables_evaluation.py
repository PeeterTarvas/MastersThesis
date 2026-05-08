import json
import numpy as np
import pandas as pd

# Map the algorithm names to the JSON files you uploaded
files = {
    'Bera': 'eval167_bera_summary.json',
    'Bercea': 'eval167_bercea_summary.json',
    'Böhm': 'eval167_boehm_summary.json',
    'Backurs': 'eval167_backurs_summary.json'
}

spread_data = []
gini_data = []


def format_stats(algo, n, array):
    """Calculates statistics and formats them to 3 decimal places."""
    return {
        'Algorithm': algo,
        'N': n,
        'Mean': f"{np.mean(array):.3f}",
        'Std': f"{np.std(array, ddof=1):.3f}" if len(array) > 1 else "0.000",
        'Median': f"{np.median(array):.3f}",
        'Min': f"{np.min(array):.3f}",
        'Max': f"{np.max(array):.3f}"
    }


for algo, file in files.items():
    with open(file, 'r') as f:
        data = json.load(f)

    n_runs = data['number of runs']
    spreads = data['G-PoF spreads']
    ginis = data['G-PoF Ginis']

    spread_data.append(format_stats(algo, n_runs, spreads))
    gini_data.append(format_stats(algo, n_runs, ginis))

# Create DataFrames
spread_df = pd.DataFrame(spread_data)
gini_df = pd.DataFrame(gini_data)

# Print results
print("Updated Table 7 (Spreads):")
print(spread_df.to_string(index=False))
print("\nUpdated Table 8 (Ginis):")
print(gini_df.to_string(index=False))

# Export to CSV (I have already done this in the background)
spread_df.to_csv('table7_spreads.csv', index=False)
gini_df.to_csv('table8_ginis.csv', index=False)
