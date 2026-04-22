import pandas as pd
import json
from fair_clustering.csv_loader import load_csv_chunked

BIN_EDGES = [0, 25, 45, 65, 120]
BIN_LABELS = ['Youth', 'YoungAdult', 'Adult', 'Senior']
INCOME_BIN_EDGES = [0, 15_000, 50_000, 150_000, float('inf')]
INCOME_BIN_LABELS = ['Low Income', 'Working Class', 'Middle Class', 'High Income']
INCOME_BIN_DESCRIPTIONS = {
    'Low Income':     '$0 – $15,000',
    'Working Class':  '$15k – $50k',
    'Middle Class':   '$50k – $150k',
    'High Income':    '$150k+',
}
## https://data.census.gov/app/mdat/ACSPUMS5Y2023/vars?cv=RC(1),HISP_RC2,RAC1P_RC1&rv=POVPIP_RC1&nv=AGEP_RC1,ucgid&wt=PWGTP&g=AwFm-BVBlBGWg&HISP_RC2=N4IgyiBcIEoKYGMD2ATOACAZkgTu~yaK6aALgIYCWANnMQBKUDOADuQHaULq6UDmldiAA0sKCHoBJMAAURIAGpQA2spAAGAIzyAcklLoAotSZwA7gAs4ODAHEcSAK4s6I0jkdwAusLXqATMLqAMxBACxBAKxBAGxBAOxBABxBAJzCmuoZmhmBmqGaEZrRmnGaiZopmun~Wf45-oH~of5h8oysHFzoABSCCNSOKIJ86GBsnEwWAJRuHt5eAL5AA&RAC1P_RC1=N4IgyiBcIEoKYGMD2ATOACAZkgTu~yaK6aALgIYCWANnMTuQhoRgBQBySpckJArhlJJ0AZwAOjDDjgjSOSglKUkAOxEAaUUgC2GauQBGcaiPTbyAT3RH0cvioTluKAHQBKEOthRYAQQDCAIwACp4gAGpQANpRIADM6gAs6gCs6gBs6gDs6gAc6gCcYb4ilOQqmtp81EoMCGXUmipOlABucJq46AA6IFwAFnA4vZ52cAC66rGBYQDq-ZTc6OTUqnCjOAKTsQBMYQBC~ggA1surKuvqY~PjAL5AA&POVPIP_RC1=N4IgyiBcIEoKYGMD2ATOACAZkgTugkgHbIC2cAtAC5LkAOSAbnDpQJ7o4CGlAlkh4lRwQAGlhQQABQDyANUn5JokLKgBtNSHIBGEdsgBmAAxHIAVhMiL25QDkkldAFEANgGc4AdwAWzDAHEcJABXWjgUUUocYLgAXRFNfVMAJgBOVOUjdGp0NIyRKJjY2IBfIA&AGEP_RC1=N4IgyiBcIEoKYGMD2ATOACAZkgTugggOZwgA0sUI~A4gKIAKZIAalANpsgCMkXAHJACcgpgDkkAF3S0ANgGc4AdwAWcHBmo4kAVwAOcFGUwBDeXAC6pTgAZSPewHYm19BKTouT0hJzaL5gF8gA

RAC1P_LABELS = {
    1: 'White',
    2: 'Black or African American',
    3: 'American Indian',
    4: 'Alaska Native',
    5: 'American Indian and Alaska Native Tribes',
    6: 'Asian',
    7: 'Native Hawaiian and Other Pacific Islander',
    8: 'Some Other Race',
    9: 'Two or More Races'
}

def analyze_data():
    print("Loading data for analysis...")
    df = load_csv_chunked("us_census_puma_data.csv")

    stats = {}

    print("Calculating statistics...")
    stats["AGEP_percentiles"] = {str(k): float(v) for k, v in df["AGEP"].quantile(
        [0.2, 0.4, 0.6, 0.8]).items()}
    stats["AGEP_min"] = float(df["AGEP"].min())
    stats["AGEP_max"] = float(df["AGEP"].max())

    df["AGE_BIN"] = pd.cut(df["AGEP"], bins=BIN_EDGES, labels=BIN_LABELS, right=False)
    total = len(df)
    age_bin_stats = {}
    stats["Total_data"] = total

    for label in BIN_LABELS:
        count = int((df["AGE_BIN"] == label).sum())
        age_bin_stats[label] = {
            "count": count,
            "percentage": round(count / total * 100, 2)
        }
    stats["AGE_BIN_stats"] = age_bin_stats


    stats["PINCP_min"] = float(df["PINCP"].min())
    stats["PINCP_max"] = float(df["PINCP"].max())

    income_quantiles = {str(k): float(v) for k, v in df["PINCP"].quantile(
        [0.2, 0.4, 0.6, 0.8]).items()}

    income_quantiles_bins = [*income_quantiles.values(), df["PINCP"].max()]
    print(income_quantiles_bins)
    income_bins = pd.cut(df["PINCP"], bins=income_quantiles_bins, labels=income_quantiles.keys(), right=False, include_lowest=True)

    income_bin_stats_quantiles = {}
    for key, value in income_quantiles.items():
        count = int((income_bins == key).sum())
        income_bin_stats_quantiles[f"{key}"] = {
            "count": count,
            "percentage": round(count / total * 100, 2),
            "bin":  value
        }

    stats["INCOME_BIN_QUANTILES_stats"] = income_bin_stats_quantiles


    income_quartiles = {str(k): float(v) for k, v in df["PINCP"].quantile(
        [0.25, 0.5, 0.75]).items()}
    stats["PINCP_quartiles"] = income_quartiles


    income_quartiles_bins = [*income_quartiles.values(), df["PINCP"].max()]
    income_bins = pd.cut(df["PINCP"], bins=income_quartiles_bins, labels=income_quartiles.keys(), right=False, include_lowest=True)

    income_bin_stats_quartiles = {}
    for key, value in income_quartiles.items():
        count = int((income_bins == key).sum())
        income_bin_stats_quartiles[f"{key}"] = {
            "count": count,
            "percentage": round(count / total * 100, 2),
            "bin":  value
        }

    stats["INCOME_BIN_QUARTILE_stats"] = income_bin_stats_quartiles

    df["INCOME_BIN"] = pd.cut(
        df["PINCP"],
        bins=INCOME_BIN_EDGES,
        labels=INCOME_BIN_LABELS,
        right=False,
        include_lowest=True
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

    race_counts = {}
    for k, v in df["RAC1P"].value_counts().items():
        count = int((df["RAC1P"] == k).sum())
        race_counts[RAC1P_LABELS[k]] = {
            "count": count,
            "percentage": round(count / total * 100, 2),
        }

    stats["RAC1P_counts"] = race_counts
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