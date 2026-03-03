"""
Fair Clustering Dataset Utilities
==================================
Two dataset sources for evaluating fair multi-attribute clustering:
  1. Synthetic data  — 3 protected attributes: Race(3) × Sex(2) × EducationTier(2) = 12 groups
  2. Folktables ACS  — 3 protected attributes: same structure, genuinely continuous features

The number of protected attributes is controlled by `n_protected_attrs` (2 or 3).
Using 3 attributes directly stresses balance constraints (more groups = harder feasibility)
and lets you study how group count affects G-PoF — directly relevant to your thesis.

Clustering features are always spatial/socioeconomic — never the protected attributes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def load_folktables_fair_clustering(
    states: list = ["CA"],
    survey_year: str = "2018",
    n_protected_attrs: int = 3,
    target_samples: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load ACS person-level data and prepare it for fair multi-attribute clustering.

    Why the previous version had stripe artefacts
    ----------------------------------------------
    WKHP (hours/week) and JWMNP (commute minutes) are integers that bunch at round
    numbers (40hrs, 60min), and MinMaxScaler preserves that discreteness. This version:
      - Log-transforms PINCP and PERNP (right-skewed income -> more symmetric)
      - Adds small Gaussian jitter (std=0.5) to integer-valued columns to break the grid
      - Uses POVPIP (income-to-poverty ratio) which is already a continuous ratio

    Protected attributes
    --------------------
    Race         : RAC1P collapsed to 3 groups (White=0 / Black=1 / Other=2)
    Sex          : SEX recoded to 0=Male / 1=Female
    EducationTier: SCHL binarised at median (0=below, 1=above) — only if n_protected_attrs=3

    Clustering features (all scaled [0,1], never the protected attributes)
    -----------------------------------------------------------------------
        Income       — log(PINCP), continuous
        Earnings     — log(PERNP), continuous
        PovertyRatio — POVPIP,     continuous ratio
        Age          — AGEP + jitter
        HoursPerWeek — WKHP + jitter
        CommuteTime  — JWMNP + jitter
    """
    if n_protected_attrs not in (2, 3):
        raise ValueError("n_protected_attrs must be 2 or 3.")

    try:
        from folktables import ACSDataSource
    except ImportError:
        raise ImportError("Run: pip install folktables")

    rng = np.random.default_rng(random_state)

    print(f"Downloading ACS {survey_year} data for states: {states} ...")
    data_source = ACSDataSource(
        survey_year=survey_year, horizon="1-Year", survey="person"
    )
    raw = data_source.get_data(states=states, download=True)

    cols_needed = ["AGEP", "WKHP", "SCHL", "PINCP", "PERNP", "POVPIP", "JWMNP", "RAC1P", "SEX"]
    df = raw[cols_needed].dropna(subset=cols_needed).copy()

    df = df[
        (df["AGEP"] >= 18) & (df["AGEP"] <= 65) &
        (df["WKHP"] >= 1) &
        (df["PINCP"] > 0) & (df["PINCP"] < 500_000) &
        (df["PERNP"] > 0) &
        (df["POVPIP"].notna()) &
        (df["JWMNP"] > 0) & (df["JWMNP"] < 200)
    ].copy()

    print(f"  After filtering: {len(df):,} rows")

    # Protected attributes
    df["Race"] = df["RAC1P"].apply(lambda r: 0 if r == 1 else (1 if r == 2 else 2))
    df["Sex"]  = df["SEX"].astype(int) - 1

    if n_protected_attrs == 3:
        edu_median = df["SCHL"].median()
        df["EducationTier"] = (df["SCHL"] >= edu_median).astype(int)
        df["Protected_Group"] = (
            "R" + df["Race"].astype(str) +
            "_S" + df["Sex"].astype(str) +
            "_E" + df["EducationTier"].astype(str)
        )
    else:
        df["Protected_Group"] = (
            "R" + df["Race"].astype(str) +
            "_S" + df["Sex"].astype(str)
        )

    # Build continuous clustering features
    df["log_Income"]   = np.log1p(df["PINCP"])
    df["log_Earnings"] = np.log1p(df["PERNP"])
    df["Age_j"]        = df["AGEP"].astype(float)  + rng.normal(0, 0.5, size=len(df))
    df["Hours_j"]      = df["WKHP"].astype(float)  + rng.normal(0, 0.5, size=len(df))
    df["Commute_j"]    = df["JWMNP"].astype(float) + rng.normal(0, 0.5, size=len(df))

    feature_cols = ["log_Income", "log_Earnings", "POVPIP", "Age_j", "Hours_j", "Commute_j"]
    df[feature_cols] = MinMaxScaler().fit_transform(df[feature_cols])

    df = df.rename(columns={
        "log_Income":   "Income",
        "log_Earnings": "Earnings",
        "POVPIP":       "PovertyRatio",
        "Age_j":        "Age",
        "Hours_j":      "HoursPerWeek",
        "Commute_j":    "CommuteTime",
    })

    # Stratified subsample
    if target_samples and len(df) > target_samples:
        df = (
            df.groupby("Protected_Group", group_keys=False)
              .apply(lambda g: g.sample(
                  frac=target_samples / len(df),
                  random_state=random_state
              ))
        )
        print(f"  Subsampled to {len(df):,} rows (stratified by Protected_Group)")

    df = df.reset_index(drop=True)

    print("\nProtected group distribution:")
    summary = df["Protected_Group"].value_counts().sort_index()
    print(summary.to_string())

    if summary.min() < 100:
        print(f"\n  WARNING: smallest group has only {summary.min()} points — "
              "consider adding more states or using n_protected_attrs=2.")

    print("\n  CLUSTERING_FEATURES = ['Income', 'Earnings', 'PovertyRatio', "
          "'Age', 'HoursPerWeek', 'CommuteTime']")
    print(f"  PROTECTED_COL       = 'Protected_Group'  ({len(summary)} groups)")

    return df


def plot_folktables_groups(df: pd.DataFrame):
    groups = sorted(df["Protected_Group"].unique())
    group_map = {g: i for i, g in enumerate(groups)}
    colour_ids = df["Protected_Group"].map(group_map)
    cmap = "tab20" if len(groups) > 6 else "tab10"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(df["Age"], df["Income"],
                    c=colour_ids, cmap=cmap, alpha=0.3, s=5)
    axes[0].set_xlabel("Age (scaled + jitter)")
    axes[0].set_ylabel("log Income (scaled)")
    axes[0].set_title("ACS: Age vs Income by Protected Group")

    axes[1].scatter(df["HoursPerWeek"], df["CommuteTime"],
                    c=colour_ids, cmap=cmap, alpha=0.3, s=5)
    axes[1].set_xlabel("Hours/Week (scaled + jitter)")
    axes[1].set_ylabel("Commute Time (scaled + jitter)")
    axes[1].set_title("ACS: Work Hours vs Commute by Protected Group")

    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=plt.get_cmap(cmap)(i / max(len(groups) - 1, 1)),
                   markersize=8, label=g)
        for i, g in enumerate(groups)
    ]
    axes[1].legend(handles=handles, title="Protected Group", loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.savefig("folktables_groups.png", dpi=150)
    plt.show()



if __name__ == "__main__":

    N_ATTRS = 3

    # ---- Synthetic ----
    print("=" * 60)
    print(f"SYNTHETIC  ({N_ATTRS} protected attributes)")
    print("=" * 60)

    # ---- Folktables ----
    print("\n" + "=" * 60)
    print(f"FOLKTABLES ACS  ({N_ATTRS} protected attributes)")
    print("=" * 60)
#
    df_acs = load_folktables_fair_clustering(
        states=["CA", "NY"],
        survey_year="2018",
        n_protected_attrs=N_ATTRS,
        target_samples=10_000,
    )
#
    ACS_FEATURES = ["Income", "Earnings", "PovertyRatio", "Age", "HoursPerWeek", "CommuteTime"]
    plot_folktables_groups(df_acs)
#
    #df_acs.to_csv("acs_fair_clustering.csv", index=False)
    print("\nAll datasets saved.")