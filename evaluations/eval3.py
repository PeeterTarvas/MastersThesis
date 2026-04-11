import numpy as np
from matplotlib import pyplot as plt

from algorithms.main_bercea_fair_clustering import fair_clustering as bercea_fc
from algorithms.main_bera_fair_clustering import fair_clustering as bera_fc

from runner import run_trials, build_bera_result, build_bercea_result
import csv_loader

FEATURE_CONFIGS = [
    {"name": "Sex", "group_id_features": ["SEX"], "L": 2, "DI": 0.013},
    {"name": "Race (Binary)", "group_id_features": ["RACE_BINARY"], "L": 2, "DI": 0.288},
    {"name": "Age", "group_id_features": ["AGE_BIN"], "L": 4, "DI": 0.042},
    {"name": "Income", "group_id_features": ["INC_BIN"], "L": 4, "DI": 0.094},
    {"name": "Race (6-bin)", "group_id_features": ["RACE_6"], "L": 6, "DI": 0.343},
    {"name": "Race (9-cat)", "group_id_features": ["RAC1P"], "L": 9, "DI": 0.413},
    {"name": "Inc × Age", "group_id_features": ["INC_BIN", "AGE_BIN"], "L": 16, "DI": None},
]

_ALG_PALETTE = {
    "bera": "#4C72B0",
    "bercea": "#DD8452",
    "boehm": "#55A868",
}

if __name__ == "__main__":
    N_SIZE = 20_000
    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"
    K = 10
    N_RUNS = 5
    ALPHA = 0.05

    all_rows: list[dict] = []
