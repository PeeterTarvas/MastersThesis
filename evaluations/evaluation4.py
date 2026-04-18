import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

from algorithms.main_bercea_fair_clustering import fair_clustering as bercea_fc
from algorithms.main_bera_fair_clustering import fair_clustering as bera_fc

from runner import run_trials, build_bera_result, build_bercea_result

ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.2]

FEATURE_CONFIGS = [
    {"name": "RACE_6", "group_id_features": ["RACE_6"], "L": 6, "DI": 0.343},
    #{"name": "INC_BIN", "group_id_features": ["INC_BIN"], "L": 4, "DI": 0.094},
]

_ALG_COLORS = {"Bera": "#4C72B0", "Bercea": "#DD8452"}
_FEAT_STYLES = {"RACE_6": "-", "INC_BIN": "--"}
_FEAT_MARKERS = {"RACE_6": "o", "INC_BIN": "s"}


def _avg_total_time(timings_list: list[dict]) -> tuple[float, float]:
    vals = [t.get("Total Time", 0.0) for t in timings_list]
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return m, s

def plot_pof_vs_alpha(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for alg in ["Bera", "Bercea"]:
        for feat_cfg in FEATURE_CONFIGS:
            fname = feat_cfg["name"]
            subset = [row for row in rows
                      if row["algorithm"] == alg and row["feature"] == fname]
            subset.sort(key=lambda r: r["alpha"])
            alphas = [row["alpha"] for row in subset]
            means  = [row["pof_mean"] for row in subset]
            stds   = [row["pof_std"] for row in subset]
            color = _ALG_COLORS[alg]
            ls = _FEAT_STYLES[fname]
            marker = _FEAT_MARKERS[fname]
            ax.errorbar(alphas, means, yerr=stds, label=f"{alg} — {fname}",
                        color=color, linestyle=ls, marker=marker,
                        capsize=4, markersize=6, linewidth=1.6)
    ax.axhline(1.0, color="red", linestyle=":", linewidth=0.8, alpha=0.5,
               label="PoF = 1")
    ax.set_xlabel("α (fairness slack)", fontsize=11)
    ax.set_ylabel("Price of Fairness (PoF)", fontsize=11)
    ax.set_title("Effect of α on PoF  (n=10 k, k=10)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig("evaluation4_pof_vs_alpha.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved evaluation4_pof_vs_alpha.png")

def plot_runtime_vs_alpha(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for alg in ["Bera", "Bercea"]:
        for feat_cfg in FEATURE_CONFIGS:
            fname = feat_cfg["name"]
            subset = [r for r in rows
                      if r["algorithm"] == alg and r["feature"] == fname]
            subset.sort(key=lambda r: r["alpha"])

            alphas = [r["alpha"] for r in subset]
            means = [r["time_mean"] for r in subset]
            stds = [r["time_std"] for r in subset]

            color = _ALG_COLORS[alg]
            ls = _FEAT_STYLES[fname]
            marker = _FEAT_MARKERS[fname]

            ax.errorbar(alphas, means, yerr=stds, label=f"{alg} — {fname}",
                        color=color, linestyle=ls, marker=marker,
                        capsize=4, markersize=6, linewidth=1.6)

    ax.set_xlabel("α (fairness slack)", fontsize=11)
    ax.set_ylabel("Total Runtime", fontsize=11)
    ax.set_title("Runtime vs α  (n=10 k, k=10)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig("evaluation4_runtime_vs_alpha.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved evaluation4_runtime_vs_alpha.png")


def print_alpha_table(rows: list[dict]) -> None:
    header = (
        f"{'Feature'} {'α':>5s}  "
        f"{'Bera PoF'}  {'Bercea PoF'}  "
        f"{'Bera Viol'}  {'Bercea Viol'}  "
        f"{'Bera Time'}  {'Bercea Time'}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    csv_lines = [
        "Feature,L,DI,Alpha,"
        "Bera_PoF_mean,Bera_PoF_std,Bercea_PoF_mean,Bercea_PoF_std,"
        "Bera_FairCost_mean,Bercea_FairCost_mean,"
        "Bera_UnfairCost_mean,Bercea_UnfairCost_mean,"
        "Bera_Violations,Bercea_Violations,"
        "Bera_Time_mean,Bera_Time_std,Bercea_Time_mean,Bercea_Time_std"
    ]

    for feat_cfg in FEATURE_CONFIGS:
        fname = feat_cfg["name"]
        L = feat_cfg["L"]
        DI = feat_cfg["DI"]
        for alpha in ALPHAS:
            bera_r = next((r for r in rows
                           if r["feature"] == fname and r["algorithm"] == "Bera"
                           and r["alpha"] == alpha), None)
            bercea_r = next((r for r in rows
                             if r["feature"] == fname and r["algorithm"] == "Bercea"
                             and r["alpha"] == alpha), None)

            def _fmt_pof(r):
                if not r:
                    return "—"
                return f"{r['pof_mean']}±{r['pof_std']}"

            def _fmt_time(r):
                if not r:
                    return "—"
                return f"{r['time_mean']}±{r['time_std']}s"

            print(
                f"{fname} {alpha}  "
                f"{_fmt_pof(bera_r)}  {_fmt_pof(bercea_r)}  "
                f"{_fmt_time(bera_r)}  {_fmt_time(bercea_r)}"
            )

            b_pm = f"{bera_r['pof_mean']}"
            b_ps = f"{bera_r['pof_std']}"
            c_pm = f"{bercea_r['pof_mean']}"
            c_ps = f"{bercea_r['pof_std']}"
            b_fc = f"{bera_r['fair_cost_mean']}"
            c_fc = f"{bercea_r['fair_cost_mean']}"
            b_uc = f"{bera_r['unfair_cost_mean']}"
            c_uc = f"{bercea_r['unfair_cost_mean']}"
            b_tm = f"{bera_r['time_mean']}"
            b_ts = f"{bera_r['time_std']}"
            c_tm = f"{bercea_r['time_mean']}"
            c_ts = f"{bercea_r['time_std']}"

            csv_lines.append(
                f"{fname},{L},{DI:.4f},{alpha},"
                f"{b_pm},{b_ps},{c_pm},{c_ps},"
                f"{b_fc},{c_fc},{b_uc},{c_uc},"
                f"{b_tm},{b_ts},{c_tm},{c_ts}"
            )

        print()

    print(sep)
    csv_path = "./evaluation4_results.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"\n  Results saved to {csv_path}")


if __name__ == "__main__":
    N_SIZE = 20_000
    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"
    K = 10
    N_RUNS = 5

    all_rows: list[dict] = []

    for feature_cfg in FEATURE_CONFIGS:
        feature_name = feature_cfg["name"]
        for alpha in ALPHAS:
            print(f"  FEATURE: {feature_name} (L={feature_cfg['L']})   α = {alpha}")
            bera_result, bera_summary = run_trials(
                max_rows=N_SIZE,
                algorithm_fn=bera_fc,
                result_builder=build_bera_result,
                group_id_features=feature_cfg["group_id_features"],
                n_runs=N_RUNS,
                feature_cols=FEATURE_COLS,
                protected_group_col=PROTECTED_COL,
                k_centers=K,
                alpha=alpha,
                weight_col=None,
            )
            bera_tm, bera_ts = _avg_total_time(bera_summary["_timings"])
            all_rows.append({
                "feature": feature_name,
                "L": feature_cfg["L"],
                "DI": feature_cfg["DI"],
                "alpha": alpha,
                "algorithm": "Bera",
                "pof_mean": bera_summary["All results PoF (mean)"],
                "pof_std": bera_summary["All results PoF (std)"],
                "fair_cost_mean": bera_summary["All results Fair Cost (mean)"],
                "unfair_cost_mean": bera_summary["All results Unfair Cost (mean)"],
                "time_mean": bera_tm,
                "time_std": bera_ts,
                "all_timings": bera_summary["_timings"],
            })

            print(f"\n  Running Bercea [2] ...")
            bercea_result, bercea_summary = run_trials(
                max_rows=N_SIZE,
                algorithm_fn=bercea_fc,
                result_builder=build_bercea_result,
                group_id_features=feature_cfg["group_id_features"],
                n_runs=N_RUNS,
                feature_cols=FEATURE_COLS,
                protected_group_col=PROTECTED_COL,
                k_cluster=K,
                alpha=alpha,
                weight_col=None,
            )
            bercea_tm, bercea_ts = _avg_total_time(bercea_summary["_timings"])
            all_rows.append({
                "feature":          feature_name,
                "L":                feature_cfg["L"],
                "DI":               feature_cfg["DI"],
                "alpha":            alpha,
                "algorithm":        "Bercea",
                "pof_mean":         bercea_summary["All results PoF (mean)"],
                "pof_std":          bercea_summary["All results PoF (std)"],
                "fair_cost_mean":   bercea_summary["All results Fair Cost (mean)"],
                "unfair_cost_mean": bercea_summary["All results Unfair Cost (mean)"],
                "time_mean":        bercea_tm,
                "time_std":         bercea_ts,
                "all_timings":      bercea_summary["_timings"],
            })
    plot_pof_vs_alpha(all_rows)
    plot_runtime_vs_alpha(all_rows)
    print_alpha_table(all_rows)