import numpy as np
from matplotlib import pyplot as plt

from algorithms.main_bercea_fair_clustering import fair_clustering as bercea_fc
from algorithms.main_bera_fair_clustering import fair_clustering as bera_fc

from runner import run_trials, build_bera_result, build_bercea_result
import csv_loader

# taken form data analysis
FEATURE_CONFIGS = [
    {"name": "Sex", "group_id_features": ["SEX"], "L": 2, "DI": 0.013},
    {"name": "Race (Binary)", "group_id_features": ["RACE_BINARY"], "L": 2, "DI": 0.288},
    {"name": "Age", "group_id_features": ["AGE_BIN"], "L": 4, "DI": 0.042}
    #{"name": "Income", "group_id_features": ["INC_BIN"], "L": 4, "DI": 0.094},
    #{"name": "Race (6-bin)", "group_id_features": ["RACE_6"], "L": 6, "DI": 0.343},
    #{"name": "Race (9-cat)", "group_id_features": ["RAC1P"], "L": 9, "DI": 0.413},
    #{"name": "Inc × Age", "group_id_features": ["INC_BIN", "AGE_BIN"], "L": 16, "DI": None},
]

_ALG_PALETTE = {
    "bera": "#4C72B0",
    "bercea": "#DD8452",
    "boehm": "#55A868",
}

_PHASE_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

def compute_mean_di(
        df_processed,
        group_col="GROUP_ID",
        lat_col="Latitude",
        lon_col="Longitude",
        n_lat_bins=20,
        n_lon_bins=30,
        min_cell_count=50,
) -> float:
    """
    Mean Dissimilarity Index across all groups.
      20 × 30 lat/lon grid, cells with < 50 points excluded,
    """
    lat = df_processed[lat_col].values.astype(np.float64)
    lon = df_processed[lon_col].values.astype(np.float64)
    groups = df_processed[group_col].values
    unique_groups = np.unique(groups)
    N = len(df_processed)

    lat_edges = np.linspace(lat.min() - 1e-9, lat.max() + 1e-9, n_lat_bins + 1)
    lon_edges = np.linspace(lon.min() - 1e-9, lon.max() + 1e-9, n_lon_bins + 1)
    lat_bin = np.digitize(lat, lat_edges) - 1
    lon_bin = np.digitize(lon, lon_edges) - 1
    cell_id = lat_bin * n_lon_bins + lon_bin

    unique_cells = np.unique(cell_id)

    # Pre-compute per-cell totals and group counts (only valid cells)
    valid_cells = []
    cell_total: dict[int, int] = {}
    cell_group: dict[tuple[int, object], int] = {}
    for cell in unique_cells:
        mask = cell_id == cell
        n_i = int(mask.sum())
        if n_i < min_cell_count:
            continue
        valid_cells.append(cell)
        cell_total[cell] = n_i
        for g in unique_groups:
            cell_group[(cell, g)] = int((mask & (groups == g)).sum())

    di_per_group = []
    for g in unique_groups:
        N_g = int((groups == g).sum())
        if N_g == 0 or N_g == N:
            continue
        di_g = 0.0
        for cell in valid_cells:
            n_i = cell_total[cell]
            n_ig = cell_group.get((cell, g), 0)
            di_g += abs(n_ig / N_g - (n_i - n_ig) / (N - N_g))
        di_g *= 0.5
        di_per_group.append(di_g)

    return float(np.mean(di_per_group)) if di_per_group else 0.0


def count_violations(result, alpha: float) -> int:
    """
    Count (cluster, group) pairs that violate proportional fairness bounds
    on the representative run's fair assignment.
    """
    codes = result.group_codes
    w = result.weights
    labels = result.labels
    n_groups = len(result.group_names)
    k = len(result.centers)

    total_w = w.sum()
    freqs = np.array([w[codes == h].sum() / total_w for h in range(n_groups)])
    lb = np.maximum(0.0, freqs - alpha)
    ub = np.minimum(1.0, freqs + alpha)

    violations = 0
    for j in range(k):
        mask_j = labels == j
        total_j = w[mask_j].sum()
        if total_j < 1e-12:
            continue
        for h in range(n_groups):
            frac = w[mask_j & (codes == h)].sum() / total_j
            if frac < lb[h] - 1e-4 or frac > ub[h] + 1e-4:
                violations += 1
    return violations



def plot_pof_vs_di(rows: list[dict], alpha: float) -> None:
    """
    Scatter: PoF (y) vs DI (x), one series per algorithm.
    Each feature annotated with name and L.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ALG_STYLE = {
        "Bera": {"color": "#4C72B0", "marker": "o", "zorder": 5},
        "Bercea": {"color": "#DD8452", "marker": "s", "zorder": 5},
    }

    # Slight horizontal jitter so same-DI points don't overlap
    DI_JITTER = {"Bera": -0.003, "Bercea": 0.003}

    for alg_name, style in ALG_STYLE.items():
        alg_rows = [r for r in rows if r["algorithm"] == alg_name and r["DI"] is not None]
        dis = [r["DI"] + DI_JITTER[alg_name] for r in alg_rows]
        pofs = [r["pof_mean"] for r in alg_rows]
        stds = [r["pof_std"] for r in alg_rows]
        ax.errorbar(
            dis, pofs, yerr=stds,
            fmt=style["marker"], color=style["color"],
            label=alg_name, capsize=4, markersize=8, linewidth=1.2,
            zorder=style["zorder"],
        )

    # Annotate once per feature at the midpoint of the two algorithms
    features_seen: dict[str, dict] = {}
    for r in rows:
        if r["DI"] is None:
            continue
        fname = r["feature"]
        if fname not in features_seen:
            features_seen[fname] = {"DI": r["DI"], "L": r["L"], "pofs": []}
        features_seen[fname]["pofs"].append(r["pof_mean"])

    # Pre-defined offsets (dx, dy in points) to reduce annotation overlap
    annotation_offsets = [
        (14, 14), (14, -18), (-90, 14), (14, 18),
        (-90, -18), (14, -24), (-90, 20),
    ]

    for idx, (fname, info) in enumerate(features_seen.items()):
        mean_pof = float(np.mean(info["pofs"]))
        dx, dy = annotation_offsets[idx % len(annotation_offsets)]
        ax.annotate(
            f"{fname}\n(L={info['L']})",
            xy=(info["DI"], mean_pof),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
            ha="left" if dx > 0 else "right",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
               label="PoF = 1 (fairness is free)")
    ax.set_xlabel("Dissimilarity Index (DI)", fontsize=11)
    ax.set_ylabel("Price of Fairness (PoF, mean ± std)", fontsize=11)
    ax.set_title(
        f"PoF vs Spatial Segregation by Feature Choice  "
        f"(n=20 k, k=10, α={alpha})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig("./pof_vs_di_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved pof_vs_di_scatter.png")

    # --- Pearson correlation analysis ---
    print(f"\n{'=' * 60}")
    print("  CORRELATION ANALYSIS")
    print(f"{'=' * 60}")
    for alg_name in ["Bera", "Bercea"]:
        alg_rows = [r for r in rows if r["algorithm"] == alg_name and r["DI"] is not None]
        if len(alg_rows) < 3:
            continue
        dis = np.array([r["DI"] for r in alg_rows])
        pofs = np.array([r["pof_mean"] for r in alg_rows])
        ls = np.array([float(r["L"]) for r in alg_rows])

        r_di = float(np.corrcoef(dis, pofs)[0, 1])
        r_l = float(np.corrcoef(ls, pofs)[0, 1])

        print(f"\n  {alg_name}:")
        print(f"    Pearson r(PoF, DI) = {r_di:+.4f}")
        print(f"    Pearson r(PoF, L)  = {r_l:+.4f}")
        if abs(r_di) > abs(r_l):
            print("    → PoF correlates more with DI than L "
                  "→ spatial segregation drives cost")
        else:
            print("    → PoF correlates more with L than DI "
                  "→ group count drives cost")
    print(f"{'=' * 60}")


def plot_runtime_by_feature(rows: list[dict]) -> None:
    """
    Grouped bar chart: total runtime per feature, one bar per algorithm.
    Also a stacked version showing phase breakdown per algorithm.
    """
    features = list(dict.fromkeys(r["feature"] for r in rows))
    algorithms = ["Bera", "Bercea"]

    # --- Figure 1: Total runtime grouped bars ---
    fig, ax = plt.subplots(figsize=(max(8, len(features) * 1.6), 5))
    bar_width = 0.35
    x = np.arange(len(features))

    BERA_TOTAL = "Total Time"
    BERCEA_TOTAL = "Total Time"
    total_keys = {"Bera": BERA_TOTAL, "Bercea": BERCEA_TOTAL}

    for i, alg in enumerate(algorithms):
        means, stds = [], []
        for fname in features:
            r = next((r for r in rows if r["feature"] == fname and r["algorithm"] == alg), None)
            if r:
                timings = r["all_timings"]
                tk = total_keys[alg]
                vals = [t.get(tk, 0.0) for t in timings]
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0))
            else:
                means.append(0.0)
                stds.append(0.0)

        offset = (i - 0.5) * bar_width
        color = _ALG_PALETTE.get(alg.lower(), "gray")
        bars = ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
                      label=alg, color=color, edgecolor="black", linewidth=0.4, zorder=3)

        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.1,
                    f"{m:.1f}s", ha="center", va="bottom", fontsize=7, color="dimgray")

    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Total wall-clock time (s)", fontsize=11)
    ax.set_title("Runtime by Feature Choice (n=20 k, k=10)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig("./runtime_by_feature.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved runtime_by_feature.png")

    # --- Figure 2: Stacked phase breakdown per algorithm ---
    BERA_PHASES = ["Data Preparation", "Vanilla K-Median", "Solve Initial LP",
                   "Iterative Rounding", "Cost Calculation"]
    BERCEA_PHASES = ["Data Preparation", "Vanilla K-Median", "Solve Initial LP",
                     "MCF Rounding", "Cost Calculation"]
    alg_phases = {"Bera": BERA_PHASES, "Bercea": BERCEA_PHASES}

    fig2, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)

    for ax_idx, alg in enumerate(algorithms):
        ax2 = axes[ax_idx]
        phases = alg_phases[alg]
        bottoms = np.zeros(len(features))

        for p_idx, phase in enumerate(phases):
            vals_per_feature = []
            for fname in features:
                r = next((r for r in rows if r["feature"] == fname and r["algorithm"] == alg), None)
                if r:
                    timings = r["all_timings"]
                    v = float(np.mean([t.get(phase, 0.0) for t in timings]))
                else:
                    v = 0.0
                vals_per_feature.append(v)
            color = _PHASE_PALETTE[p_idx % len(_PHASE_PALETTE)]
            ax2.bar(x, vals_per_feature, bar_width * 1.6, bottom=bottoms,
                    label=phase, color=color, edgecolor="black", linewidth=0.3, zorder=3)
            bottoms += np.array(vals_per_feature)

        ax2.set_xticks(x)
        ax2.set_xticklabels(features, fontsize=9, rotation=25, ha="right")
        ax2.set_title(f"{alg} — Phase Breakdown", fontsize=12)
        ax2.set_ylabel("Time (s)" if ax_idx == 0 else "", fontsize=11)
        ax2.legend(fontsize=7, loc="upper left")
        ax2.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    fig2.suptitle("Per-Phase Runtime by Feature Choice (n=20 k, k=10)", fontsize=13, y=1.01)
    fig2.tight_layout()
    fig2.savefig("./runtime_by_feature_phases.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved runtime_by_feature_phases.png")


def print_feature_table(rows: list[dict]) -> None:
    """Print and save a summary table of results per feature, including runtime."""

    features = list(dict.fromkeys(r["feature"] for r in rows))  # insertion order

    BERA_TOTAL = "Total Time"
    BERCEA_TOTAL = "Total Time"
    total_keys = {"Bera": BERA_TOTAL, "Bercea": BERCEA_TOTAL}

    header = (
        f"{'Feature':<18s} {'L':>3s} {'DI':>6s}  "
        f"{'Bera PoF':>16s}  {'Bercea PoF':>16s}  "
        f"{'Bera Viol.':>10s}  {'Bercea Viol.':>10s}  "
        f"{'Bera Time':>12s}  {'Bercea Time':>12s}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    csv_lines = [
        "Feature,L,DI,"
        "Bera_PoF_mean,Bera_PoF_std,Bercea_PoF_mean,Bercea_PoF_std,"
        "Bera_FairCost_mean,Bercea_FairCost_mean,"
        "Bera_UnfairCost_mean,Bercea_UnfairCost_mean,"
        "Bera_Violations,Bercea_Violations,"
        "Bera_Time_mean,Bera_Time_std,Bercea_Time_mean,Bercea_Time_std"
    ]

    for fname in features:
        f_rows = [r for r in rows if r["feature"] == fname]
        bera_r = next((r for r in f_rows if r["algorithm"] == "Bera"), None)
        bercea_r = next((r for r in f_rows if r["algorithm"] == "Bercea"), None)

        L = f_rows[0]["L"]
        DI = f_rows[0]["DI"]
        di_str = f"{DI:.3f}" if DI is not None else "  N/A"

        def _fmt_pof(r):
            return f"{r['pof_mean']:.4f}±{r['pof_std']:.4f}" if r else "—"

        def _fmt_viol(r):
            return f"{r['violations']}" if r else "—"

        def _fmt_time(r, alg):
            if not r:
                return "—"
            tk = total_keys[alg]
            vals = [t.get(tk, 0.0) for t in r["all_timings"]]
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            return f"{m:.1f}±{s:.1f}s"

        def _time_vals(r, alg):
            if not r:
                return 0.0, 0.0
            tk = total_keys[alg]
            vals = [t.get(tk, 0.0) for t in r["all_timings"]]
            return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

        print(
            f"{fname:<18s} {L:>3d} {di_str:>6s}  "
            f"{_fmt_pof(bera_r):>16s}  {_fmt_pof(bercea_r):>16s}  "
            f"{_fmt_viol(bera_r):>10s}  {_fmt_viol(bercea_r):>10s}  "
            f"{_fmt_time(bera_r, 'Bera'):>12s}  {_fmt_time(bercea_r, 'Bercea'):>12s}"
        )

        b_pm = f"{bera_r['pof_mean']:.6f}" if bera_r else ""
        b_ps = f"{bera_r['pof_std']:.6f}" if bera_r else ""
        c_pm = f"{bercea_r['pof_mean']:.6f}" if bercea_r else ""
        c_ps = f"{bercea_r['pof_std']:.6f}" if bercea_r else ""
        b_fc = f"{bera_r['fair_cost_mean']:.2f}" if bera_r else ""
        c_fc = f"{bercea_r['fair_cost_mean']:.2f}" if bercea_r else ""
        b_uc = f"{bera_r['unfair_cost_mean']:.2f}" if bera_r else ""
        c_uc = f"{bercea_r['unfair_cost_mean']:.2f}" if bercea_r else ""
        b_v = str(bera_r["violations"]) if bera_r else ""
        c_v = str(bercea_r["violations"]) if bercea_r else ""
        di_csv = f"{DI:.4f}" if DI is not None else ""
        b_tm, b_ts = _time_vals(bera_r, "Bera")
        c_tm, c_ts = _time_vals(bercea_r, "Bercea")

        csv_lines.append(
            f"{fname},{L},{di_csv},"
            f"{b_pm},{b_ps},{c_pm},{c_ps},"
            f"{b_fc},{c_fc},{b_uc},{c_uc},{b_v},{c_v},"
            f"{b_tm:.2f},{b_ts:.2f},{c_tm:.2f},{c_ts:.2f}"
        )

    print(sep)

    csv_path = "./evaluation3_results.csv"
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_lines))
    print(f"\n  Results saved to {csv_path}")



if __name__ == "__main__":
    N_SIZE = 10_000
    FEATURE_COLS = ["Lat_Scaled", "Lon_Scaled"]
    PROTECTED_COL = "GROUP_ID"
    K = 10
    N_RUNS = 5
    ALPHA = 0.05


    # ------------------------------------------------------------------
    # Step 2: Run Bera & Bercea for each feature configuration
    # ------------------------------------------------------------------
    all_rows: list[dict] = []

    for cfg in FEATURE_CONFIGS:
        print(f"\n{'#' * 60}")
        print(f"  FEATURE: {cfg['name']}  (L={cfg['L']}, DI={cfg['DI']:.3f})")
        print(f"{'#' * 60}")

        # ---- Bera (Iterative Rounding) ----
        print(f"\n  Running Bera [1] ...")
        bera_result, bera_summary = run_trials(
            max_rows=N_SIZE,
            algorithm_fn=bera_fc,
            result_builder=build_bera_result,
            group_id_features=cfg["group_id_features"],
            n_runs=N_RUNS,
            feature_cols=FEATURE_COLS,
            protected_group_col=PROTECTED_COL,
            k_centers=K,
            alpha=ALPHA,
            weight_col=None,
        )
        bera_viol = count_violations(bera_result, ALPHA)

        all_rows.append({
            "feature": cfg["name"],
            "L": cfg["L"],
            "DI": cfg["DI"],
            "algorithm": "Bera",
            "pof_mean": bera_summary["All results PoF (mean)"],
            "pof_std": bera_summary["All results PoF (std)"],
            "fair_cost_mean": bera_summary["All results Fair Cost (mean)"],
            "unfair_cost_mean": bera_summary["All results Unfair Cost (mean)"],
            "violations": bera_viol,
            "avg_timing": bera_summary["Avg Timing"],
            "all_timings": bera_summary["_timings"],
        })

        # ---- Bercea (MCF Rounding) ----
        print(f"\n  Running Bercea [2] ...")
        bercea_result, bercea_summary = run_trials(
            max_rows=N_SIZE,
            algorithm_fn=bercea_fc,
            result_builder=build_bercea_result,
            group_id_features=cfg["group_id_features"],
            n_runs=N_RUNS,
            feature_cols=FEATURE_COLS,
            protected_group_col=PROTECTED_COL,
            k_cluster=K,
            alpha=ALPHA,
            weight_col=None,
        )
        bercea_viol = count_violations(bercea_result, ALPHA)

        all_rows.append({
            "feature": cfg["name"],
            "L": cfg["L"],
            "DI": cfg["DI"],
            "algorithm": "Bercea",
            "pof_mean": bercea_summary["All results PoF (mean)"],
            "pof_std": bercea_summary["All results PoF (std)"],
            "fair_cost_mean": bercea_summary["All results Fair Cost (mean)"],
            "unfair_cost_mean": bercea_summary["All results Unfair Cost (mean)"],
            "violations": bercea_viol,
            "avg_timing": bercea_summary["Avg Timing"],
            "all_timings": bercea_summary["_timings"],
        })

    # ------------------------------------------------------------------
    # Step 3: Outputs
    # ------------------------------------------------------------------
    plot_pof_vs_di(all_rows, ALPHA)
    plot_runtime_by_feature(all_rows)
    print_feature_table(all_rows)
