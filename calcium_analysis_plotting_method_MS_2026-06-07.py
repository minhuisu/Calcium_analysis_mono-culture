#!/usr/bin/env python3
"""Publishable Ex5 calcium-imaging analysis method and plotting style.

This script is intentionally path-agnostic. Provide all input/output locations
with command-line arguments; no local user-specific or cluster-specific paths
are embedded.

Expected table input for plotting/statistics:

    direct_tiff_data_all_cells.csv

The table should contain one row per cell per frame and include at least:

    Condition, Well, Position, Cell_Global_ID, Time_sec, Phase,
    Raw_Intensity_488

If Raw_Intensity_488 and Phase are present, dF/F0 is recomputed using the
current Ex5 plotting convention: F0 per cell is the mean of the lowest 20% of
baseline/pre 488 frames. If those columns are absent, an existing dF/F0 column
is used.

Example:

    python Ex5_D6_G_publishable_analysis_plotting_method_2026-06-07.py plot \
      --trace-csv <analysis-output>/direct_tiff_data_all_cells.csv \
      --output-dir <figure-output>

    python Ex5_D6_G_publishable_analysis_plotting_method_2026-06-07.py method \
      --output-json <figure-output>/accepted_ex5_method.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "matplotlib_cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


RESPONSE_T0_SEC = 120.0
F0_LOW_FRACTION = 0.20
TREATMENT_COLOR = "#7b2cbf"
DEFAULT_STEM = (
    "Ex5_DMSO_Nimodipine5uM_EGTA_KCl_PerWellMean_"
    "cellpose3nuclei_clipP0p5P99p5_d38_cellprobNeg2_flow06_"
    "withFlow09Area500RescueFlagged_nucleiSeededBodyproc_dFF0_onegraph_meanSEM"
)


ACCEPTED_EX5_METHOD = {
    "experiment": "Ex5_D6_G glioma mono-culture calcium imaging",
    "timing": {
        "frame_interval_sec": 5.0,
        "baseline_pre_frame_count": 24,
        "response_post_frame_count": 24,
    },
    "registration": {
        "channel": "561 nm",
        "reference_frame": "pre/baseline frame 0",
        "shift_estimation": "phase cross-correlation",
        "apply_same_shifts_to_channels": ["561 nm", "488 nm"],
    },
    "nuclei_segmentation_primary": {
        "projection": "registered pre/baseline median 561",
        "model": "Cellpose3 nuclei",
        "input_clip_percentiles": [0.5, 99.5],
        "diameter_px": 38,
        "cellprob_threshold": -2,
        "flow_threshold": 0.6,
        "gaussian_sigma": 0,
        "post_segmentation_561_median_filter": False,
        "global_seed_area_filter": False,
    },
    "bright_561_qc_and_rescue": {
        "bright_projection": "registered pre/baseline median 561",
        "bright_pixel_percentile": 99,
        "flag_threshold_uncovered_fraction": 0.05,
        "high_flag_threshold_uncovered_fraction": 0.10,
        "rescue_applies_only_to_flagged_fields": True,
        "rescue_model": "Cellpose3 nuclei",
        "rescue_input_clip_percentiles": [0.5, 99.5],
        "rescue_diameter_px": 38,
        "rescue_cellprob_threshold": -2,
        "rescue_flow_threshold": 0.9,
        "rescue_gaussian_sigma": 0,
        "minimum_rescue_area_px": 500,
        "minimum_bright_overlap_px": 30,
        "maximum_primary_mask_overlap_fraction": 0.25,
    },
    "roi_definition": {
        "one_roi_per_nucleus": True,
        "cell_body": "nucleus-seeded registered pre/baseline median 488",
        "body_plus_processes": "registered pre/baseline P95 488 extension assigned to the seeded body label",
        "quantification_roi": "same final body+process polygon for all frames of each cell",
    },
    "exclusions": {
        "float_away": "post-treatment 561 retention in original pre seed ROI",
        "drift": "body+process ROI overlapping pixels outside all-frame valid registered area",
        "edge": "body+process ROI touching image border",
    },
    "quantification": {
        "signal": "mean registered 488 intensity inside final edge/drift/float-away-excluded body+process ROI",
        "f0_for_current_plots": "mean of the lowest 20% baseline/pre Raw_Intensity_488 frames per cell",
        "dff0": "Raw_Intensity_488 / F0 - 1",
    },
}


PLOT_PRESETS = {
    "dmso_nim5_egta": {
        "title": "Calcium activity in glioma mono-culture",
        "data_order": [
            ("DMSO_KCl", "DMSO control", "#7a7a7a", 1.0),
            ("EGTA_KCl", "No extracellular calcium", "#317EC2", 0.70),
            ("Nim-5uM_KCl", "5$\\mu$M Nimodipine", "#c9252d", 1.0),
        ],
        "legend_rows": [
            ("DMSO control", "#7a7a7a", 0.925, 1.0),
            ("5$\\mu$M Nimodipine", "#c9252d", 0.835, 1.0),
            ("No extracellular", "#317EC2", 0.745, 0.70),
        ],
        "legend_continuation": ("calcium", 0.675),
    },
    "dmso_nim5": {
        "title": "Calcium activity in glioma mono-culture",
        "data_order": [
            ("DMSO_KCl", "DMSO control", "#7a7a7a", 1.0),
            ("Nim-5uM_KCl", "5$\\mu$M Nimodipine", "#c9252d", 1.0),
        ],
        "legend_rows": [
            ("DMSO control", "#7a7a7a", 0.925, 1.0),
            ("5$\\mu$M Nimodipine", "#c9252d", 0.835, 1.0),
        ],
        "legend_continuation": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    method = subparsers.add_parser("method", help="Write the accepted Ex5 method JSON.")
    method.add_argument("--output-json", required=True, help="Path for the method JSON file.")

    plot = subparsers.add_parser("plot", help="Plot per-well mean dF/F0 with the Ex5 publication style.")
    add_plot_args(plot)

    welch = subparsers.add_parser("welch", help="Run well-level Welch t-test on response-curve metrics.")
    add_trace_args(welch)
    welch.add_argument("--output-dir", required=True)
    welch.add_argument("--stem", default="Ex5_responseCurveStats_WelchTTest")
    welch.add_argument("--control-condition", default="DMSO_KCl")
    welch.add_argument("--treatment-condition", default="Nim-5uM_KCl")
    welch.add_argument("--comparison-label", default="treatment vs control")

    all_mode = subparsers.add_parser("all", help="Write method JSON, plot, and optional Welch test.")
    add_plot_args(all_mode)
    all_mode.add_argument("--method-json", default=None)
    all_mode.add_argument("--run-welch", action="store_true")
    all_mode.add_argument("--control-condition", default="DMSO_KCl")
    all_mode.add_argument("--treatment-condition", default="Nim-5uM_KCl")
    all_mode.add_argument("--comparison-label", default="5 uM Nimodipine vs DMSO")
    return parser.parse_args()


def add_trace_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trace-csv", required=True, help="Cell x frame table, usually direct_tiff_data_all_cells.csv.")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument(
        "--f0-low-fraction",
        type=float,
        default=F0_LOW_FRACTION,
        help="Lowest baseline fraction used to compute F0 per cell.",
    )


def add_plot_args(parser: argparse.ArgumentParser) -> None:
    add_trace_args(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--preset", choices=sorted(PLOT_PRESETS), default="dmso_nim5_egta")
    parser.add_argument("--response-t0-sec", type=float, default=RESPONSE_T0_SEC)
    parser.add_argument("--treatment-label", default="15mM KCl")
    parser.add_argument("--ylabel", default=r"Mean $\Delta F/F_0$")
    parser.add_argument("--dpi", type=int, default=300)


def write_method_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ACCEPTED_EX5_METHOD, indent=2) + "\n")


def lowest_fraction_mean(values: pd.Series, fraction: float) -> float:
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    n_low = max(1, int(np.ceil(finite.size * fraction)))
    n_low = min(n_low, finite.size)
    return float(np.partition(finite, n_low - 1)[:n_low].mean())


def read_requested_columns(csv_path: Path, requested: list[str], chunksize: int):
    header = pd.read_csv(csv_path, nrows=0)
    usecols = [col for col in requested if col in header.columns]
    missing = sorted(set(["Condition", "Well", "Position", "Cell_Global_ID", "Time_sec"]) - set(usecols))
    if missing:
        raise ValueError(f"Required columns are missing from {csv_path.name}: {missing}")
    return pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize), set(usecols)


def load_trace_table(csv_path: Path, conditions: set[str], chunksize: int, f0_low_fraction: float) -> pd.DataFrame:
    requested = [
        "Condition",
        "Well",
        "Position",
        "Cell_Global_ID",
        "Time_sec",
        "Phase",
        "Raw_Intensity_488",
        "dF/F0",
    ]
    chunks, available = read_requested_columns(csv_path, requested, chunksize)
    frames = []
    for chunk in chunks:
        chunk = chunk[chunk["Condition"].isin(conditions)].copy()
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        raise ValueError(f"No requested conditions found in {csv_path}")
    df = pd.concat(frames, ignore_index=True)

    if {"Phase", "Raw_Intensity_488"}.issubset(available):
        baseline = df[df["Phase"].astype(str).str.lower().isin({"baseline", "pre"})]
        f0 = baseline.groupby("Cell_Global_ID")["Raw_Intensity_488"].apply(
            lambda values: lowest_fraction_mean(values, f0_low_fraction)
        )
        df["F0_plot"] = df["Cell_Global_ID"].map(f0)
        df["dF/F0_plot"] = df["Raw_Intensity_488"] / df["F0_plot"] - 1.0
    elif "dF/F0" in available:
        df["dF/F0_plot"] = df["dF/F0"].astype(float)
    else:
        raise ValueError("Input table must contain Raw_Intensity_488 + Phase or an existing dF/F0 column.")

    df.loc[~np.isfinite(df["dF/F0_plot"]), "dF/F0_plot"] = np.nan
    return df


def summarize_per_well(trace_df: pd.DataFrame, label_map: dict[str, str]) -> pd.DataFrame:
    df = trace_df.copy()
    df["Plot_Label"] = df["Condition"].map(label_map)
    return (
        df.groupby(["Condition", "Plot_Label", "Well", "Time_sec"], as_index=False)
        .agg(
            Mean_dFF0=("dF/F0_plot", "mean"),
            Cell_Count=("Cell_Global_ID", "nunique"),
            Position_Count=("Position", "nunique"),
        )
        .sort_values(["Condition", "Well", "Time_sec"])
        .reset_index(drop=True)
    )


def summarize_across_wells(well_summary: pd.DataFrame) -> pd.DataFrame:
    out = (
        well_summary.groupby(["Condition", "Plot_Label", "Time_sec"], as_index=False)
        .agg(
            Mean_dFF0=("Mean_dFF0", "mean"),
            SD_WellMean_dFF0=("Mean_dFF0", "std"),
            Well_Count=("Well", "nunique"),
        )
        .sort_values(["Condition", "Time_sec"])
        .reset_index(drop=True)
    )
    out["SEM_WellMean_dFF0"] = out["SD_WellMean_dFF0"].fillna(0) / np.sqrt(out["Well_Count"].clip(lower=1))
    return out


def build_metadata(trace_df: pd.DataFrame, group_specs: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    rows = []
    for condition, label, color, alpha in group_specs:
        df = trace_df[trace_df["Condition"] == condition]
        rows.append(
            {
                "Condition": condition,
                "Legend_Label": label,
                "Color": color,
                "Line_Alpha": alpha,
                "Well_Count": int(df["Well"].nunique()),
                "Wells": ",".join(f"w{int(w)}" for w in sorted(df["Well"].dropna().unique())),
                "Position_Count": int(df[["Well", "Position"]].drop_duplicates().shape[0]),
                "Cell_Count": int(df["Cell_Global_ID"].nunique()),
                "F0_Definition": "mean of lowest 20% baseline/pre Raw_Intensity_488 frames per cell",
            }
        )
    return pd.DataFrame(rows)


def y_limits(group_summary: pd.DataFrame) -> tuple[float, float]:
    lower = group_summary["Mean_dFF0"] - group_summary["SEM_WellMean_dFF0"]
    upper = group_summary["Mean_dFF0"] + group_summary["SEM_WellMean_dFF0"]
    values = np.concatenate([lower.to_numpy(float), upper.to_numpy(float)])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -0.1, 0.1
    low = float(values.min())
    high = float(values.max())
    pad = max(0.05, 0.16 * (high - low))
    return low - pad, high + pad


def plot_ex5_style(group_summary: pd.DataFrame, preset: dict, args: argparse.Namespace, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 36,
            "axes.titlesize": 43,
            "axes.labelsize": 45,
            "xtick.labelsize": 38,
            "ytick.labelsize": 38,
            "legend.fontsize": 34,
            "axes.linewidth": 2.0,
        }
    )

    fig, ax = plt.subplots(figsize=(15.0, 10.2), dpi=args.dpi)
    ymin, ymax = y_limits(group_summary)

    for condition, label, color, alpha in preset["data_order"]:
        sub = group_summary[group_summary["Condition"] == condition].sort_values("Time_sec")
        if sub.empty:
            continue
        x = sub["Time_sec"].to_numpy(float)
        mean = sub["Mean_dFF0"].to_numpy(float)
        sem = sub["SEM_WellMean_dFF0"].to_numpy(float)
        ax.plot(x, mean, color=color, alpha=alpha, linewidth=3.5, label=label, zorder=4)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.20, linewidth=0, zorder=2)

    ax.axvline(args.response_t0_sec, color=TREATMENT_COLOR, linestyle="--", linewidth=2.8, zorder=5)
    ax.text(
        args.response_t0_sec + 3,
        ymax - 0.06 * (ymax - ymin),
        args.treatment_label,
        color=TREATMENT_COLOR,
        fontsize=38,
        ha="left",
        va="top",
    )

    ax.set_title(preset["title"], pad=20)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(args.ylabel)
    ax.set_xlim(-2, 240)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([0, 60, 120, 180, 240])
    ax.grid(False)
    ax.tick_params(length=8, width=2.0)

    legend_x = 0.055
    text_x = 0.135
    for label, color, y, alpha in preset["legend_rows"]:
        ax.plot(
            [legend_x, legend_x + 0.055],
            [y, y],
            transform=ax.transAxes,
            color=color,
            alpha=alpha,
            linewidth=4.2,
            solid_capstyle="round",
            clip_on=False,
        )
        ax.text(
            text_x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=34,
            color="#222222",
            ha="left",
            va="center",
        )
    continuation = preset.get("legend_continuation")
    if continuation is not None:
        label, y = continuation
        ax.text(text_x, y, label, transform=ax.transAxes, fontsize=34, color="#222222", ha="left", va="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)


def response_metrics(well_summary: pd.DataFrame, response_t0_sec: float = RESPONSE_T0_SEC) -> pd.DataFrame:
    rows = []
    for (condition, label, well), sub in well_summary.groupby(["Condition", "Plot_Label", "Well"], sort=False):
        sub = sub.sort_values("Time_sec")
        response = sub[sub["Time_sec"] >= response_t0_sec]
        early = sub[(sub["Time_sec"] >= response_t0_sec) & (sub["Time_sec"] <= response_t0_sec + 60.0)]
        if response.empty:
            continue
        x = response["Time_sec"].to_numpy(dtype=float)
        y = response["Mean_dFF0"].to_numpy(dtype=float)
        duration = x[-1] - x[0]
        rows.append(
            {
                "Condition": condition,
                "Group_Label": label,
                "Well": int(well),
                "Response_Time_Start_sec": float(x[0]),
                "Response_Time_End_sec": float(x[-1]),
                "Response_Frame_Count": int(len(response)),
                "Response_Mean_dFF0": float(np.nanmean(y)),
                "Response_AUC_dFF0_sec": float(np.trapz(y, x)),
                "Response_AUC_Normalized_dFF0": float(np.trapz(y, x) / duration) if duration > 0 else np.nan,
                "Response_Peak_dFF0": float(np.nanmax(y)),
                "Response_t0_dFF0": float(response.loc[response["Time_sec"].idxmin(), "Mean_dFF0"]),
                "Response_Early_120_180_Mean_dFF0": float(np.nanmean(early["Mean_dFF0"])) if not early.empty else np.nan,
                "Mean_Cell_Count_Per_Timepoint": float(response["Cell_Count"].mean()),
                "Position_Count": int(response["Position_Count"].max()),
            }
        )
    return pd.DataFrame(rows)


def welch_df(control: np.ndarray, treatment: np.ndarray) -> float:
    control_var = float(np.var(control, ddof=1))
    treatment_var = float(np.var(treatment, ddof=1))
    control_n = len(control)
    treatment_n = len(treatment)
    se2 = control_var / control_n + treatment_var / treatment_n
    return float(
        (se2**2)
        / ((control_var / control_n) ** 2 / (control_n - 1) + (treatment_var / treatment_n) ** 2 / (treatment_n - 1))
    )


def welch_tests(metrics_df: pd.DataFrame, control_condition: str, treatment_condition: str, comparison_label: str):
    metrics = [
        "Response_AUC_Normalized_dFF0",
        "Response_AUC_dFF0_sec",
        "Response_Mean_dFF0",
        "Response_Peak_dFF0",
        "Response_t0_dFF0",
        "Response_Early_120_180_Mean_dFF0",
    ]
    rows = []
    for metric in metrics:
        control = metrics_df.loc[metrics_df["Condition"] == control_condition, metric].dropna().to_numpy(dtype=float)
        treatment = metrics_df.loc[metrics_df["Condition"] == treatment_condition, metric].dropna().to_numpy(dtype=float)
        test = stats.ttest_ind(treatment, control, equal_var=False, alternative="two-sided")
        df = welch_df(control, treatment)
        rows.append(
            {
                "Metric": metric,
                "Comparison": comparison_label,
                "Control_Condition": control_condition,
                "Treatment_Condition": treatment_condition,
                "Control_N": int(len(control)),
                "Treatment_N": int(len(treatment)),
                "Control_Mean": float(np.mean(control)),
                "Treatment_Mean": float(np.mean(treatment)),
                "Mean_Difference_Treatment_minus_Control": float(np.mean(treatment) - np.mean(control)),
                "Control_SD": float(np.std(control, ddof=1)),
                "Treatment_SD": float(np.std(treatment, ddof=1)),
                "Welch_t": float(test.statistic),
                "Welch_df": df,
                "Welch_p_two_sided": float(test.pvalue),
            }
        )
    return pd.DataFrame(rows)


def run_plot(args: argparse.Namespace):
    preset = PLOT_PRESETS[args.preset]
    group_specs = preset["data_order"]
    conditions = {condition for condition, _label, _color, _alpha in group_specs}
    label_map = {condition: label for condition, label, _color, _alpha in group_specs}
    trace_df = load_trace_table(Path(args.trace_csv), conditions, args.chunksize, args.f0_low_fraction)
    well_summary = summarize_per_well(trace_df, label_map)
    group_summary = summarize_across_wells(well_summary)
    metadata = build_metadata(trace_df, group_specs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{args.stem}.png"
    plot_ex5_style(group_summary, preset, args, png_path)
    well_summary.to_csv(output_dir / f"{args.stem}_per_well_mean.csv", index=False)
    group_summary.to_csv(output_dir / f"{args.stem}_across_well_mean_sem.csv", index=False)
    metadata.to_csv(output_dir / f"{args.stem}_metadata.csv", index=False)
    return well_summary, group_summary, metadata, png_path


def run_welch(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    conditions = {args.control_condition, args.treatment_condition}
    trace_df = load_trace_table(Path(args.trace_csv), conditions, args.chunksize, args.f0_low_fraction)
    label_map = {
        args.control_condition: args.control_condition,
        args.treatment_condition: args.treatment_condition,
    }
    well_summary = summarize_per_well(trace_df, label_map)
    metrics = response_metrics(well_summary, RESPONSE_T0_SEC)
    welch = welch_tests(metrics, args.control_condition, args.treatment_condition, args.comparison_label)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / f"{args.stem}_per_well_response_metrics.csv", index=False)
    welch.to_csv(output_dir / f"{args.stem}_welch_ttest.csv", index=False)
    return metrics, welch


def main() -> None:
    args = parse_args()
    if args.mode == "method":
        write_method_json(Path(args.output_json))
        print(Path(args.output_json))
    elif args.mode == "plot":
        *_unused, png_path = run_plot(args)
        print(png_path)
    elif args.mode == "welch":
        _metrics, welch = run_welch(args)
        print(welch.to_string(index=False))
    elif args.mode == "all":
        if args.method_json:
            write_method_json(Path(args.method_json))
        _well, _group, _metadata, png_path = run_plot(args)
        print(png_path)
        if args.run_welch:
            _metrics, welch = run_welch(args)
            print(welch.to_string(index=False))


if __name__ == "__main__":
    main()
