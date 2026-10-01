from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    FIG,
    KAMP_FEATURES,
    KAMP_PATH,
    OUT,
    SOURCE_PATH,
    aggregate_source,
    ensure_output_dirs,
    load_kamp,
    load_source_raw,
    read_config,
    sha256,
    source_consistency,
    write_csv,
    write_json,
)

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 130})


def main() -> None:
    config = read_config()
    ensure_output_dirs()
    source_raw = load_source_raw(config)
    source = aggregate_source(source_raw)
    consistency = source_consistency(source_raw)
    kamp, quality = load_kamp(config)
    zero_valid_sensor_ids = source.loc[
        source["current_valid_rows"].eq(0) | source["force_valid_rows"].eq(0),
        "sample_id",
    ].astype(int).tolist()
    stage0_status = (
        "stage0_fail_source_sensor_valid_count_zero"
        if zero_valid_sensor_ids
        else "stage0_pass"
    )

    category_counts = source.groupby("category").size().to_dict()
    communication = source_raw.loc[
        source_raw["Comments"].fillna("").astype(str).str.contains("Communication error", case=False),
        ["Sample ID", "Force (N)", "Current (A)", "Comments"],
    ]
    thickness_inconsistent = consistency.loc[
        consistency["Thickness A (mm)_nunique"].gt(1)
        | consistency["Thickness B (mm)_nunique"].gt(1),
        "Sample ID",
    ].astype(int).tolist()
    pull_inconsistent = consistency.loc[
        consistency["PullTest (N)_nunique"].gt(1), "Sample ID"
    ].astype(int).tolist()

    condition = source.groupby(
        ["pressure_psi", "welding_time_ms", "angle_deg", "condition_group"], sort=True
    ).agg(
        samples=("sample_id", "size"),
        good=("abnormal", lambda values: int((values == 0).sum())),
        abnormal=("abnormal", "sum"),
        bad=("category", lambda values: int((values == "Bad").sum())),
        explode=("category", lambda values: int((values == "Explode").sum())),
        abnormal_rate=("abnormal", "mean"),
        nugget_mean=("nugget_mm", "mean"),
        nugget_median=("nugget_mm", "median"),
        pulltest_mean=("pulltest_n", "mean"),
        pulltest_median=("pulltest_n", "median"),
    ).reset_index()
    condition["category_composition"] = condition.apply(
        lambda row: "|".join(
            name
            for name, value in [
                ("Good", row["good"]),
                ("Bad", row["bad"]),
                ("Explode", row["explode"]),
            ]
            if value > 0
        ),
        axis=1,
    )

    source_ranges = {
        "pressure_like": (source["pressure_bar"].min(), source["pressure_bar"].max()),
        "current": (source["current_min_ka"].min(), source["current_max_ka"].max()),
        "welding_time": (source["welding_time_ms"].min(), source["welding_time_ms"].max()),
        "thickness_a": (source["thickness_a_mm"].min(), source["thickness_a_mm"].max()),
        "thickness_b": (source["thickness_b_mm"].min(), source["thickness_b_mm"].max()),
    }
    kamp_values = {
        "pressure_like": kamp["F"],
        "current": kamp["I"],
        "welding_time": kamp["t"],
        "thickness_a": pd.Series(np.repeat(0.7, len(kamp))),
        "thickness_b": pd.Series(np.repeat(0.7, len(kamp))),
    }
    overlap_rows = []
    support_flags = pd.DataFrame(index=kamp.index)
    extrapolation = pd.DataFrame(index=kamp.index)
    for feature, (lower, upper) in source_ranges.items():
        values = kamp_values[feature].astype(float)
        inside = values.between(lower, upper)
        support_flags[feature] = inside
        below = np.maximum(lower - values, 0)
        above = np.maximum(values - upper, 0)
        source_iqr = {
            "pressure_like": source["pressure_bar"].quantile(0.75) - source["pressure_bar"].quantile(0.25),
            "current": source["current_mean_ka"].quantile(0.75) - source["current_mean_ka"].quantile(0.25),
            "welding_time": source["welding_time_ms"].quantile(0.75) - source["welding_time_ms"].quantile(0.25),
            "thickness_a": source["thickness_a_mm"].quantile(0.75) - source["thickness_a_mm"].quantile(0.25),
            "thickness_b": source["thickness_b_mm"].quantile(0.75) - source["thickness_b_mm"].quantile(0.25),
        }[feature]
        distance = below + above
        extrapolation[f"{feature}_boundary_distance"] = distance
        extrapolation[f"{feature}_boundary_distance_source_iqr"] = (
            distance / source_iqr if source_iqr else np.nan
        )
        overlap_rows.append(
            {
                "feature": feature,
                "source_min": lower,
                "source_max": upper,
                "kamp_min": values.min(),
                "kamp_max": values.max(),
                "kamp_rows_in_source_support": int(inside.sum()),
                "kamp_share_in_source_support": float(inside.mean()),
                "max_boundary_distance": float(distance.max()),
                "max_boundary_distance_source_iqr": float(distance.max() / source_iqr) if source_iqr else np.nan,
                "role": "negative_control_domain_mismatch",
            }
        )
    overlap = pd.DataFrame(overlap_rows)
    support_flags["joint"] = support_flags.all(axis=1)
    absolute_rows = kamp[["excel_row", "date", "F", "I", "V", "t"]].copy()
    for column in support_flags.columns:
        absolute_rows[f"in_source_{column}_support"] = support_flags[column].to_numpy()
    absolute_rows["n_features_in_support"] = support_flags.drop(columns="joint").sum(axis=1).to_numpy()
    absolute_rows = pd.concat([absolute_rows, extrapolation], axis=1)
    absolute_rows["raw_direct_model_score"] = np.nan
    absolute_rows["raw_direct_model_score_status"] = "not_run_stage0_fail"
    absolute_rows["role"] = "negative_control_not_transfer_gate"

    audit = {
        "experiment": config["experiment"],
        "status": stage0_status,
        "source_sha256": sha256(SOURCE_PATH),
        "kamp_sha256": sha256(KAMP_PATH),
        "source_raw_rows": len(source_raw),
        "source_columns": list(source_raw.columns),
        "source_unique_sample_ids": int(source_raw["Sample ID"].nunique()),
        "source_weld_rows": len(source),
        "source_category_counts": category_counts,
        "communication_error_sample_ids": communication["Sample ID"].astype(int).tolist(),
        "zero_valid_sensor_sample_ids": zero_valid_sensor_ids,
        "source_samples_with_valid_current_and_force": int(
            (~source["sample_id"].isin(zero_valid_sensor_ids)).sum()
        ),
        "thickness_inconsistent_sample_ids": thickness_inconsistent,
        "pulltest_inconsistent_sample_ids": pull_inconsistent,
        "condition_groups": int(source["condition_group"].nunique()),
        "kamp_rows": len(kamp),
        "kamp_dates": sorted(kamp["date"].unique().tolist()),
        "kamp_pressure_overlap_rows": int(support_flags["pressure_like"].sum()),
        "kamp_current_overlap_rows": int(support_flags["current"].sum()),
        "kamp_time_overlap_rows": int(support_flags["welding_time"].sum()),
        "kamp_joint_support_rows": int(support_flags["joint"].sum()),
        "absolute_overlap_role": "negative_control_not_transfer_gate",
        "result_rows": len(quality),
    }
    write_json(audit, "source_data_audit.json")
    write_csv(consistency, "source_sample_consistency_audit.csv")
    write_csv(source, "source_weld_level_495.csv")
    write_csv(condition, "source_condition_quality_table.csv")
    write_csv(overlap, "domain_overlap_summary.csv")
    write_csv(absolute_rows, "absolute_transfer_negative_control.csv")
    if zero_valid_sensor_ids:
        gate_rows = pd.DataFrame(
            [
                {
                    "gate": gate,
                    "scope": "global",
                    "status": "not_evaluated",
                    "metric_summary": "stage0_fail_source_sensor_valid_count_zero",
                    "consequence": "model_and_transfer_not_run",
                }
                for gate in ["A", "B", "C", "regression", "D", "E"]
            ]
        )
        write_csv(gate_rows, "transfer_gate_decisions.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    plot_specs = [
        ("pressure_like", source["pressure_bar"], kamp["F"], "Pressure-like (bar)"),
        ("current", source["current_mean_ka"], kamp["I"], "Current (kA)"),
        ("welding_time", source["welding_time_ms"], kamp["t"], "Welding time (ms)"),
    ]
    colors = ["#4c78a8", "#f58518"]
    for axis, (feature, source_values, kamp_series, label) in zip(axes, plot_specs):
        axis.boxplot(
            [source_values.dropna(), kamp_series.dropna()],
            tick_labels=[f"External\n(n={len(source_values)})", f"KAMP\n(n={len(kamp_series)})"],
            patch_artist=True,
            boxprops={"facecolor": colors[0], "alpha": 0.35},
            medianprops={"color": "black"},
        )
        axis.get_xticklabels()[1].set_color(colors[1])
        axis.set_title(label)
        match = overlap.loc[overlap["feature"].eq(feature)].iloc[0]
        axis.text(
            0.02,
            0.98,
            f"KAMP in source support: {int(match.kamp_rows_in_source_support):,}/{len(kamp):,}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Figure 1. Absolute domain mismatch negative control", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Absolute non-overlap diagnoses raw direct-transfer extrapolation; it is not a gate for domain-normalized transfer.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(FIG / "figure1_absolute_domain_mismatch_negative_control.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    """
    manifest = {
        "inputs": {
            str(SOURCE_PATH.relative_to(SOURCE_PATH.parents[1])).replace("\", "/"): sha256(SOURCE_PATH),
            str(KAMP_PATH.relative_to(KAMP_PATH.parents[1])).replace("\", "/"): sha256(KAMP_PATH),
        },
        "stage0_outputs": [
            "source_data_audit.json",
            "source_sample_consistency_audit.csv",
            "source_weld_level_495.csv",
            "source_condition_quality_table.csv",
            "domain_overlap_summary.csv",
            "absolute_transfer_negative_control.csv",
            "transfer_gate_decisions.csv",
            "figures/figure1_absolute_domain_mismatch_negative_control.png",
        ],
    }
    """
    manifest = {
        "inputs": {
            SOURCE_PATH.relative_to(SOURCE_PATH.parents[1]).as_posix(): sha256(SOURCE_PATH),
            KAMP_PATH.relative_to(KAMP_PATH.parents[1]).as_posix(): sha256(KAMP_PATH),
        },
        "stage0_outputs": [
            "source_data_audit.json",
            "source_sample_consistency_audit.csv",
            "source_weld_level_495.csv",
            "source_condition_quality_table.csv",
            "domain_overlap_summary.csv",
            "figures/figure1_absolute_domain_mismatch_negative_control.png",
        ],
    }
    write_json(manifest, "input_manifest.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if zero_valid_sensor_ids:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
