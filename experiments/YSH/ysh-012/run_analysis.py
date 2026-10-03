from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
YSH = HERE.parent
Y11A = YSH / "ysh-011a" / "outputs"
Y7 = YSH / "ysh-007" / "outputs"
Y8 = YSH / "ysh-008" / "outputs"
OUT = HERE / "outputs"
FIG = OUT / "figures"

VIEWS = [
    ("M1_mean", "K1"),
    ("M1_mean", "K2"),
    ("M2_max", "K1"),
    ("M2_max", "K2"),
]
RANK_COLUMNS = [f"rank_{mapping}_{reference}" for mapping, reference in VIEWS]
TIER_COLUMNS = {
    "strict_4of4_top5": "strict_4of4_top5",
    "sensitivity_3of4_top5": "sensitivity_3of4_top5",
    "sensitivity_median_ge_095": "sensitivity_median_ge_095",
    "stable_high": "stable_high",
}
RAW4 = ["F", "I", "V", "t"]


def write_csv(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.12g")


def write_json(payload: dict, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value).__name__)


def midpoint_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return pd.Series(values).rank(method="average").to_numpy(float) / (len(values) + 1.0)


def load_consensus_rows() -> tuple[pd.DataFrame, dict]:
    verification = json.loads((Y11A / "verification.json").read_text(encoding="utf-8-sig"))
    if not verification.get("passed"):
        raise RuntimeError("YSH-011A verification did not pass")

    scores = pd.read_csv(Y11A / "kamp_external_if_scores.csv", encoding="utf-8-sig")
    observed_views = set(zip(scores["mapping"], scores["target_reference"]))
    if observed_views != set(VIEWS):
        raise RuntimeError(f"Unexpected YSH-011A views: {sorted(observed_views)}")
    sizes = scores.groupby(["mapping", "target_reference"])["excel_row"].nunique()
    if len(scores) != 4 * 11939 or not sizes.eq(11939).all():
        raise RuntimeError("Unexpected YSH-011A score row count")

    base_columns = ["row_index", "excel_row", "date", *RAW4]
    base = (
        scores.loc[
            scores["mapping"].eq(VIEWS[0][0])
            & scores["target_reference"].eq(VIEWS[0][1]),
            base_columns,
        ]
        .sort_values("excel_row")
        .reset_index(drop=True)
    )
    if base["excel_row"].nunique() != 11939:
        raise RuntimeError("Base view does not contain 11,939 unique rows")

    rows = base.copy()
    for mapping, reference in VIEWS:
        part = scores.loc[
            scores["mapping"].eq(mapping) & scores["target_reference"].eq(reference),
            ["excel_row", "external_if_rank_percentile"],
        ].rename(columns={"external_if_rank_percentile": f"rank_{mapping}_{reference}"})
        rows = rows.merge(part, on="excel_row", validate="one_to_one")

    ranks = rows[RANK_COLUMNS]
    rows["top5_support"] = ranks.ge(0.95).sum(axis=1).astype(int)
    rows["median_rank"] = ranks.median(axis=1)
    rows["rank_min"] = ranks.min(axis=1)
    rows["rank_max"] = ranks.max(axis=1)
    rows["rank_range"] = rows["rank_max"] - rows["rank_min"]
    rows["rank_iqr"] = ranks.quantile(0.75, axis=1) - ranks.quantile(0.25, axis=1)
    rows["strict_4of4_top5"] = rows["top5_support"].eq(4)
    rows["sensitivity_3of4_top5"] = rows["top5_support"].ge(3)
    rows["sensitivity_median_ge_095"] = rows["median_rank"].ge(0.95)
    rows["stable_high"] = rows["median_rank"].ge(0.95) & rows["rank_range"].le(0.05)
    rows["interpretation"] = "reference_robust_external_anomaly_candidate_not_defect_label"
    return rows, verification


def attach_context(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    physical = pd.read_csv(Y7 / "physical_row_followup.csv", encoding="utf-8-sig")
    ae = pd.read_csv(Y8 / "ae_row_scores.csv", encoding="utf-8-sig")
    repeat = pd.read_csv(Y8 / "repeat298_mapping.csv", encoding="utf-8-sig")
    physical = physical.sort_values("excel_row").reset_index(drop=True)
    ae = ae.sort_values("excel_row").reset_index(drop=True)

    context = physical[
        [
            "excel_row",
            "global_nll",
            "mahalanobis_d2",
            "max_abs_conditional_z",
            "any_physical_candidate",
        ]
    ].merge(
        ae[["excel_row", "ae_total_error", "ae_candidate"]],
        on="excel_row",
        validate="one_to_one",
    )
    context["GMM_NLL_top5"] = midpoint_rank(context["global_nll"].to_numpy()) >= 0.95
    context["GMM_d2_top5"] = midpoint_rank(context["mahalanobis_d2"].to_numpy()) >= 0.95
    context["conditional_z_top5"] = midpoint_rank(context["max_abs_conditional_z"].to_numpy()) >= 0.95
    context["AE_total_top5"] = midpoint_rank(context["ae_total_error"].to_numpy()) >= 0.95
    context["conditional_z_ge_2_5"] = context["max_abs_conditional_z"].ge(2.5)
    context = context.rename(
        columns={
            "any_physical_candidate": "physical_candidate",
            "ae_candidate": "AE_candidate",
        }
    )
    joined = rows.merge(context, on="excel_row", validate="one_to_one")

    repeat_context = repeat[
        [
            "excel_row",
            "occurrence_id",
            "relative_position",
            "gmm_nll_q90",
            "gmm_d2_q90",
            "conditional_z_2_5",
            "physical_global_candidate",
        ]
    ].merge(
        ae[["excel_row", "ae_total_error", "ae_candidate"]],
        on="excel_row",
        validate="one_to_one",
    )
    joined = joined.merge(
        repeat_context[["excel_row", "occurrence_id", "relative_position"]],
        on="excel_row",
        how="left",
        validate="one_to_one",
    )
    joined["in_repeat298"] = joined["relative_position"].notna()
    return joined, repeat_context


def unique_raw4_table(rows: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    consistency_columns = RANK_COLUMNS + [
        "top5_support",
        "median_rank",
        "rank_range",
        *TIER_COLUMNS.values(),
    ]
    maximum_range = 0.0
    for column in consistency_columns:
        if pd.api.types.is_bool_dtype(rows[column]):
            inconsistent = rows.groupby(RAW4)[column].nunique().max()
            maximum_range = max(maximum_range, float(inconsistent - 1))
        else:
            ranges = rows.groupby(RAW4)[column].agg(lambda x: float(x.max() - x.min()))
            maximum_range = max(maximum_range, float(ranges.max()))
    if maximum_range > 1e-12:
        raise RuntimeError(f"Identical Raw4 received different consensus values: {maximum_range}")

    aggregations = {
        "row_count": ("excel_row", "size"),
        "date_count": ("date", "nunique"),
        "dates": ("date", lambda x: "|".join(sorted(set(map(str, x))))),
        "excel_row_min": ("excel_row", "min"),
        "excel_row_max": ("excel_row", "max"),
        "repeat298_row_count": ("in_repeat298", "sum"),
        "repeat298_occurrence_count": ("occurrence_id", "nunique"),
    }
    for column in consistency_columns:
        aggregations[column] = (column, "first")
    unique = rows.groupby(RAW4, as_index=False).agg(**aggregations)
    unique["in_repeat298"] = unique["repeat298_row_count"].gt(0)
    unique["interpretation"] = "unique_Raw4_motif_not_independent_defect_event"
    return unique, maximum_range


def candidate_summary(rows: pd.DataFrame, unique: pd.DataFrame) -> pd.DataFrame:
    output = []
    for tier, column in TIER_COLUMNS.items():
        row_selected = rows.loc[rows[column]]
        unique_selected = unique.loc[unique[column]]
        output.append(
            {
                "tier": tier,
                "definition": {
                    "strict_4of4_top5": "top5_support == 4",
                    "sensitivity_3of4_top5": "top5_support >= 3",
                    "sensitivity_median_ge_095": "median_rank >= 0.95",
                    "stable_high": "median_rank >= 0.95 and rank_range <= 0.05",
                }[tier],
                "row_count": len(row_selected),
                "row_fraction": len(row_selected) / len(rows),
                "unique_Raw4_count": len(unique_selected),
                "unique_Raw4_fraction": len(unique_selected) / len(unique),
                "row_to_unique_ratio": len(row_selected) / len(unique_selected) if len(unique_selected) else np.nan,
                "median_of_median_rank": row_selected["median_rank"].median(),
                "median_rank_range": row_selected["rank_range"].median(),
                "max_rank_range": row_selected["rank_range"].max(),
                "repeat298_rows": int(row_selected["in_repeat298"].sum()),
                "repeat298_row_fraction_within_tier": row_selected["in_repeat298"].mean(),
                "repeat298_unique_Raw4": int(unique_selected["in_repeat298"].sum()),
                "repeat298_unique_fraction_within_tier": unique_selected["in_repeat298"].mean(),
            }
        )
    return pd.DataFrame(output)


def overlap_table(rows: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "GMM_NLL_top5",
        "GMM_d2_top5",
        "conditional_z_top5",
        "AE_total_top5",
        "conditional_z_ge_2_5",
        "AE_candidate",
        "physical_candidate",
    ]
    output = []
    for tier, tier_column in TIER_COLUMNS.items():
        candidate = rows[tier_column].to_numpy(bool)
        for method in methods:
            existing = rows[method].to_numpy(bool)
            overlap = np.logical_and(candidate, existing).sum()
            union = np.logical_or(candidate, existing).sum()
            output.append(
                {
                    "tier": tier,
                    "existing_method": method,
                    "candidate_rows": int(candidate.sum()),
                    "existing_rows": int(existing.sum()),
                    "overlap_rows": int(overlap),
                    "candidate_coverage": overlap / candidate.sum() if candidate.sum() else np.nan,
                    "existing_coverage": overlap / existing.sum() if existing.sum() else np.nan,
                    "jaccard": overlap / union if union else np.nan,
                }
            )
    return pd.DataFrame(output)


def feature_profiles(rows: pd.DataFrame, unique: pd.DataFrame) -> pd.DataFrame:
    output = []
    for view_name, frame in [("row", rows), ("unique_Raw4", unique)]:
        for variable in RAW4:
            all_values = frame[variable].astype(float)
            all_median = all_values.median()
            all_q25 = all_values.quantile(0.25)
            all_q75 = all_values.quantile(0.75)
            all_iqr = all_q75 - all_q25
            for tier, column in TIER_COLUMNS.items():
                selected = frame.loc[frame[column], variable].astype(float)
                candidate_median = selected.median()
                output.append(
                    {
                        "analysis_view": view_name,
                        "tier": tier,
                        "variable": variable,
                        "candidate_n": len(selected),
                        "candidate_median": candidate_median,
                        "candidate_q25": selected.quantile(0.25),
                        "candidate_q75": selected.quantile(0.75),
                        "all_n": len(all_values),
                        "all_median": all_median,
                        "all_q25": all_q25,
                        "all_q75": all_q75,
                        "robust_median_shift": (
                            (candidate_median - all_median) / all_iqr if all_iqr > 0 else np.nan
                        ),
                    }
                )
    return pd.DataFrame(output)


def date_profile(rows: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "total_rows": ("excel_row", "size"),
        "unique_Raw4": ("excel_row", lambda x: rows.loc[x.index, RAW4].drop_duplicates().shape[0]),
    }
    for tier, column in TIER_COLUMNS.items():
        aggregations[f"{tier}_rows"] = (column, "sum")
    output = rows.groupby("date", as_index=False).agg(**aggregations)
    for tier in TIER_COLUMNS:
        output[f"{tier}_fraction"] = output[f"{tier}_rows"] / output["total_rows"]
    return output


def repeat_profile(rows: pd.DataFrame, repeat_context: pd.DataFrame) -> pd.DataFrame:
    joined = repeat_context.merge(
        rows[
            [
                "excel_row",
                *RANK_COLUMNS,
                "top5_support",
                "median_rank",
                "rank_range",
                *TIER_COLUMNS.values(),
            ]
        ],
        on="excel_row",
        validate="one_to_one",
    )
    aggregations = {
        "occurrence_count": ("occurrence_id", "nunique"),
        "top5_support": ("top5_support", "first"),
        "median_rank": ("median_rank", "first"),
        "rank_range": ("rank_range", "first"),
        "ae_candidate_share": ("ae_candidate", "mean"),
        "ae_total_error_median": ("ae_total_error", "median"),
        "gmm_nll_q90_share": ("gmm_nll_q90", "mean"),
        "gmm_d2_q90_share": ("gmm_d2_q90", "mean"),
        "conditional_z_2_5_share": ("conditional_z_2_5", "mean"),
        "physical_candidate_share": ("physical_global_candidate", "mean"),
    }
    for column in RANK_COLUMNS + list(TIER_COLUMNS.values()):
        aggregations[column] = (column, "first")
    return joined.groupby("relative_position", as_index=False).agg(**aggregations)


def make_figures(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    features: pd.DataFrame,
    overlaps: pd.DataFrame,
    dates: pd.DataFrame,
    repeat: pd.DataFrame,
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 130})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    support = rows["top5_support"].value_counts().sort_index().reindex(range(5), fill_value=0)
    axes[0].bar(support.index.astype(str), support.values, color="#4c78a8")
    axes[0].set(title="Top-5% support across four views", xlabel="support count", ylabel="KAMP rows")
    scatter = axes[1].scatter(
        rows["median_rank"], rows["rank_range"], c=rows["top5_support"], s=7, alpha=0.22, cmap="viridis"
    )
    axes[1].axvline(0.95, color="#e45756", linestyle="--", linewidth=1)
    axes[1].axhline(0.05, color="#f58518", linestyle=":", linewidth=1)
    axes[1].set(title="Rank level and dispersion", xlabel="median rank", ylabel="rank range")
    fig.colorbar(scatter, ax=axes[1], label="Top-5% support")
    fig.suptitle("Figure 1. Reference-robust candidate stability")
    fig.tight_layout()
    fig.savefig(FIG / "figure1_support_rank_stability.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    selected = features.loc[features["analysis_view"].eq("unique_Raw4")]
    pivot = selected.pivot(index="tier", columns="variable", values="robust_median_shift").loc[list(TIER_COLUMNS)]
    fig, axis = plt.subplots(figsize=(10, 5.5))
    vmax = max(1.0, float(np.nanmax(np.abs(pivot.to_numpy()))))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            axis.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    axis.set_title("Figure 2. Unique-Raw4 candidate median shift / overall IQR")
    fig.colorbar(image, ax=axis, label="robust median shift")
    fig.tight_layout()
    fig.savefig(FIG / "figure2_feature_profile.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pivot = overlaps.pivot(index="tier", columns="existing_method", values="jaccard").loc[list(TIER_COLUMNS)]
    fig, axis = plt.subplots(figsize=(13, 5.5))
    vmax = max(0.2, float(pivot.max().max()))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            axis.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8)
    axis.set_title("Figure 3. Candidate tiers versus existing anomaly sets — row Jaccard")
    fig.colorbar(image, ax=axis, label="Jaccard")
    fig.tight_layout()
    fig.savefig(FIG / "figure3_existing_overlap.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    x = np.arange(len(dates))
    axes[0].bar(x - 0.18, dates["strict_4of4_top5_fraction"], width=0.36, label="strict 4/4", color="#4c78a8")
    axes[0].bar(x + 0.18, dates["sensitivity_3of4_top5_fraction"], width=0.36, label=">=3/4", color="#f58518")
    axes[0].set_xticks(x, dates["date"], rotation=35, ha="right")
    axes[0].set(ylabel="fraction within date", title="Date concentration (Result not used)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].plot(repeat["relative_position"], repeat["median_rank"], color="#4c78a8", linewidth=1.3, label="median of four ranks")
    strict_positions = repeat.loc[repeat["strict_4of4_top5"], "relative_position"]
    axes[1].scatter(strict_positions, np.repeat(0.985, len(strict_positions)), s=18, color="#e45756", label="strict 4/4")
    axes[1].axhline(0.95, color="#e45756", linestyle="--", linewidth=1)
    axes[1].axvspan(1, 12, color="#f58518", alpha=0.12, label="AE 1–12")
    axes[1].axvspan(139, 143, color="#e45756", alpha=0.10, label="GMM local 139–143")
    axes[1].set(
        xlabel="repeat298 relative position (exact motif, not a cycle)",
        ylabel="median rank",
        title="repeat298 membership",
        ylim=(0, 1.01),
    )
    axes[1].legend(ncol=4, fontsize=8)
    axes[1].grid(alpha=0.15)
    fig.suptitle("Figure 4. Date and repeat298 structure of robust candidates")
    fig.tight_layout()
    fig.savefig(FIG / "figure4_date_repeat_structure.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def verify(
    rows: pd.DataFrame,
    unique: pd.DataFrame,
    summary: pd.DataFrame,
    overlaps: pd.DataFrame,
    features: pd.DataFrame,
    dates: pd.DataFrame,
    repeat: pd.DataFrame,
    raw4_max_range: float,
) -> dict:
    required_csv = [
        "robust_candidate_rows.csv",
        "robust_candidate_unique_raw4.csv",
        "robust_candidate_summary.csv",
        "existing_method_overlap.csv",
        "feature_profile.csv",
        "date_profile.csv",
        "repeat298_profile.csv",
    ]
    figures = [
        "figure1_support_rank_stability.png",
        "figure2_feature_profile.png",
        "figure3_existing_overlap.png",
        "figure4_date_repeat_structure.png",
    ]
    strict = summary.loc[summary["tier"].eq("strict_4of4_top5")].iloc[0]
    checks = {
        "ysh011a_input_rows_11939": len(rows) == 11939 and rows["excel_row"].nunique() == 11939,
        "four_frozen_rank_views": len(RANK_COLUMNS) == 4 and not rows[RANK_COLUMNS].isna().any().any(),
        "unique_Raw4_1574": len(unique) == 1574,
        "identical_Raw4_consensus_consistent": raw4_max_range <= 1e-12,
        "strict_candidate_nonempty": int(strict["row_count"]) > 0,
        "all_tiers_present": len(summary) == 4,
        "overlap_methods_complete": len(overlaps) == 4 * 7,
        "feature_profiles_complete": len(features) == 2 * 4 * 4,
        "date_rows_expected": len(dates) == rows["date"].nunique(),
        "repeat_positions_298": len(repeat) == 298,
        "required_csv_present": all((OUT / name).exists() for name in required_csv),
        "four_report_figures_present": all((FIG / name).exists() for name in figures),
        "new_model_training": False,
        "result_sheet_used": False,
        "ensemble_score_created": False,
    }
    passed = all(
        value
        for key, value in checks.items()
        if key not in {"new_model_training", "result_sheet_used", "ensemble_score_created"}
    ) and not any(checks[key] for key in ["new_model_training", "result_sheet_used", "ensemble_score_created"])
    motif_concentrated = bool(strict["repeat298_unique_fraction_within_tier"] >= 0.80)
    strict_overlap = overlaps.loc[overlaps["tier"].eq("strict_4of4_top5")]
    distinct_from_existing = bool(strict_overlap["jaccard"].lt(0.20).all())
    return {
        "experiment": "YSH-012",
        "checks": checks,
        "judgments": {
            "operationally_identified": bool(int(strict["row_count"]) > 0 and raw4_max_range <= 1e-12),
            "motif_concentrated_threshold_0_80": motif_concentrated,
            "distinct_from_existing_all_strict_jaccard_below_0_20": distinct_from_existing,
            "actual_reliability": "untestable_with_current_data",
        },
        "passed": bool(passed),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    rows, _ = load_consensus_rows()
    rows, repeat_context = attach_context(rows)
    unique, raw4_max_range = unique_raw4_table(rows)
    summary = candidate_summary(rows, unique)
    overlaps = overlap_table(rows)
    features = feature_profiles(rows, unique)
    dates = date_profile(rows)
    repeat = repeat_profile(rows, repeat_context)

    write_csv(rows, "robust_candidate_rows.csv")
    write_csv(unique, "robust_candidate_unique_raw4.csv")
    write_csv(summary, "robust_candidate_summary.csv")
    write_csv(overlaps, "existing_method_overlap.csv")
    write_csv(features, "feature_profile.csv")
    write_csv(dates, "date_profile.csv")
    write_csv(repeat, "repeat298_profile.csv")

    make_figures(rows, summary, features, overlaps, dates, repeat)
    verification = verify(rows, unique, summary, overlaps, features, dates, repeat, raw4_max_range)
    write_json(verification, "verification.json")
    if not verification["passed"]:
        raise RuntimeError(f"Verification failed: {verification}")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
