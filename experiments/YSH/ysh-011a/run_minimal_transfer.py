from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest


HERE = Path(__file__).resolve().parent
YSH = HERE.parent
Y11 = YSH / "ysh-011"
Y7_OUT = YSH / "ysh-007" / "outputs"
Y8_OUT = YSH / "ysh-008" / "outputs"
Y10_OUT = YSH / "ysh-010" / "outputs"
OUT = HERE / "outputs"
FIG = OUT / "figures"

MAPPINGS = {
    "M1_mean": "current_mean_ka",
    "M2_max": "current_max_ka",
}
REFERENCES = {
    "K1": "all_KAMP_rows",
    "K2": "unique_Raw4_equal_weight",
}
TOP_K = (0.01, 0.05, 0.10)


def load_ysh011():
    name = "ysh011_minimal_reuse"
    spec = importlib.util.spec_from_file_location(name, Y11 / "run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else np.nan


def exact_source_if_scores(runner, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, _, source = runner.source_raw_and_features(config)
    kamp = runner.load_kamp(config).sort_values("excel_row").reset_index(drop=True)
    target_values = kamp[["I", "t"]].to_numpy(float)
    unique_raw4 = kamp[["F", "I", "V", "t"]].drop_duplicates()
    reference_values = {
        "K1": target_values,
        "K2": unique_raw4[["I", "t"]].to_numpy(float),
    }

    rows: list[pd.DataFrame] = []
    metadata: list[dict] = []
    for mapping, current_column in MAPPINGS.items():
        source_values = source[[current_column, "welding_time_ms"]].to_numpy(float)
        source_q = runner.quantile_transform(source_values, source_values)
        model = IsolationForest(**config["isolation_forest"])
        model.fit(source_q)

        for target_reference, reference in reference_values.items():
            target_q = runner.quantile_transform(reference, target_values)
            score = -model.score_samples(target_q)
            rank = midpoint_rank(score)
            part = kamp[["excel_row", "date", "F", "I", "V", "t"]].copy()
            part.insert(0, "row_index", np.arange(len(part), dtype=int))
            part["mapping"] = mapping
            part["target_reference"] = target_reference
            part["external_if_score"] = score
            part["external_if_rank_percentile"] = rank
            for nominal in TOP_K:
                part[f"top_{int(nominal * 100)}pct"] = rank >= (1.0 - nominal)
            part["interpretation"] = "relative_anomaly_candidate_not_defect_label"
            rows.append(part)

            raw4_score_range = (
                part.groupby(["F", "I", "V", "t"])["external_if_score"]
                .agg(lambda x: float(x.max() - x.min()))
                .max()
            )
            metadata.append(
                {
                    "mapping": mapping,
                    "source_current_column": current_column,
                    "source_time_column": "welding_time_ms",
                    "source_samples": len(source),
                    "source_q_reference_samples": len(source_values),
                    "source_category_used_for_fit": False,
                    "model": "IsolationForest",
                    "n_estimators": config["isolation_forest"]["n_estimators"],
                    "max_samples": config["isolation_forest"]["max_samples"],
                    "contamination": config["isolation_forest"]["contamination"],
                    "random_state": config["isolation_forest"]["random_state"],
                    "target_reference": target_reference,
                    "target_reference_population": REFERENCES[target_reference],
                    "target_reference_rows": len(reference),
                    "max_same_raw4_score_range": raw4_score_range,
                }
            )

    scores = pd.concat(rows, ignore_index=True)
    return scores, pd.DataFrame(metadata)


def topk_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mapping, reference), part in scores.groupby(["mapping", "target_reference"], sort=False):
        for nominal in TOP_K:
            threshold = 1.0 - nominal
            selected = part.loc[part["external_if_rank_percentile"].ge(threshold)]
            rows.append(
                {
                    "mapping": mapping,
                    "target_reference": reference,
                    "top_k": nominal,
                    "rank_threshold": threshold,
                    "rows": len(selected),
                    "actual_fraction": len(selected) / len(part),
                    "unique_Raw4": selected[["F", "I", "V", "t"]].drop_duplicates().shape[0],
                    "tie_policy": "include_all_ties",
                }
            )
    return pd.DataFrame(rows)


def pair_stability(scores: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = []
    if mode == "reference":
        groups = [(mapping, part, "K1", "K2") for mapping, part in scores.groupby("mapping")]
        id_name = "mapping"
    elif mode == "mapping":
        groups = [(reference, part, "M1_mean", "M2_max") for reference, part in scores.groupby("target_reference")]
        id_name = "target_reference"
    else:
        raise ValueError(mode)

    for group_id, part, left_name, right_name in groups:
        column = "target_reference" if mode == "reference" else "mapping"
        score_wide = part.pivot(index="excel_row", columns=column, values="external_if_score")
        rank_wide = part.pivot(index="excel_row", columns=column, values="external_if_rank_percentile")
        for nominal in TOP_K:
            threshold = 1.0 - nominal
            rows.append(
                {
                    id_name: group_id,
                    "left": left_name,
                    "right": right_name,
                    "top_k": nominal,
                    "spearman_score": spearmanr(score_wide[left_name], score_wide[right_name]).statistic,
                    "spearman_rank": spearmanr(rank_wide[left_name], rank_wide[right_name]).statistic,
                    "jaccard": jaccard(
                        rank_wide[left_name].to_numpy() >= threshold,
                        rank_wide[right_name].to_numpy() >= threshold,
                    ),
                    "left_rows": int((rank_wide[left_name] >= threshold).sum()),
                    "right_rows": int((rank_wide[right_name] >= threshold).sum()),
                }
            )
    return pd.DataFrame(rows)


def existing_context() -> pd.DataFrame:
    physical = pd.read_csv(Y7_OUT / "physical_row_followup.csv", encoding="utf-8-sig")
    ae = pd.read_csv(Y8_OUT / "ae_row_scores.csv", encoding="utf-8-sig")
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
    context["conditional_z_2_5"] = context["max_abs_conditional_z"].ge(2.5)
    return context.sort_values("excel_row").reset_index(drop=True)


def matched_supervised(mapping: str, target_reference: str) -> pd.DataFrame:
    y10 = pd.read_csv(Y10_OUT / "kamp_external_transfer_scores.csv", encoding="utf-8-sig")
    reference = {"K1": "K1_global", "K2": "K2_unique_raw4"}[target_reference]
    selected = y10.loc[
        y10["representation"].eq("Q")
        & y10["mapping"].eq(mapping)
        & y10["target_reference"].eq(reference),
        ["excel_row", "external_abnormal_similarity"],
    ].copy()
    if selected["excel_row"].nunique() != 11939 or len(selected) != 11939:
        raise RuntimeError(f"Unexpected matched YSH-010 score rows: {mapping} {target_reference}")
    return selected


def compare_existing(scores: pd.DataFrame) -> pd.DataFrame:
    context = existing_context()
    continuous_columns = {
        "GMM_NLL": "global_nll",
        "GMM_d2": "mahalanobis_d2",
        "conditional_z_score": "max_abs_conditional_z",
        "AE_total_error": "ae_total_error",
    }
    native_columns = {
        "conditional_z_candidate": "conditional_z_2_5",
        "AE_candidate": "ae_candidate",
        "physical_candidate": "any_physical_candidate",
    }
    rows = []
    for (mapping, reference), part in scores.groupby(["mapping", "target_reference"], sort=False):
        aligned = part.sort_values("excel_row").merge(context, on="excel_row", validate="one_to_one")
        supervised = matched_supervised(mapping, reference)
        aligned = aligned.merge(supervised, on="excel_row", validate="one_to_one")
        methods: dict[str, tuple[np.ndarray, str]] = {
            name: (aligned[column].to_numpy(float), "top_k_rank")
            for name, column in continuous_columns.items()
        }
        methods["YSH010_supervised_matched"] = (
            aligned["external_abnormal_similarity"].to_numpy(float),
            "top_k_rank",
        )
        methods.update(
            {
                name: (aligned[column].to_numpy(bool), "native_candidate")
                for name, column in native_columns.items()
            }
        )
        external_score = aligned["external_if_score"].to_numpy(float)
        external_rank = aligned["external_if_rank_percentile"].to_numpy(float)
        for method, (existing, selection_rule) in methods.items():
            is_continuous = selection_rule == "top_k_rank"
            existing_rank = midpoint_rank(existing.astype(float)) if is_continuous else None
            rho = spearmanr(external_score, existing).statistic if is_continuous else np.nan
            for nominal in TOP_K:
                threshold = 1.0 - nominal
                existing_flag = (
                    existing_rank >= threshold if is_continuous else existing.astype(bool)
                )
                external_flag = external_rank >= threshold
                rows.append(
                    {
                        "mapping": mapping,
                        "target_reference": reference,
                        "existing_method": method,
                        "existing_selection_rule": selection_rule,
                        "top_k": nominal,
                        "spearman": rho,
                        "jaccard": jaccard(external_flag, existing_flag),
                        "external_if_rows": int(external_flag.sum()),
                        "existing_rows": int(existing_flag.sum()),
                    }
                )
    return pd.DataFrame(rows)


def repeat298_profile(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(Y8_OUT / "repeat298_mapping.csv", encoding="utf-8-sig")
    ae = pd.read_csv(Y8_OUT / "ae_row_scores.csv", encoding="utf-8-sig")
    mapping = mapping.merge(
        ae[["excel_row", "ae_total_error", "ae_candidate"]],
        on="excel_row",
        validate="one_to_one",
    )
    joined = mapping.merge(
        scores[
            [
                "excel_row",
                "mapping",
                "target_reference",
                "external_if_score",
                "external_if_rank_percentile",
            ]
        ],
        on="excel_row",
        validate="one_to_many",
    )
    group = ["mapping", "target_reference", "relative_position"]
    profile = (
        joined.groupby(group, as_index=False)
        .agg(
            occurrence_count=("occurrence_id", "nunique"),
            external_if_score_median=("external_if_score", "median"),
            external_if_score_min=("external_if_score", "min"),
            external_if_score_max=("external_if_score", "max"),
            external_if_rank_median=("external_if_rank_percentile", "median"),
            external_if_rank_min=("external_if_rank_percentile", "min"),
            external_if_rank_max=("external_if_rank_percentile", "max"),
            top5_occurrence_share=("external_if_rank_percentile", lambda x: float((x >= 0.95).mean())),
            top10_occurrence_share=("external_if_rank_percentile", lambda x: float((x >= 0.90).mean())),
            ae_total_error_median=("ae_total_error", "median"),
            ae_candidate_share=("ae_candidate", "mean"),
            gmm_nll_q90_share=("gmm_nll_q90", "mean"),
            gmm_d2_q90_share=("gmm_d2_q90", "mean"),
            conditional_z_2_5_share=("conditional_z_2_5", "mean"),
            physical_candidate_share=("physical_global_candidate", "mean"),
        )
    )
    sanity = (
        joined.groupby(["mapping", "target_reference", "relative_position"])["external_if_score"]
        .agg(["min", "max"])
        .reset_index()
    )
    sanity["score_range"] = sanity["max"] - sanity["min"]
    sanity = (
        sanity.groupby(["mapping", "target_reference"], as_index=False)
        .agg(
            relative_positions=("relative_position", "nunique"),
            max_occurrence_score_range=("score_range", "max"),
        )
    )
    sanity["identical_within_tolerance_1e_12"] = sanity["max_occurrence_score_range"].le(1e-12)
    return profile, sanity


def make_figures(
    scores: pd.DataFrame,
    k1_k2: pd.DataFrame,
    comparisons: pd.DataFrame,
    profile: pd.DataFrame,
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 130})
    colors = {"K1": "#4c78a8", "K2": "#f58518"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for column, mapping in enumerate(MAPPINGS):
        for reference, part in scores.loc[scores["mapping"].eq(mapping)].groupby("target_reference"):
            axes[0, column].hist(
                part["external_if_score"], bins=60, alpha=0.48, color=colors[reference], label=reference
            )
            axes[1, column].hist(
                part["external_if_rank_percentile"], bins=50, alpha=0.48, color=colors[reference], label=reference
            )
        axes[0, column].set_title(f"{mapping}: external IF score")
        axes[1, column].set_title(f"{mapping}: midpoint rank")
        axes[0, column].legend()
        axes[1, column].legend()
        axes[0, column].grid(alpha=0.15)
        axes[1, column].grid(alpha=0.15)
    fig.suptitle("Figure 1. KAMP external IF distributions — F2 only")
    fig.tight_layout()
    fig.savefig(FIG / "figure1_external_if_distributions.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, mapping in zip(axes, MAPPINGS):
        part = scores.loc[scores["mapping"].eq(mapping)]
        wide = part.pivot(index="excel_row", columns="target_reference", values="external_if_score")
        rho = k1_k2.loc[k1_k2["mapping"].eq(mapping), "spearman_score"].iloc[0]
        axis.scatter(wide["K1"], wide["K2"], s=7, alpha=0.16, color="#4c78a8", edgecolors="none")
        axis.set(title=f"{mapping} (Spearman={rho:.3f})", xlabel="K1 score", ylabel="K2 score")
        axis.grid(alpha=0.2)
    fig.suptitle("Figure 2. K1 versus K2 external IF score")
    fig.tight_layout()
    fig.savefig(FIG / "figure2_k1_k2_scatter.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    top5 = comparisons.loc[comparisons["top_k"].eq(0.05)].copy()
    top5["combination"] = top5["mapping"] + " / " + top5["target_reference"]
    pivot = top5.pivot(index="combination", columns="existing_method", values="jaccard")
    fig, axis = plt.subplots(figsize=(14, 5.5))
    image = axis.imshow(pivot.to_numpy(float), aspect="auto", cmap="Blues", vmin=0, vmax=max(0.3, pivot.max().max()))
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            axis.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8)
    axis.set_title("Figure 3. External IF versus existing KAMP anomaly — Top-5% Jaccard")
    fig.colorbar(image, ax=axis, label="Jaccard")
    fig.tight_layout()
    fig.savefig(FIG / "figure3_existing_top5_jaccard.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for axis, mapping in zip(axes, MAPPINGS):
        selected = profile.loc[profile["mapping"].eq(mapping)]
        for reference, part in selected.groupby("target_reference"):
            axis.plot(
                part["relative_position"],
                part["external_if_rank_median"],
                color=colors[reference],
                linewidth=1.5,
                label=reference,
            )
        context = selected.loc[selected["target_reference"].eq("K1")]
        physical_positions = context.loc[context["physical_candidate_share"].gt(0), "relative_position"]
        axis.scatter(physical_positions, np.repeat(0.02, len(physical_positions)), marker="|", s=55, color="#54a24b", label="physical candidate position")
        axis.axhline(0.95, color="#e45756", linestyle="--", linewidth=1, label="Top 5%")
        axis.axhline(0.90, color="#e45756", linestyle=":", linewidth=1, label="Top 10%")
        axis.axvspan(1, 12, color="#f58518", alpha=0.12, label="AE 1–12")
        axis.axvspan(139, 143, color="#e45756", alpha=0.10, label="GMM local 139–143")
        axis.set(ylabel="external IF rank", title=mapping, ylim=(0, 1.01))
        axis.grid(alpha=0.15)
        axis.legend(ncol=4, fontsize=8, loc="lower right")
    axes[-1].set_xlabel("repeat298 relative position (exact repeated motif; not named a cycle)")
    fig.suptitle("Figure 4. repeat298 external IF rank profile")
    fig.tight_layout()
    fig.savefig(FIG / "figure4_repeat298_if_rank_profile.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def verify(
    scores: pd.DataFrame,
    metadata: pd.DataFrame,
    topk: pd.DataFrame,
    k1_k2: pd.DataFrame,
    m1_m2: pd.DataFrame,
    comparisons: pd.DataFrame,
    profile: pd.DataFrame,
    repeat_sanity: pd.DataFrame,
) -> dict:
    required = [
        "kamp_external_if_scores.csv",
        "kamp_external_if_topk.csv",
        "k1_k2_stability.csv",
        "m1_m2_stability.csv",
        "existing_anomaly_comparison.csv",
        "repeat298_if_profile.csv",
        "verification.json",
    ]
    figure_names = [
        "figure1_external_if_distributions.png",
        "figure2_k1_k2_scatter.png",
        "figure3_existing_top5_jaccard.png",
        "figure4_repeat298_if_rank_profile.png",
    ]
    group_sizes = scores.groupby(["mapping", "target_reference"])["excel_row"].nunique()
    checks = {
        "required_csv_files_present": all((OUT / name).exists() for name in required if name.endswith(".csv")),
        "score_rows_exactly_4_x_11939": len(scores) == 4 * 11939 and bool(group_sizes.eq(11939).all()),
        "score_groups_exactly_4": len(group_sizes) == 4,
        "only_F2_features": set(metadata["source_time_column"]) == {"welding_time_ms"}
        and set(metadata["source_current_column"]) == set(MAPPINGS.values()),
        "only_M1_M2": set(scores["mapping"]) == set(MAPPINGS),
        "only_K1_K2": set(scores["target_reference"]) == set(REFERENCES),
        "source_samples_493": set(metadata["source_samples"]) == {493},
        "source_category_not_used_for_fit": not metadata["source_category_used_for_fit"].astype(bool).any(),
        "if_configuration_frozen": set(metadata["n_estimators"]) == {500}
        and set(metadata["max_samples"]) == {"auto"}
        and set(metadata["contamination"]) == {"auto"}
        and set(metadata["random_state"]) == {42},
        "topk_rows_expected": len(topk) == 12,
        "k1_k2_rows_expected": len(k1_k2) == 6,
        "m1_m2_rows_expected": len(m1_m2) == 6,
        "comparison_rows_expected": len(comparisons) == 4 * 8 * 3,
        "repeat_profile_rows_expected": len(profile) == 4 * 298,
        "repeat298_identical_scores": bool(repeat_sanity["identical_within_tolerance_1e_12"].all()),
        "minimal_report_figures_present": all((FIG / name).exists() for name in figure_names),
        "result_sheet_used": False,
    }
    payload = {
        "experiment": "YSH-011A",
        "scope": "F2_M1_M2_IF_K1_K2_only",
        "required_files": required,
        "minimal_report_figures": figure_names,
        "checks": checks,
        "passed": bool(all(value for key, value in checks.items() if key != "result_sheet_used") and not checks["result_sheet_used"]),
    }
    return payload


def main() -> None:
    runner = load_ysh011()
    config = runner.read_config()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    scores, metadata = exact_source_if_scores(runner, config)
    write_csv(scores, "kamp_external_if_scores.csv")
    write_csv(metadata, "source_if_model_metadata.csv")

    topk = topk_summary(scores)
    k1_k2 = pair_stability(scores, "reference")
    m1_m2 = pair_stability(scores, "mapping")
    comparisons = compare_existing(scores)
    profile, repeat_sanity = repeat298_profile(scores)
    write_csv(topk, "kamp_external_if_topk.csv")
    write_csv(k1_k2, "k1_k2_stability.csv")
    write_csv(m1_m2, "m1_m2_stability.csv")
    write_csv(comparisons, "existing_anomaly_comparison.csv")
    write_csv(profile, "repeat298_if_profile.csv")
    write_csv(repeat_sanity, "repeat298_if_sanity.csv")

    make_figures(scores, k1_k2, comparisons, profile)
    verification = verify(
        scores,
        metadata,
        topk,
        k1_k2,
        m1_m2,
        comparisons,
        profile,
        repeat_sanity,
    )
    write_json(verification, "verification.json")
    if not verification["passed"]:
        raise RuntimeError(f"Verification failed: {verification}")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
