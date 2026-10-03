from __future__ import annotations

import json
from itertools import combinations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from common import (
    FIG,
    MODELS,
    OUT,
    Y7,
    Y8,
    aggregate_source,
    ensure_output_dirs,
    episode_ids,
    json_default,
    load_kamp,
    load_source_raw,
    quantile_transform,
    read_config,
    robust_fit,
    robust_transform,
    source_model_cohort,
    source_transform_apply,
    source_transform_fit,
    write_csv,
    write_json,
)
from transfer_experiment import MAPPINGS, fit_logistic, transfer_matrix

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 130})
FEATURE_NAMES = ["pressure_like", "current", "time"]


def target_reference_matrix(
    kamp: pd.DataFrame,
    physical: pd.DataFrame,
    representation: str,
    reference: str,
) -> tuple[np.ndarray, dict]:
    raw = kamp[["F", "I", "t"]].to_numpy(float)
    if reference == "K1_global":
        values = raw
        ref = raw
        population = "all_KAMP_rows"
    elif reference == "K2_unique_raw4":
        values = raw
        ref = kamp[["F", "I", "V", "t"]].drop_duplicates()[["F", "I", "t"]].to_numpy(float)
        population = "unique_KAMP_Raw4_equal_weight"
    elif reference == "K3_gmm_relative":
        values = physical[["conditional_z_F", "conditional_z_I", "conditional_z_t"]].to_numpy(float)
        ref = values
        population = "YSH007_signed_conditional_z_all_rows"
    else:
        raise ValueError(reference)

    if representation == "Q":
        transformed = quantile_transform(ref, values)
        stats = {"reference_population": population, "n": len(ref)}
    elif representation == "RZ":
        median, iqr = robust_fit(ref)
        transformed = robust_transform(values, median, iqr)
        stats = {"reference_population": population, "n": len(ref), "median": median, "iqr": iqr}
    else:
        raise ValueError(representation)
    return transformed, stats


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else np.nan


def build_reference_rows(reference: str, representation: str, stats: dict) -> list[dict]:
    rows = []
    for index, feature in enumerate(FEATURE_NAMES):
        row = {
            "representation": representation,
            "target_reference": reference,
            "feature": feature,
            "reference_population": stats["reference_population"],
            "n": stats["n"],
        }
        if "median" in stats:
            row["median"] = stats["median"][index]
            row["iqr"] = stats["iqr"][index]
        rows.append(row)
    return rows


def source_models_and_scores(
    source: pd.DataFrame,
    kamp: pd.DataFrame,
    physical: pd.DataFrame,
    thresholds: pd.DataFrame,
    combo_status: dict,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[str]]:
    y = source["abnormal"].to_numpy(int)
    base = kamp[["excel_row", "date", "F", "I", "V", "t"]].copy()
    score_frames = []
    reference_rows = []
    evaluated = []
    target_cache: dict[tuple[str, str], np.ndarray] = {}

    for representation in ["RZ", "Q"]:
        for mapping in MAPPINGS:
            key = f"{representation}_{mapping}"
            if not combo_status[key]["B"]:
                continue
            source_x = transfer_matrix(source, mapping)
            transform = source_transform_fit(source_x, y, representation)
            model = fit_logistic(config).fit(source_transform_apply(source_x, transform), y)
            threshold_row = thresholds.loc[
                thresholds["representation"].eq(representation) & thresholds["mapping"].eq(mapping)
            ]
            if len(threshold_row) != 1:
                raise AssertionError(f"Missing frozen threshold for {key}")
            threshold = float(threshold_row.iloc[0]["threshold"])
            joblib.dump(
                {"model": model, "source_transform": transform, "threshold": threshold, "mapping": mapping},
                MODELS / f"source_logistic_{representation}_{mapping}.joblib",
            )
            for reference in ["K1_global", "K2_unique_raw4", "K3_gmm_relative"]:
                cache_key = (representation, reference)
                if cache_key not in target_cache:
                    target_cache[cache_key], stats = target_reference_matrix(
                        kamp, physical, representation, reference
                    )
                    reference_rows.extend(build_reference_rows(reference, representation, stats))
                score = model.decision_function(target_cache[cache_key])
                flag = score >= threshold
                combo = f"{representation}_{reference}_{mapping}"
                evaluated.append(combo)
                frame = base.copy()
                frame["representation"] = representation
                frame["target_reference"] = reference
                frame["mapping"] = mapping
                frame["combo"] = combo
                frame["external_abnormal_similarity"] = score
                frame["source_oof_threshold"] = threshold
                frame["external_abnormal_like"] = flag
                frame["gate_B_pass"] = True
                frame["interpretation"] = "relative_similarity_not_defect_label"
                score_frames.append(frame)
    if not score_frames:
        raise RuntimeError("No Gate-B-passing source combination is eligible for KAMP scoring")
    return pd.concat(score_frames, ignore_index=True), pd.DataFrame(reference_rows), evaluated, list(target_cache)


def absolute_negative_control(source: pd.DataFrame, kamp: pd.DataFrame, config: dict) -> pd.DataFrame:
    frame = pd.read_csv(OUT / "absolute_transfer_negative_control.csv", encoding="utf-8-sig")
    y = source["abnormal"].to_numpy(int)
    target = kamp[["F", "I", "t"]].to_numpy(float)
    for mapping in MAPPINGS:
        x = transfer_matrix(source, mapping)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(
            C=config["logistic"]["C"], class_weight="balanced", max_iter=config["logistic"]["max_iter"], random_state=42
        ).fit(scaler.transform(x), y)
        frame[f"raw_direct_model_score_{mapping}"] = model.decision_function(scaler.transform(target))
    frame["raw_direct_model_score"] = frame[[f"raw_direct_model_score_{m}" for m in MAPPINGS]].median(axis=1)
    frame["raw_direct_model_score_status"] = "diagnostic_only_all_rows_outside_joint_support"
    return frame


def build_row_summary(scores: pd.DataFrame, physical: pd.DataFrame) -> pd.DataFrame:
    identity = ["excel_row", "date", "F", "I", "V", "t"]
    primary = scores.loc[scores["target_reference"].isin(["K1_global", "K2_unique_raw4"])].copy()
    flags = primary.pivot(index="excel_row", columns="combo", values="external_abnormal_like")
    values = primary.pivot(index="excel_row", columns="combo", values="external_abnormal_similarity")
    row = scores[identity].drop_duplicates("excel_row").set_index("excel_row")
    row["eligible_primary_support_count"] = flags.sum(axis=1).astype(int)
    row["eligible_primary_combo_count"] = flags.shape[1]
    row["eligible_primary_support_fraction"] = flags.mean(axis=1)
    row["eligible_primary_score_median"] = values.median(axis=1)
    row["external_abnormal_like_any_eligible"] = flags.any(axis=1)
    row["transfer_consensus_candidate"] = False
    row["transfer_consensus_status"] = "not_available_incomplete_preregistered_12_combo_family"
    sensitivity = scores.loc[scores["target_reference"].eq("K3_gmm_relative")].pivot(
        index="excel_row", columns="combo", values="external_abnormal_like"
    )
    row["eligible_k3_support_count"] = sensitivity.sum(axis=1).astype(int)
    row["transfer_consensus_with_gmm_sensitivity"] = False
    row = row.reset_index()
    row = row.merge(
        physical[["excel_row", "safe_segment"]], on="excel_row", how="left", validate="one_to_one"
    )
    return row


def stability_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, subset in [
        ("eligible_primary_K1K2", scores.loc[scores["target_reference"].isin(["K1_global", "K2_unique_raw4"])]),
        ("eligible_all_K1K2K3", scores),
    ]:
        wide_score = subset.pivot(index="excel_row", columns="combo", values="external_abnormal_similarity")
        wide_flag = subset.pivot(index="excel_row", columns="combo", values="external_abnormal_like")
        for left, right in combinations(wide_score.columns, 2):
            rows.append(
                {
                    "family": family,
                    "combo_left": left,
                    "combo_right": right,
                    "spearman": spearmanr(wide_score[left], wide_score[right]).statistic,
                    "jaccard": jaccard(wide_flag[left], wide_flag[right]),
                    "left_candidates": int(wide_flag[left].sum()),
                    "right_candidates": int(wide_flag[right].sum()),
                }
            )
    return pd.DataFrame(rows)


def count_table(scores: pd.DataFrame, kamp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    episodes = []
    context = kamp[["excel_row", "date", "F", "I", "V", "t"]].copy()
    physical = pd.read_csv(Y7 / "physical_row_followup.csv", encoding="utf-8-sig")
    context = context.merge(physical[["excel_row", "safe_segment"]], on="excel_row", how="left", validate="one_to_one")
    for combo, part in scores.groupby("combo", sort=True):
        flags = part.sort_values("excel_row")["external_abnormal_like"].to_numpy(bool)
        flagged = context.loc[flags].copy()
        context["flag"] = flags
        context["episode_id"] = episode_ids(context, "flag")
        rows.append(
            {
                "combo": combo,
                "rows": len(context),
                "candidate_rows": int(flags.sum()),
                "candidate_share": float(flags.mean()),
                "candidate_unique_raw4": int(flagged[["F", "I", "V", "t"]].drop_duplicates().shape[0]),
                "candidate_episodes": int(context["episode_id"].max()),
            }
        )
        for episode_id, episode in context.loc[context["episode_id"].gt(0)].groupby("episode_id"):
            episodes.append(
                {
                    "combo": combo,
                    "episode_id": int(episode_id),
                    "date": episode["date"].iloc[0],
                    "safe_segment": episode["safe_segment"].iloc[0],
                    "start_excel_row": int(episode["excel_row"].min()),
                    "end_excel_row": int(episode["excel_row"].max()),
                    "rows": len(episode),
                    "interpretation": "candidate_episode_not_defect_event",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(episodes)


def existing_overlap(scores: pd.DataFrame, kamp: pd.DataFrame, physical: pd.DataFrame) -> pd.DataFrame:
    ae = pd.read_csv(Y8 / "ae_row_scores.csv", encoding="utf-8-sig")
    physical_columns = [
        "excel_row",
        "high_global_nll",
        "high_internal_d2",
        "max_abs_conditional_z",
        "any_physical_candidate",
    ]
    context = kamp[["excel_row", "F", "I", "V", "t"]].merge(
        physical[physical_columns], on="excel_row", how="left", validate="one_to_one"
    ).merge(ae[["excel_row", "ae_candidate"]], on="excel_row", how="left", validate="one_to_one")
    methods = {
        "AE": context["ae_candidate"].astype(bool).to_numpy(),
        "GMM_NLL_q90": context["high_global_nll"].astype(bool).to_numpy(),
        "GMM_d2_q90": context["high_internal_d2"].astype(bool).to_numpy(),
        "conditional_z_2_5": context["max_abs_conditional_z"].ge(2.5).to_numpy(),
        "physical_global": context["any_physical_candidate"].astype(bool).to_numpy(),
    }
    rows = []
    raw_keys = pd.MultiIndex.from_frame(context[["F", "I", "V", "t"]])
    for combo, part in scores.groupby("combo", sort=True):
        transfer = part.sort_values("excel_row")["external_abnormal_like"].to_numpy(bool)
        for method, other in methods.items():
            transfer_keys = set(raw_keys[transfer])
            method_keys = set(raw_keys[other])
            rows.append(
                {
                    "combo": combo,
                    "method": method,
                    "role": "eligible_combo_diagnostic_not_GateE_consensus",
                    "transfer_rows": int(transfer.sum()),
                    "method_rows": int(other.sum()),
                    "row_intersection": int(np.logical_and(transfer, other).sum()),
                    "row_jaccard": jaccard(transfer, other),
                    "transfer_unique_raw4": len(transfer_keys),
                    "method_unique_raw4": len(method_keys),
                    "unique_raw4_intersection": len(transfer_keys & method_keys),
                    "unique_raw4_jaccard": len(transfer_keys & method_keys) / len(transfer_keys | method_keys) if transfer_keys | method_keys else np.nan,
                }
            )
    return pd.DataFrame(rows)


def repeat_profile(scores: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(Y8 / "repeat298_mapping.csv", encoding="utf-8-sig")
    ae = pd.read_csv(Y8 / "ae_row_scores.csv", encoding="utf-8-sig")
    mapping = mapping.merge(
        ae[["excel_row", "ae_candidate"]], on="excel_row", how="left", validate="one_to_one"
    )
    joined = mapping.merge(
        scores[["excel_row", "combo", "target_reference", "external_abnormal_similarity", "external_abnormal_like"]],
        on="excel_row", how="left", validate="one_to_many"
    )
    sanity = []
    for combo, part in joined.loc[joined["target_reference"].isin(["K1_global", "K2_unique_raw4"])].groupby("combo"):
        wide = part.pivot(index="relative_position", columns="occurrence_id", values="external_abnormal_similarity")
        sanity.append(
            {
                "combo": combo,
                "relative_positions": len(wide),
                "occurrences": wide.shape[1],
                "max_occurrence_score_range": float((wide.max(axis=1) - wide.min(axis=1)).max()),
                "identical_within_tolerance_1e_12": bool(((wide.max(axis=1) - wide.min(axis=1)).abs() <= 1e-12).all()),
            }
        )
    profile = joined.groupby("relative_position", sort=True).agg(
        transfer_score_median=("external_abnormal_similarity", "median"),
        transfer_flag_share=("external_abnormal_like", "mean"),
        ae_candidate_share=("ae_candidate", "mean"),
        gmm_nll_q90_share=("gmm_nll_q90", "mean"),
        gmm_d2_q90_share=("gmm_d2_q90", "mean"),
        conditional_z_2_5_share=("conditional_z_2_5", "mean"),
        physical_global_share=("physical_global_candidate", "mean"),
    ).reset_index()
    return profile, pd.DataFrame(sanity)


def regression_directions(
    source: pd.DataFrame,
    kamp: pd.DataFrame,
    physical: pd.DataFrame,
    regression_metrics: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    gate = config["gate"]
    passing = regression_metrics.loc[
        regression_metrics["model"].eq("ridge")
        & regression_metrics["split"].eq("group")
        & regression_metrics["r2"].gt(gate["regression_r2_min_exclusive"])
        & regression_metrics["spearman"].gt(gate["regression_spearman_min_exclusive"])
        & regression_metrics["dummy_mae_improvement"].ge(gate["regression_dummy_mae_improvement_min"])
    ]
    rows = []
    for metric in passing.itertuples(index=False):
        valid = source[metric.target].notna()
        if metric.target == "pulltest_n":
            valid &= ~source["sample_id"].isin([161, 185, 213])
        subset = source.loc[valid].reset_index(drop=True)
        y = subset[metric.target].to_numpy(float)
        x = transfer_matrix(subset, metric.mapping)
        transform = source_transform_fit(x, subset["abnormal"].to_numpy(int), metric.representation)
        model = Ridge(alpha=config["ridge_alpha"]).fit(source_transform_apply(x, transform), y)
        for reference in ["K1_global", "K2_unique_raw4", "K3_gmm_relative"]:
            target_x, _ = target_reference_matrix(kamp, physical, metric.representation, reference)
            centered = model.predict(target_x) - np.median(y)
            direction = -centered
            for excel_row, date, value in zip(kamp["excel_row"], kamp["date"], direction):
                rows.append(
                    {
                        "excel_row": excel_row,
                        "date": date,
                        "target": metric.target,
                        "representation": metric.representation,
                        "target_reference": reference,
                        "mapping": metric.mapping,
                        "relative_low_quality_direction_score": value,
                        "interpretation": "relative_direction_not_absolute_quality_prediction",
                    }
                )
    return pd.DataFrame(rows)


def figures(source: pd.DataFrame, kamp: pd.DataFrame, physical: pd.DataFrame, scores: pd.DataFrame, stability: pd.DataFrame, profile: pd.DataFrame) -> None:
    y = source["abnormal"].to_numpy(int)
    source_x = transfer_matrix(source, "M1_mean")
    source_q = source_transform_apply(source_x, source_transform_fit(source_x, y, "Q"))
    k1, _ = target_reference_matrix(kamp, physical, "Q", "K1_global")
    k2, _ = target_reference_matrix(kamp, physical, "Q", "K2_unique_raw4")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for index, (axis, feature) in enumerate(zip(axes, FEATURE_NAMES)):
        axis.boxplot(
            [source_q[y == 0, index], source_q[y == 1, index], k1[:, index], k2[:, index]],
            tick_labels=["Source Good", "Source Abnormal", "KAMP K1", "KAMP K2"],
            showfliers=False,
        )
        axis.set_title(feature)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Figure 5. Q-space domain-normalized feature comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "figure5_domain_normalized_feature_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for combo, part in scores.groupby("combo"):
        axes[0].hist(part["external_abnormal_similarity"], bins=50, alpha=0.35, label=combo)
    axes[0].set_title("Eligible KAMP similarity-score distributions")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.2)
    primary = stability.loc[stability["family"].eq("eligible_primary_K1K2")]
    axes[1].scatter(primary["spearman"], primary["jaccard"], color="#4c78a8")
    axes[1].axvline(0.70, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(0.50, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Spearman", ylabel="Flag Jaccard", title="Eligible pairwise stability")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Figure 6. Eligible transfer scores and stability (incomplete 12-combo family)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "figure6_transfer_score_stability.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    axes[0].plot(profile["relative_position"], profile["transfer_score_median"], color="#4c78a8")
    axes[0].set_ylabel("Median eligible score")
    for column, label, color in [
        ("transfer_flag_share", "Transfer eligible", "#4c78a8"),
        ("ae_candidate_share", "AE", "#f58518"),
        ("gmm_nll_q90_share", "GMM NLL q90", "#54a24b"),
        ("gmm_d2_q90_share", "GMM d2 q90", "#e45756"),
        ("physical_global_share", "Physical", "#b279a2"),
    ]:
        axes[1].plot(profile["relative_position"], profile[column], label=label, color=color, alpha=0.85)
    axes[1].set(xlabel="repeat298 relative position", ylabel="Share across occurrences/eligible combos")
    axes[1].legend(ncol=5, fontsize=8)
    for axis in axes:
        axis.axvspan(1, 12, color="#f58518", alpha=0.12)
        axis.axvspan(139, 143, color="#e45756", alpha=0.12)
        axis.grid(alpha=0.2)
    fig.suptitle("Figure 7. repeat298 eligible transfer profile")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "figure7_repeat298_transfer_profile.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    config = read_config()
    ensure_output_dirs()
    source = source_model_cohort(aggregate_source(load_source_raw(config)), config)
    kamp, _quality = load_kamp(config)
    physical = pd.read_csv(Y7 / "physical_row_followup.csv", encoding="utf-8-sig")
    physical = physical.sort_values("excel_row").reset_index(drop=True)
    if not np.array_equal(kamp["excel_row"].to_numpy(), physical["excel_row"].to_numpy()):
        raise AssertionError("KAMP/YSH007 excel_row mismatch")
    thresholds = pd.read_csv(OUT / "source_thresholds.csv", encoding="utf-8-sig")
    gate_summary = json.loads((OUT / "source_gate_summary.json").read_text(encoding="utf-8"))
    if not all(gate_summary[name] for name in ["gate_A", "gate_B", "gate_C"]):
        raise RuntimeError("Source gates A-C did not pass")

    scores, references, evaluated, _ = source_models_and_scores(
        source, kamp, physical, thresholds, gate_summary["combo_status"], config
    )
    absolute = absolute_negative_control(source, kamp, config)
    row_summary = build_row_summary(scores, physical)
    stability = stability_table(scores)
    counts, episodes = count_table(scores, kamp)
    overlap = existing_overlap(scores, kamp, physical)
    profile, repeat_sanity = repeat_profile(scores, config)
    regression_metrics = pd.read_csv(OUT / "source_regression_metrics.csv", encoding="utf-8-sig")
    directions = regression_directions(source, kamp, physical, regression_metrics, config)

    primary = stability.loc[stability["family"].eq("eligible_primary_K1K2")]
    median_jaccard = float(primary["jaccard"].median())
    median_spearman = float(primary["spearman"].median())
    preregistered_primary_combos = 12
    evaluated_primary_combos = int(scores["target_reference"].isin(["K1_global", "K2_unique_raw4"]).groupby(scores["combo"]).any().sum())
    representation_coverage = set(scores.loc[scores["target_reference"].isin(["K1_global", "K2_unique_raw4"]), "representation"]) == {"RZ", "Q"}
    mapping_coverage = set(scores.loc[scores["target_reference"].isin(["K1_global", "K2_unique_raw4"]), "mapping"]) == set(MAPPINGS)
    gate_d = (
        evaluated_primary_combos == preregistered_primary_combos
        and representation_coverage
        and mapping_coverage
        and median_jaccard >= config["gate"]["stability_jaccard_median_min"]
        and median_spearman >= config["gate"]["stability_spearman_median_min"]
    )
    gates = pd.read_csv(OUT / "transfer_gate_decisions.csv", encoding="utf-8-sig")
    gate_rows = pd.DataFrame(
        [
            {
                "gate": "D",
                "scope": "preregistered_primary_12_combos",
                "status": "pass" if gate_d else "fail",
                "metric_summary": f"evaluated={evaluated_primary_combos}/12;median_jaccard={median_jaccard:.6f};median_spearman={median_spearman:.6f};RZ_Q_coverage={representation_coverage};mapping_coverage={mapping_coverage}",
                "consequence": "result_association_allowed" if gate_d else "transfer_not_stable_no_result_association",
            },
            {
                "gate": "E",
                "scope": "primary_consensus_vs_existing_methods",
                "status": "not_evaluated",
                "metric_summary": "primary_consensus_unavailable_incomplete_12_combo_family",
                "consequence": "eligible_combo_overlap_is_diagnostic_only",
            },
        ]
    )
    gates = pd.concat([gates.loc[~gates["gate"].isin(["D", "E"])], gate_rows], ignore_index=True)

    write_csv(scores, "kamp_external_transfer_scores.csv")
    write_csv(row_summary, "kamp_transfer_row_summary.csv")
    write_csv(references, "kamp_normalization_references.csv")
    write_csv(stability, "transfer_stability_pairwise.csv")
    write_csv(counts, "transfer_candidate_counts.csv")
    write_csv(episodes, "transfer_episode_table.csv")
    write_csv(overlap, "transfer_existing_candidate_overlap.csv")
    write_csv(profile, "repeat298_transfer_profile.csv")
    write_csv(repeat_sanity, "repeat298_transfer_sanity.csv")
    write_csv(directions, "kamp_relative_quality_scores.csv")
    write_csv(absolute, "absolute_transfer_negative_control.csv")
    write_csv(gates, "transfer_gate_decisions.csv")
    figures(source, kamp, physical, scores, stability, profile)

    manifest_path = OUT / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "completed_through_gate_D_fail_branch"
    manifest["approved_source_exclusion"] = {"sample_ids": [250, 252], "model_samples": 493}
    manifest["executed_outputs"] = sorted(
        [
            path.name
            for path in OUT.iterdir()
            if path.is_file() and path.name != "input_manifest.json"
        ]
        + [f"figures/{path.name}" for path in FIG.glob("*.png")]
        + [f"models/{path.name}" for path in MODELS.glob("*.joblib")]
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {
        "source_model_samples": len(source),
        "eligible_source_combos": sorted({f"{row.representation}_{row.mapping}" for row in scores.itertuples()}),
        "evaluated_target_combos": len(set(evaluated)),
        "evaluated_primary_combos": evaluated_primary_combos,
        "preregistered_primary_combos": preregistered_primary_combos,
        "primary_pairwise_median_jaccard": median_jaccard,
        "primary_pairwise_median_spearman": median_spearman,
        "representation_coverage_RZ_and_Q": representation_coverage,
        "mapping_coverage_M1_M2_M3": mapping_coverage,
        "gate_D": gate_d,
        "gate_E": "not_evaluated_primary_consensus_unavailable",
        "result_association": "not_run_gate_D_fail" if not gate_d else "allowed_not_implemented",
        "repeat298_all_primary_scores_identical": bool(repeat_sanity["identical_within_tolerance_1e_12"].all()),
        "relative_quality_score_rows": len(directions),
    }
    write_json(payload, "target_transfer_summary.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
