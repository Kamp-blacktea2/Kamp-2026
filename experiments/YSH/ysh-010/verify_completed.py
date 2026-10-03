from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, MODELS, OUT, KAMP_PATH, SOURCE_PATH, json_default, read_config, sha256, write_json


def main() -> None:
    config = read_config()
    audit = json.loads((OUT / "source_data_audit.json").read_text(encoding="utf-8"))
    source = pd.read_csv(OUT / "source_weld_level_495.csv", encoding="utf-8-sig")
    cohort = pd.read_csv(OUT / "source_model_cohort_493.csv", encoding="utf-8-sig")
    consistency = pd.read_csv(OUT / "source_sample_consistency_audit.csv", encoding="utf-8-sig")
    condition = pd.read_csv(OUT / "source_condition_quality_table.csv", encoding="utf-8-sig")
    absolute = pd.read_csv(OUT / "absolute_transfer_negative_control.csv", encoding="utf-8-sig")
    source_metrics = pd.read_csv(OUT / "source_model_metrics.csv", encoding="utf-8-sig")
    regression_metrics = pd.read_csv(OUT / "source_regression_metrics.csv", encoding="utf-8-sig")
    thresholds = pd.read_csv(OUT / "source_thresholds.csv", encoding="utf-8-sig")
    gates = pd.read_csv(OUT / "transfer_gate_decisions.csv", encoding="utf-8-sig")
    scores = pd.read_csv(OUT / "kamp_external_transfer_scores.csv", encoding="utf-8-sig")
    row_summary = pd.read_csv(OUT / "kamp_transfer_row_summary.csv", encoding="utf-8-sig")
    stability = pd.read_csv(OUT / "transfer_stability_pairwise.csv", encoding="utf-8-sig")
    repeat_profile = pd.read_csv(OUT / "repeat298_transfer_profile.csv", encoding="utf-8-sig")
    repeat_sanity = pd.read_csv(OUT / "repeat298_transfer_sanity.csv", encoding="utf-8-sig")
    directions = pd.read_csv(OUT / "kamp_relative_quality_scores.csv", encoding="utf-8-sig")
    target_summary = json.loads((OUT / "target_transfer_summary.json").read_text(encoding="utf-8"))

    excluded = sorted(config["excluded_sample_ids"])
    zero_valid = sorted(
        source.loc[
            source["current_valid_rows"].eq(0) | source["force_valid_rows"].eq(0), "sample_id"
        ].astype(int).tolist()
    )
    score_combos = sorted(scores["combo"].unique().tolist())
    expected_combos = sorted(
        f"Q_{reference}_{mapping}"
        for reference in ["K1_global", "K2_unique_raw4", "K3_gmm_relative"]
        for mapping in ["M1_mean", "M2_max"]
    )
    primary_stability = stability.loc[stability["family"].eq("eligible_primary_K1K2")]
    model_names = sorted(path.name for path in MODELS.glob("source_logistic_*.joblib"))

    checks = {
        "source_hash_matches": sha256(SOURCE_PATH) == config["source_sha256"],
        "kamp_hash_matches": sha256(KAMP_PATH) == config["kamp_sha256"],
        "source_audit_rows_495": len(source) == 495 and len(consistency) == 495,
        "source_audit_ids_unique": source["sample_id"].is_unique,
        "zero_valid_ids_equal_approved_exclusions": zero_valid == excluded == [250, 252],
        "audit_status_493_approved": audit["status"] == "stage0_pass_with_approved_493_sample_exclusion",
        "cohort_rows_493": len(cohort) == 493 and cohort["sample_id"].is_unique,
        "cohort_excludes_only_250_252": set(source["sample_id"]) - set(cohort["sample_id"]) == {250, 252},
        "cohort_class_counts": int(cohort["abnormal"].eq(0).sum()) == 441 and int(cohort["abnormal"].eq(1).sum()) == 52,
        "cohort_common_features_complete": not cohort[["pressure_bar", "welding_time_ms", "current_min_ka", "current_mean_ka", "current_max_ka"]].isna().any().any(),
        "condition_groups_19": len(condition) == 19,
        "absolute_rows_11939": len(absolute) == 11939,
        "absolute_joint_support_zero": int(absolute["in_source_joint_support"].sum()) == 0,
        "absolute_scores_diagnostic_only": absolute["raw_direct_model_score_status"].eq("diagnostic_only_all_rows_outside_joint_support").all(),
        "source_metrics_present": len(source_metrics) > 0 and len(regression_metrics) > 0,
        "RZ_group_invalid": source_metrics.loc[source_metrics["model"].eq("logistic") & source_metrics["representation"].eq("RZ") & source_metrics["split"].eq("group"), "roc_auc"].isna().all(),
        "Q_gateB_exact_M1_M2": set(thresholds.loc[thresholds["gate_B_pass"].astype(bool), "mapping"]) == {"M1_mean", "M2_max"},
        "scores_rows_6x11939": len(scores) == 6 * 11939,
        "scores_combo_schema_exact": score_combos == expected_combos,
        "scores_excel_rows_complete_each_combo": scores.groupby("combo")["excel_row"].nunique().eq(11939).all(),
        "row_summary_rows_11939": len(row_summary) == 11939 and row_summary["excel_row"].is_unique,
        "primary_consensus_not_claimed": not row_summary["transfer_consensus_candidate"].astype(bool).any(),
        "primary_stability_six_pairs": len(primary_stability) == 6,
        "gateD_failed_for_incomplete_family": target_summary["gate_D"] is False and target_summary["evaluated_primary_combos"] == 4,
        "gateE_not_evaluated": target_summary["gate_E"] == "not_evaluated_primary_consensus_unavailable",
        "gate_table_D_fail": gates.loc[gates["gate"].eq("D"), "status"].tolist() == ["fail"],
        "gate_table_E_not_evaluated": gates.loc[gates["gate"].eq("E"), "status"].tolist() == ["not_evaluated"],
        "repeat_profile_298": len(repeat_profile) == 298 and repeat_profile["relative_position"].tolist() == list(range(1, 299)),
        "repeat_primary_four_combos": len(repeat_sanity) == 4,
        "repeat_scores_identical": repeat_sanity["identical_within_tolerance_1e_12"].astype(bool).all() and np.isclose(repeat_sanity["max_occurrence_score_range"], 0).all(),
        "relative_quality_rows_18x11939": len(directions) == 18 * 11939,
        "source_logistic_models_exact": model_names == ["source_logistic_Q_M1_mean.joblib", "source_logistic_Q_M2_max.joblib"],
        "result_association_not_run": target_summary["result_association"] == "not_run_gate_D_fail" and not (OUT / "transfer_result_daily_comparison.csv").exists(),
        "figure8_not_created": not (FIG / "figure8_result_exploratory_comparison.png").exists(),
    }
    figure_names = [
        "figure1_absolute_domain_mismatch_negative_control.png",
        "figure2_source_quality_map.png",
        "figure3_source_model_validation.png",
        "figure4_source_quality_outputs.png",
        "figure5_domain_normalized_feature_comparison.png",
        "figure6_transfer_score_stability.png",
        "figure7_repeat298_transfer_profile.png",
    ]
    for number, name in enumerate(figure_names, start=1):
        path = FIG / name
        checks[f"figure{number}_exists"] = path.exists()
        if path.exists():
            image = plt.imread(path)
            checks[f"figure{number}_nonempty"] = image.shape[0] > 100 and image.shape[1] > 100

    payload = {
        "experiment": "YSH-010",
        "verification_status": "pass" if all(checks.values()) else "fail",
        "experiment_status": "completed_through_gate_D_fail_branch",
        "checks": checks,
        "verified_counts": {
            "source_raw_rows": audit["source_raw_rows"],
            "source_audit_samples": len(source),
            "source_model_samples": len(cohort),
            "kamp_rows": len(row_summary),
            "target_score_rows": len(scores),
            "target_combos": len(score_combos),
            "relative_quality_score_rows": len(directions),
            "repeat_positions": len(repeat_profile),
        },
    }
    write_json(payload, "verification.json")
    if payload["verification_status"] != "pass":
        failed = [name for name, value in checks.items() if not value]
        raise AssertionError(f"Verification failed: {failed}")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
