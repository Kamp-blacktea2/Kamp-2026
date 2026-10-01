from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from common import (
    FIG,
    MODELS,
    OUT,
    KAMP_PATH,
    SOURCE_PATH,
    json_default,
    read_config,
    sha256,
    write_json,
)


def main() -> None:
    config = read_config()
    audit = json.loads((OUT / "source_data_audit.json").read_text(encoding="utf-8"))
    source = pd.read_csv(OUT / "source_weld_level_495.csv", encoding="utf-8-sig")
    consistency = pd.read_csv(OUT / "source_sample_consistency_audit.csv", encoding="utf-8-sig")
    condition = pd.read_csv(OUT / "source_condition_quality_table.csv", encoding="utf-8-sig")
    overlap = pd.read_csv(OUT / "domain_overlap_summary.csv", encoding="utf-8-sig")
    absolute = pd.read_csv(OUT / "absolute_transfer_negative_control.csv", encoding="utf-8-sig")
    gates = pd.read_csv(OUT / "transfer_gate_decisions.csv", encoding="utf-8-sig")

    checks = {
        "source_hash_matches": sha256(SOURCE_PATH) == config["source_sha256"],
        "kamp_hash_matches": sha256(KAMP_PATH) == config["kamp_sha256"],
        "source_weld_rows_495": len(source) == 495,
        "source_sample_ids_unique": source["sample_id"].is_unique,
        "source_consistency_rows_495": len(consistency) == 495,
        "condition_groups_19": len(condition) == 19,
        "zero_valid_ids_exact": source.loc[
            source["current_valid_rows"].eq(0) | source["force_valid_rows"].eq(0), "sample_id"
        ].astype(int).tolist() == [250, 252],
        "audit_status_stage0_fail": audit["status"] == "stage0_fail_source_sensor_valid_count_zero",
        "absolute_rows_11939": len(absolute) == 11939,
        "absolute_score_not_run": absolute["raw_direct_model_score_status"].eq("not_run_stage0_fail").all(),
        "pressure_overlap_712": int(absolute["in_source_pressure_like_support"].sum()) == 712,
        "current_overlap_0": int(absolute["in_source_current_support"].sum()) == 0,
        "time_overlap_0": int(absolute["in_source_welding_time_support"].sum()) == 0,
        "joint_overlap_0": int(absolute["in_source_joint_support"].sum()) == 0,
        "gates_all_not_evaluated": gates["status"].eq("not_evaluated").all(),
        "gates_reason_exact": gates["metric_summary"].eq(
            "stage0_fail_source_sensor_valid_count_zero"
        ).all(),
        "no_fitted_models": not any(MODELS.glob("*")),
        "no_source_model_metrics": not (OUT / "source_model_metrics.csv").exists(),
        "no_kamp_similarity_scores": not (OUT / "kamp_external_transfer_scores.csv").exists(),
        "figure1_exists": (FIG / "figure1_absolute_domain_mismatch_negative_control.png").exists(),
    }
    image = plt.imread(FIG / "figure1_absolute_domain_mismatch_negative_control.png")
    checks["figure1_nonempty"] = image.shape[0] > 100 and image.shape[1] > 100
    checks["domain_overlap_role_negative_control"] = overlap["role"].eq(
        "negative_control_domain_mismatch"
    ).all()

    payload = {
        "experiment": "YSH-010",
        "verification_status": "pass" if all(checks.values()) else "fail",
        "experiment_status": "stopped_at_stage0_by_preregistered_rule",
        "checks": checks,
        "verified_counts": {
            "source_raw_rows": audit["source_raw_rows"],
            "source_weld_rows": len(source),
            "source_valid_sensor_samples": audit["source_samples_with_valid_current_and_force"],
            "zero_valid_sensor_sample_ids": audit["zero_valid_sensor_sample_ids"],
            "kamp_rows": len(absolute),
            "condition_groups": len(condition),
        },
    }
    write_json(payload, "verification.json")
    if payload["verification_status"] != "pass":
        failed = [name for name, value in checks.items() if not value]
        raise AssertionError(f"Verification failed: {failed}")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
