from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
Y11 = HERE.parent / "ysh-011"
SOURCE_OUT = Y11 / "outputs"
OUT = HERE / "outputs"
FIG = OUT / "figures"


def load_runner():
    spec = importlib.util.spec_from_file_location("ysh011_runner", Y11 / "run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.12g")


def main() -> None:
    runner = load_runner()
    config = runner.read_config()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(SOURCE_OUT / "source_anomaly_metrics.csv", encoding="utf-8-sig")
    completeness = pd.read_csv(SOURCE_OUT / "source_oof_completeness.csv", encoding="utf-8-sig")
    bootstrap = pd.read_csv(SOURCE_OUT / "source_group_bootstrap_ci.csv", encoding="utf-8-sig")
    unsupervised = pd.read_csv(SOURCE_OUT / "source_unsupervised_oof_scores.csv", encoding="utf-8-sig")

    core_filter = (
        metrics["branch"].eq("U")
        & metrics["representation"].eq("Q")
        & metrics["family"].isin(["F2", "F3"])
        & metrics["mapping"].isin(["M1_mean", "M2_max"])
        & metrics["model"].isin(["GMM", "IF"])
    )
    core_metrics = metrics.loc[core_filter].copy()
    core_seed42 = core_metrics.loc[
        core_metrics["split_scheme"].eq("primary_label_blind_groupkfold")
        & core_metrics["seed"].eq(42)
    ].copy()
    write_csv(core_seed42, "core_source_metrics_seed42.csv")
    write_csv(core_metrics, "core_source_metrics_all_available_seeds.csv")

    oof_filter = (
        unsupervised["split_scheme"].eq("primary_label_blind_groupkfold")
        & unsupervised["seed"].eq(42)
        & unsupervised["representation"].eq("Q")
        & unsupervised["family"].isin(["F2", "F3"])
        & unsupervised["mapping"].isin(["M1_mean", "M2_max"])
        & unsupervised["model"].isin(["GMM", "IF"])
    )
    core_oof = unsupervised.loc[oof_filter].copy()
    expected = 493 * 2 * 2 * 2
    if len(core_oof) != expected:
        raise RuntimeError(f"Unexpected Core OOF rows: {len(core_oof)} != {expected}")
    write_csv(core_oof, "core_source_oof_seed42.csv")

    statuses = runner.transfer_status_table(metrics, completeness)
    statuses = runner.add_bootstrap_flags(statuses, bootstrap)
    status_filter = (
        statuses["branch"].eq("U")
        & statuses["representation"].eq("Q")
        & statuses["family"].isin(["F2", "F3"])
        & statuses["mapping"].isin(["M1_mean", "M2_max"])
        & statuses["model"].isin(["GMM", "IF"])
    )
    core_status = statuses.loc[status_filter].copy()
    write_csv(core_status, "source_transfer_status_by_category.csv")

    runner.OUT = OUT
    runner.FIG = FIG
    if not (OUT / "target_execution_summary.json").exists():
        runner.run_target_transfer(config)

    required = [
        "core_source_metrics_seed42.csv",
        "core_source_oof_seed42.csv",
        "source_transfer_status_by_category.csv",
        "kamp_external_anomaly_scores.csv",
        "kamp_topk_realized_coverage.csv",
        "transfer_rank_stability.csv",
        "transfer_existing_score_comparison.csv",
        "repeat298_anomaly_profile.csv",
    ]
    missing = [name for name in required if not (OUT / name).exists()]
    target_scores = pd.read_csv(OUT / "kamp_external_anomaly_scores.csv", encoding="utf-8-sig")
    target_sizes = target_scores.groupby(
        ["spec_id", "category_direction", "target_reference"]
    )["excel_row"].nunique()
    verification = {
        "required_files_present": not missing,
        "missing_files": missing,
        "core_oof_rows_expected": len(core_oof) == expected,
        "core_oof_sample_count_each": bool(
            core_oof.groupby(["family", "mapping", "model"])["sample_id"].nunique().eq(493).all()
        ),
        "target_score_rows_11939_each": bool(target_sizes.eq(11939).all()),
        "result_sheet_used": False,
    }
    verification["passed"] = bool(
        verification["required_files_present"]
        and verification["core_oof_rows_expected"]
        and verification["core_oof_sample_count_each"]
        and verification["target_score_rows_11939_each"]
        and not verification["result_sheet_used"]
    )
    (OUT / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not verification["passed"]:
        raise RuntimeError(f"Core verification failed: {verification}")

    summary = {
        "source_training_reused": True,
        "new_source_model_training": False,
        "primary_seed": 42,
        "core_oof_rows": len(core_oof),
        "core_metric_rows_seed42": len(core_seed42),
        "core_status_counts": core_status["transfer_status"].value_counts().to_dict(),
        "selected_for_kamp_category_directions": int(
            (
                core_status["transfer_status"].isin(["SUPPORTED_FOR_TRANSFER", "SENSITIVITY_ONLY"])
                & core_status["primary_all_positive"].astype(bool)
            ).sum()
        ),
        "result_sheet_used": False,
    }
    (OUT / "core_recovery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
