from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import OUT, json_default, write_json


def finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def main() -> None:
    metrics = pd.read_csv(OUT / "source_model_metrics.csv", encoding="utf-8-sig")
    regression = pd.read_csv(OUT / "source_regression_metrics.csv", encoding="utf-8-sig")
    thresholds = pd.read_csv(OUT / "source_thresholds.csv", encoding="utf-8-sig")
    counts = pd.read_csv(OUT / "transfer_candidate_counts.csv", encoding="utf-8-sig")
    stability = pd.read_csv(OUT / "transfer_stability_pairwise.csv", encoding="utf-8-sig")
    overlap = pd.read_csv(OUT / "transfer_existing_candidate_overlap.csv", encoding="utf-8-sig")
    repeat = pd.read_csv(OUT / "repeat298_transfer_profile.csv", encoding="utf-8-sig")
    repeat_sanity = pd.read_csv(OUT / "repeat298_transfer_sanity.csv", encoding="utf-8-sig")
    directions = pd.read_csv(OUT / "kamp_relative_quality_scores.csv", encoding="utf-8-sig")

    selected = metrics.loc[
        metrics["model"].eq("logistic") & metrics["representation"].isin(["RZ", "Q"])
    ]
    source_rows = []
    for row in selected.itertuples(index=False):
        source_rows.append(
            {
                "representation": row.representation,
                "mapping": row.mapping,
                "split": row.split,
                "roc_auc": finite_or_none(row.roc_auc),
                "pr_auc": finite_or_none(row.pr_auc),
                "balanced_accuracy_at_0": finite_or_none(row.balanced_accuracy),
                "invalid_folds": int(row.invalid_folds),
            }
        )

    gate = {
        "r2": 0.0,
        "spearman": 0.20,
        "dummy_mae_improvement": 0.05,
    }
    ridge_group = regression.loc[regression["model"].eq("ridge") & regression["split"].eq("group")].copy()
    ridge_group["passed"] = (
        ridge_group["r2"].gt(gate["r2"])
        & ridge_group["spearman"].gt(gate["spearman"])
        & ridge_group["dummy_mae_improvement"].ge(gate["dummy_mae_improvement"])
    )
    regression_rows = ridge_group.loc[ridge_group["passed"]].to_dict("records")
    for row in regression_rows:
        for key, value in list(row.items()):
            if isinstance(value, float) and not np.isfinite(value):
                row[key] = None

    primary = stability.loc[stability["family"].eq("eligible_primary_K1K2")]
    payload = {
        "source_logistic": source_rows,
        "source_thresholds": thresholds.loc[
            thresholds["representation"].eq("Q"),
            ["representation", "mapping", "threshold", "balanced_accuracy", "specificity", "recall", "gate_A_pass", "gate_B_pass"],
        ].to_dict("records"),
        "regression_group_ridge_pass_count": int(ridge_group["passed"].sum()),
        "regression_group_ridge_passes": regression_rows,
        "transfer_counts": counts.to_dict("records"),
        "eligible_primary_pairwise": {
            "pairs": len(primary),
            "median_jaccard": finite_or_none(primary["jaccard"].median()),
            "min_jaccard": finite_or_none(primary["jaccard"].min()),
            "max_jaccard": finite_or_none(primary["jaccard"].max()),
            "median_spearman": finite_or_none(primary["spearman"].median()),
            "min_spearman": finite_or_none(primary["spearman"].min()),
            "max_spearman": finite_or_none(primary["spearman"].max()),
        },
        "eligible_overlap_ranges": [
            {
                "method": method,
                "row_intersection_min": int(part["row_intersection"].min()),
                "row_intersection_max": int(part["row_intersection"].max()),
                "row_jaccard_min": finite_or_none(part["row_jaccard"].min()),
                "row_jaccard_max": finite_or_none(part["row_jaccard"].max()),
                "unique_raw4_jaccard_min": finite_or_none(part["unique_raw4_jaccard"].min()),
                "unique_raw4_jaccard_max": finite_or_none(part["unique_raw4_jaccard"].max()),
            }
            for method, part in overlap.groupby("method", sort=True)
        ],
        "repeat298": {
            "relative_positions": len(repeat),
            "positions_any_eligible_flag": int(repeat["transfer_flag_share"].gt(0).sum()),
            "positions_majority_eligible_flag": int(repeat["transfer_flag_share"].ge(0.5).sum()),
            "positions_all_eligible_flag": int(repeat["transfer_flag_share"].eq(1).sum()),
            "primary_max_occurrence_score_range": finite_or_none(repeat_sanity["max_occurrence_score_range"].max()),
            "primary_all_identical": bool(repeat_sanity["identical_within_tolerance_1e_12"].all()),
        },
        "relative_quality_scores": {
            "rows": len(directions),
            "targets": sorted(directions["target"].unique().tolist()),
            "combinations": int(directions[["target", "representation", "target_reference", "mapping"]].drop_duplicates().shape[0]),
        },
    }
    write_json(payload, "report_metrics.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
