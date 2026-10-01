from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "outputs"
YSH005 = HERE.parent / "ysh-005" / "outputs"
RAW = ROOT / "data" / "Welding_Data_Set_01.xlsx"
RAW_SHA256 = "d514d6aaa121630c04d7d51c97a56025e78e2c86868813f7722ba5db922c1f33"
FEATURES = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
SHORT = ["F", "I", "V", "t"]
METHODS = ["gmm_nll_q90", "gmm_d2_q90", "conditional_z_2_5",
           "physical_global_candidate"]
RULES = [("ge1", 1), ("ge3_primary", 3), ("eq4", 4)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def rule(values: np.ndarray, name: str, threshold: int) -> np.ndarray:
    return values == 4 if name == "eq4" else values >= threshold


def runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    positions = np.flatnonzero(mask) + 1
    if not len(positions):
        return []
    cuts = np.flatnonzero(np.diff(positions) > 1) + 1
    return [(int(x[0]), int(x[-1]), int(len(x))) for x in np.split(positions, cuts)]


def main() -> None:
    check(sha256(RAW) == RAW_SHA256, "raw digest")
    raw = pd.read_excel(RAW, sheet_name="Raw data")
    result = pd.read_excel(RAW, sheet_name="result")
    raw["excel_row"] = np.arange(len(raw)) + 2
    raw["date"] = pd.to_datetime(raw["working time"]).dt.strftime("%Y-%m-%d")
    result["date"] = pd.to_datetime(result["working time"]).dt.strftime("%Y-%m-%d")
    train = raw.date.le("2020-03-31").to_numpy()

    artifact = np.load(YSH005 / "models" / "ae_42_reconstruction.npz")
    scaled, reconstructed = artifact["input"], artifact["output"]
    feature_error = ((reconstructed - scaled) ** 2).astype(float)
    total_error = ((reconstructed - scaled) ** 2).mean(axis=1).astype(float)
    threshold = total_error[train].mean() + 8 * total_error[train].std(ddof=0)
    candidates = total_error >= threshold

    rows = pd.read_csv(OUT / "ae_row_scores.csv")
    required = ["excel_row", "date", "ae_total_error", "ae_candidate"] + [
        f"ae_error_{name}" for name in SHORT]
    check(set(required).issubset(rows.columns), "required row columns")
    check(len(rows) == len(raw) == 11939 and rows.excel_row.is_unique, "row identity")
    check(np.array_equal(rows.excel_row, raw.excel_row), "row order")
    check(np.allclose(rows.ae_total_error, total_error, atol=1e-14, rtol=0),
          "total AE error")
    check(np.array_equal(rows.ae_candidate.to_numpy(bool), candidates), "AE candidates")
    for index, name in enumerate(SHORT):
        check(np.allclose(rows[f"ae_error_{name}"], feature_error[:, index],
                          atol=1e-14, rtol=0), f"feature error {name}")

    metadata = json.loads((OUT / "ae_model_metadata.json").read_text(encoding="utf-8"))
    check(np.isclose(metadata["threshold"], threshold, atol=1e-14, rtol=0),
          "threshold")
    check(metadata["train_rows"] == 8470 and metadata["train_candidates"] == 42 and
          metadata["test_candidates"] == 14, "YSH-005 split and flags")

    joined = pd.read_csv(OUT / "ae_gmm_agreement_rows.csv")
    profile = pd.read_csv(OUT / "ae_gmm_relative_profile.csv")
    summary = pd.read_csv(OUT / "ae_gmm_agreement_summary.csv")
    check(len(joined) == 1192 and len(profile) == 298, "repeat dimensions")
    check(joined.groupby("relative_position").size().eq(4).all(), "four occurrences")
    error_fields = ["ae_total_error"] + [f"ae_error_{name}" for name in SHORT]
    ranges = joined.groupby("relative_position")[error_fields].agg(np.ptp)
    check((ranges.to_numpy() == 0).all(), "identical Raw4 AE errors")
    check(joined.groupby("relative_position").ae_candidate.nunique().eq(1).all(),
          "identical Raw4 candidate")

    grouped = joined.groupby("relative_position", sort=True)
    ae_support = grouped.ae_candidate.sum().astype(int).to_numpy()
    check(np.array_equal(profile.ae_candidate_support, ae_support), "AE support")
    for rule_name, threshold_count in RULES:
        ae_mask = rule(ae_support, rule_name, threshold_count)
        for method in METHODS:
            gmm_support = grouped[method].sum().astype(int).to_numpy()
            gmm_mask = rule(gmm_support, rule_name, threshold_count)
            recorded = summary[(summary.support_rule == rule_name) &
                               (summary.gmm_method == method)].iloc[0]
            tp = int((ae_mask & gmm_mask).sum())
            fp = int((~ae_mask & gmm_mask).sum())
            fn = int((ae_mask & ~gmm_mask).sum())
            tn = int((~ae_mask & ~gmm_mask).sum())
            check((recorded[["tp", "fp", "fn", "tn"]].to_numpy(int) ==
                   [tp, fp, fn, tn]).all(), f"agreement {rule_name} {method}")
            check(np.isclose(recorded.cohen_kappa,
                             cohen_kappa_score(ae_mask, gmm_mask), atol=1e-14),
                  f"kappa {rule_name} {method}")

    episodes = pd.read_csv(OUT / "ae_gmm_episodes.csv")
    for rule_name, threshold_count in RULES:
        ae_mask = rule(ae_support, rule_name, threshold_count)
        recorded = episodes[(episodes.scope == "motif") &
                            (episodes.episode_type == "AE") &
                            (episodes.support_rule == rule_name)]
        check(list(recorded[["start_position", "end_position", "length"]]
                   .itertuples(index=False, name=None)) == runs(ae_mask),
              f"AE motif episodes {rule_name}")
        for method in METHODS:
            gmm_support = grouped[method].sum().astype(int).to_numpy()
            gmm_mask = rule(gmm_support, rule_name, threshold_count)
            for episode_type, mask in [("GMM", gmm_mask),
                                       ("consensus", ae_mask & gmm_mask)]:
                recorded = episodes[(episodes.scope == "motif") &
                                    (episodes.episode_type == episode_type) &
                                    (episodes.gmm_method == method) &
                                    (episodes.support_rule == rule_name)]
                actual = list(recorded[["start_position", "end_position", "length"]]
                              .itertuples(index=False, name=None))
                check(actual == runs(mask),
                      f"{episode_type} episodes {rule_name} {method}")

    result_rows = pd.read_csv(OUT / "ae_result_exploratory_comparison.csv")
    check(len(result_rows) == 4 and result_rows.ae_candidate_count.eq(12).all(),
          "Result occurrence comparison")
    check(result_rows.ae_total_error_mean.nunique() == 1 and
          result_rows.ae_total_error_median.nunique() == 1,
          "occurrence AE equality")
    quality = result.pivot_table(index="date", columns="defect type", values="defect",
                                 aggfunc="first")
    for row in result_rows.itertuples():
        for defect_type in (1, 2, 3):
            expected = (quality.at[row.date, defect_type]
                        if row.date in quality.index and defect_type in quality else np.nan)
            actual = getattr(row, f"result_type{defect_type}")
            check((pd.isna(expected) and pd.isna(actual)) or expected == actual,
                  f"Result value {row.date} type {defect_type}")

    old_verification = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    for name, expected in old_verification["key_output_sha256"].items():
        check(sha256(OUT / name) == expected, f"GMM output changed: {name}")

    report = {
        "status": "passed", "raw_sha256": RAW_SHA256,
        "rows": len(rows), "threshold": float(threshold),
        "all_candidates": int(candidates.sum()),
        "repeat_rows": len(joined), "repeat_positions": len(profile),
        "repeat_candidate_rows": int(joined.ae_candidate.sum()),
        "repeat_candidate_positions_primary": int((ae_support >= 3).sum()),
        "identical_raw4_errors_exact": True,
        "agreement_recalculated": True, "episodes_recalculated": True,
        "result_rechecked_last": True, "gmm_outputs_unchanged": True,
    }
    (OUT / "ae_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = OUT / "ae_followup_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["verification_status"] = "passed"
    manifest["code_sha256"]["verify_ae.py"] = sha256(HERE / "verify_ae.py")
    manifest["verification_sha256"] = sha256(OUT / "ae_verification.json")
    for name, path in [("plan_sha256", HERE / "실험계획.md"),
                       ("result_sha256", HERE / "실험결과.md")]:
        if path.exists():
            manifest[name] = sha256(path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
