from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "outputs"
RAW_PATH = ROOT / "data" / "Welding_Data_Set_01.xlsx"
RAW_SHA256 = "d514d6aaa121630c04d7d51c97a56025e78e2c86868813f7722ba5db922c1f33"
FEATURES = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
METHODS = ["gmm_nll_q90", "gmm_d2_q90", "conditional_z_2_5",
           "physical_global_candidate"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def runs(mask: np.ndarray):
    positions = np.flatnonzero(mask) + 1
    if not len(positions):
        return []
    cuts = np.flatnonzero(np.diff(positions) > 1) + 1
    return [(int(x[0]), int(x[-1]), int(len(x))) for x in np.split(positions, cuts)]


def main() -> None:
    check(sha256(RAW_PATH) == RAW_SHA256, "raw digest")
    raw = pd.read_excel(RAW_PATH, sheet_name="Raw data")
    result = pd.read_excel(RAW_PATH, sheet_name="result")
    mapping = pd.read_csv(OUT / "repeat298_mapping.csv")
    audit = pd.read_csv(OUT / "repeat298_occurrence_audit.csv")
    parents = pd.read_csv(OUT / "repeat298_parent_blocks.csv")
    decomposition = pd.read_csv(OUT / "repeat298_gmm_decomposition.csv")
    profile = pd.read_csv(OUT / "repeat298_relative_profile.csv")
    physical = pd.read_csv(OUT / "physical_screening_motif_sensitivity.csv")
    episodes = pd.read_csv(OUT / "repeat298_episodes.csv")
    comparison = pd.read_csv(OUT / "result_exploratory_comparison.csv")
    ae_audit = pd.read_csv(OUT / "ae_input_audit.csv")

    check(len(raw) == 11939, "raw rows")
    check(len(mapping) == 1192 and mapping.excel_row.is_unique, "mapping rows")
    check(len(audit) == 4 and (audit.rows == 298).all(), "four occurrences")
    check(mapping.groupby("relative_position").size().eq(4).all(), "four per position")
    check(mapping.groupby("occurrence_id").size().eq(298).all(), "298 per occurrence")
    first = None
    for _, part in mapping.sort_values("relative_position").groupby("occurrence_id"):
        values = part.sort_values("relative_position")[FEATURES].to_numpy()
        first = values if first is None else first
        check(np.array_equal(first, values), "Raw4 exact equality")
    check(len(parents) == 6, "parent candidate blocks")
    counts = decomposition.set_index("group").rows.astype(int)
    check(counts["A"] + counts["B-all"] == len(raw), "A/B partition")
    check(counts["B-repeat298-core"] + counts["B-nonrepeat"] == counts["B-all"],
          "core/nonrepeat partition")
    check(counts["B-repeat298-core"] + counts["parent-repeat-other"] +
          counts["outside-parent"] == counts["B-all"], "parent partition")
    check(counts["outside-parent"] == 16, "outside parent count")
    check(len(profile) == 298 and np.array_equal(
        profile.relative_position.to_numpy(), np.arange(1, 299)), "profile positions")
    grouped = mapping.groupby("relative_position")
    for method in METHODS:
        expected = grouped[method].sum().astype(int).to_numpy()
        check(np.array_equal(expected, profile[f"{method}_support"].to_numpy()),
              f"{method} support")
        primary = expected >= 3
        recorded = episodes[(episodes.scope == "motif") &
                            (episodes.method == method) &
                            (episodes.support_rule == "ge3_primary")]
        actual = list(recorded[["start_position", "end_position", "length"]]
                      .itertuples(index=False, name=None))
        check(actual == runs(primary), f"{method} primary episodes")
    check(len(physical) == 298, "physical comparison positions")
    existing = physical.physical_global_candidate_positive_ge3.astype(bool)
    sensitivity = physical.motif_relative_physical_sensitivity.astype(bool)
    expected_labels = np.select(
        [existing & sensitivity, existing & ~sensitivity, ~existing & sensitivity],
        ["both", "existing_only", "sensitivity_only"], default="neither")
    check(np.array_equal(expected_labels, physical.comparison.to_numpy()),
          "physical comparison labels")
    check(len(comparison) == 4, "Result occurrence rows")
    march27 = comparison.loc[comparison.date.eq("2020-03-27")].iloc[0]
    march31 = comparison.loc[comparison.date.eq("2020-03-31")].iloc[0]
    check(pd.isna(march27[["result_type1", "result_type2", "result_type3"]]).all(),
          "03-27 Result missing")
    check(pd.isna(march31.result_type3), "03-31 type3 missing")
    check(ae_audit.status.iloc[0] == "not_run_missing_input" and
          int(ae_audit.matching_exports.iloc[0]) == 0, "AE missing state")

    key_outputs = ["repeat298_mapping.csv", "repeat298_gmm_decomposition.csv",
                   "repeat298_relative_profile.csv", "repeat298_episodes.csv",
                   "result_exploratory_comparison.csv", "summary.json"]
    report = {
        "status": "passed_partial_missing_ae", "raw_sha256": RAW_SHA256,
        "raw_rows": len(raw), "repeat_rows": len(mapping),
        "relative_positions": len(profile), "parent_candidate_blocks": len(parents),
        "parent_partition": {name: int(counts[name]) for name in
                             ("B-repeat298-core", "parent-repeat-other", "outside-parent")},
        "support_and_episode_recalculated": True,
        "result_missingness_preserved": True,
        "ae_status": "not_run_missing_input",
        "key_output_sha256": {name: sha256(OUT / name) for name in key_outputs},
    }
    (OUT / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = OUT / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["verification_status"] = report["status"]
    manifest["code_sha256"]["verify.py"] = sha256(HERE / "verify.py")
    manifest["verification_sha256"] = sha256(OUT / "verification.json")
    result_path = HERE / "실험결과.md"
    if result_path.exists():
        manifest["result_sha256"] = sha256(result_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
