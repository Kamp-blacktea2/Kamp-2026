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
SHORT = ["F", "I", "V", "t"]
OCCURRENCES = [
    ("occ_20200325", "2020-03-25", 1879, 2176),
    ("occ_20200327", "2020-03-27", 4693, 4990),
    ("occ_20200331", "2020-03-31", 7803, 8100),
    ("occ_20200403", "2020-04-03", 10879, 11176),
]
B_DATES = {row[1] for row in OCCURRENCES}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, float_format="%.12g")


def runs(mask: np.ndarray) -> list[tuple[int, int, int]]:
    positions = np.flatnonzero(mask) + 1
    if not len(positions):
        return []
    cuts = np.flatnonzero(np.diff(positions) > 1) + 1
    return [(int(part[0]), int(part[-1]), int(len(part)))
            for part in np.split(positions, cuts)]


def load_inputs():
    if sha256(RAW_PATH) != RAW_SHA256:
        raise ValueError("Raw SHA-256 mismatch")
    raw = pd.read_excel(RAW_PATH, sheet_name="Raw data")
    result = pd.read_excel(RAW_PATH, sheet_name="result")
    if len(raw) != 11939 or raw[FEATURES].isna().any().any():
        raise ValueError("Raw structure differs from the frozen plan")
    raw = raw.copy()
    raw["excel_row"] = np.arange(len(raw)) + 2
    raw["date"] = pd.to_datetime(raw["working time"]).dt.strftime("%Y-%m-%d")
    raw["regime"] = np.where(raw["date"].isin(B_DATES), "B", "A")
    result = result.copy()
    result["date"] = pd.to_datetime(result["working time"]).dt.strftime("%Y-%m-%d")
    base = HERE.parent / "ysh-007" / "outputs"
    lodo = pd.read_csv(base / "lodo_row_scores.csv")
    physical = pd.read_csv(base / "physical_row_followup.csv")
    in_sample = pd.read_csv(base / "in_sample_scores.csv")
    blocks = pd.read_csv(HERE.parent / "ysh-001" / "outputs" /
                         "exact_repeated_blocks.csv")
    for frame, label in ((lodo, "lodo"), (physical, "physical"),
                         (in_sample, "in_sample")):
        if len(frame) != len(raw) or not frame["excel_row"].is_unique:
            raise ValueError(f"{label} row coverage mismatch")
    return raw, result, lodo, physical, in_sample, blocks


def find_ae_exports() -> list[dict]:
    required = {"excel_row", "date", "ae_total_error", "ae_candidate"}
    found: list[dict] = []
    for suffix in ("*.csv", "*.parquet"):
        for path in (ROOT / "experiments").rglob(suffix):
            if HERE in path.parents:
                continue
            try:
                columns = set(pd.read_csv(path, nrows=0).columns) if path.suffix == ".csv" \
                    else set(pd.read_parquet(path).columns)
            except Exception:
                continue
            if required.issubset(columns):
                found.append({"path": str(path.relative_to(ROOT)),
                              "columns": sorted(columns)})
    return found


def repeat_and_parent_audit(raw: pd.DataFrame, blocks: pd.DataFrame):
    core = np.zeros(len(raw), dtype=bool)
    occurrence = np.full(len(raw), "", dtype=object)
    relative = np.zeros(len(raw), dtype=int)
    audit_rows, reference = [], None
    for occurrence_id, date, start, end in OCCURRENCES:
        part = raw.loc[raw["excel_row"].between(start, end), FEATURES]
        values = part.to_numpy()
        identical = reference is None or np.array_equal(reference, values)
        if reference is None:
            reference = values.copy()
        rows = raw["excel_row"].between(start, end).to_numpy()
        core |= rows
        occurrence[rows] = occurrence_id
        relative[rows] = np.arange(1, rows.sum() + 1)
        digest = hashlib.sha256(part.to_csv(index=False).encode("utf-8")).hexdigest()
        audit_rows.append({
            "occurrence_id": occurrence_id, "date": date,
            "excel_start": start, "excel_end": end, "rows": len(part),
            "idx_start": int(raw.loc[raw.excel_row.eq(start), "idx"].iloc[0]),
            "idx_end": int(raw.loc[raw.excel_row.eq(end), "idx"].iloc[0]),
            "date_match": bool((raw.loc[rows, "date"] == date).all()),
            "raw4_identical_to_first": bool(identical), "raw4_sha256": digest,
        })
    if core.sum() != 1192 or not all(row["raw4_identical_to_first"] for row in audit_rows):
        raise AssertionError("repeat298 mapping or equality failed")

    selected = np.zeros(len(blocks), dtype=bool)
    contains: list[list[str]] = [[] for _ in range(len(blocks))]
    for i, block in blocks.iterrows():
        for occurrence_id, _, start, end in OCCURRENCES:
            if ((block.a_start <= start <= end <= block.a_end) or
                    (block.b_start <= start <= end <= block.b_end)):
                selected[i] = True
                contains[i].append(occurrence_id)
    parent = blocks.loc[selected].copy()
    parent["source_row"] = parent.index + 2
    parent["contains_occurrences"] = ["|".join(contains[i]) for i in parent.index]
    parent_union = np.zeros(len(raw), dtype=bool)
    for block in parent.itertuples(index=False):
        for start, end in ((block.a_start, block.a_end), (block.b_start, block.b_end)):
            parent_union |= raw["excel_row"].between(start, end).to_numpy()
    parent_union &= raw["regime"].eq("B").to_numpy()
    if not np.all(parent_union[core]):
        raise AssertionError("repeat298 core is not contained in parent union")
    parent_other = parent_union & ~core
    outside_parent = raw["regime"].eq("B").to_numpy() & ~parent_union
    for row in audit_rows:
        row["parent_candidate_count"] = int(parent["contains_occurrences"].str.contains(
            row["occurrence_id"], regex=False).sum())
    return (core, occurrence, relative, parent_union, parent_other,
            outside_parent, pd.DataFrame(audit_rows), parent)


def assemble_scores(raw, lodo, physical, in_sample):
    raw_cols = ["excel_row", "date", "regime", "idx"] + FEATURES
    lodo_cols = ["excel_row", "assigned_component", "posterior_max", "entropy",
                 "global_nll", "mahalanobis_d2", "global_nll_train_percentile",
                 "mahalanobis_d2_train_percentile", "fold", "held_out_date"]
    phys_cols = ["excel_row", "conditional_z_F", "conditional_z_I",
                 "conditional_z_V", "conditional_z_t", "indentation_candidate",
                 "insufficient_candidate", "crack_related_candidate",
                 "any_physical_candidate", "max_abs_conditional_z",
                 "row_review_candidate", "safe_segment"]
    score = raw[raw_cols].merge(lodo[lodo_cols], on="excel_row", validate="1:1")
    score = score.merge(physical[phys_cols], on="excel_row", validate="1:1")
    fixed = in_sample[["excel_row", "assigned_component"]].rename(
        columns={"assigned_component": "in_sample_component"})
    score = score.merge(fixed, on="excel_row", validate="1:1")
    score["gmm_nll_q90"] = score.global_nll_train_percentile.ge(.90)
    score["gmm_d2_q90"] = score.mahalanobis_d2_train_percentile.ge(.90)
    score["gmm_nll_q95"] = score.global_nll_train_percentile.ge(.95)
    score["gmm_d2_q95"] = score.mahalanobis_d2_train_percentile.ge(.95)
    score["conditional_z_2_5"] = score.max_abs_conditional_z.ge(2.5)
    score["physical_global_candidate"] = score.row_review_candidate.astype(bool)
    zcols = [f"conditional_z_{name}" for name in SHORT]
    score["gmm_dominant_variable"] = np.array(SHORT)[
        np.abs(score[zcols].to_numpy()).argmax(axis=1)]
    return score


def group_decomposition(score, core, parent_other, outside_parent):
    masks = {
        "A": score.regime.eq("A").to_numpy(),
        "B-all": score.regime.eq("B").to_numpy(),
        "B-repeat298-core": core,
        "B-nonrepeat": score.regime.eq("B").to_numpy() & ~core,
        "parent-repeat-other": parent_other,
        "outside-parent": outside_parent,
    }
    metrics = ["global_nll", "mahalanobis_d2", "posterior_max",
               "max_abs_conditional_z"]
    flags = ["gmm_nll_q90", "gmm_d2_q90", "gmm_nll_q95", "gmm_d2_q95",
             "conditional_z_2_5", "physical_global_candidate"]
    rows = []
    for name, mask in masks.items():
        part = score.loc[mask]
        row = {"group": name, "rows": len(part),
               "share_of_all": len(part) / len(score)}
        for field in metrics:
            values = part[field]
            row.update({f"{field}_mean": values.mean(),
                        f"{field}_median": values.median(),
                        f"{field}_p90": values.quantile(.90),
                        f"{field}_p95": values.quantile(.95)})
        for field in flags:
            row[f"{field}_count"] = int(part[field].sum())
            row[f"{field}_rate"] = float(part[field].mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    diff_rows = []
    lookup = {name: score.loc[mask] for name, mask in masks.items()}
    for field in ("global_nll", "mahalanobis_d2"):
        a = lookup["A"][field].mean()
        b = lookup["B-all"][field].mean()
        non = lookup["B-nonrepeat"][field].mean()
        outside = lookup["outside-parent"][field].mean()
        denominator = b - a
        diff_rows.append({
            "metric": field, "A_mean": a, "B_all_mean": b,
            "B_nonrepeat_mean": non, "outside_parent_mean": outside,
            "B_minus_A": denominator,
            "repeat_removed_gap_reduction": ((b - non) / denominator
                                              if abs(denominator) > 1e-12 else np.nan),
            "outside_parent_gap_vs_A": outside - a,
        })
    return table, pd.DataFrame(diff_rows), masks


def component_overlap(score, masks):
    rows = []
    for basis, field in (("in_sample_fixed", "in_sample_component"),
                         ("lodo_date_specific", "assigned_component")):
        for group, mask in masks.items():
            counts = score.loc[mask, field].value_counts().sort_index()
            for component, count in counts.items():
                rows.append({"score_basis": basis, "group": group,
                             "component": int(component), "rows": int(count),
                             "share": float(count / mask.sum())})
    return pd.DataFrame(rows)


def relative_profile(repeat: pd.DataFrame):
    grouped = repeat.groupby("relative_position", sort=True)
    profile = pd.DataFrame({"relative_position": np.arange(1, 299)})
    for feature, short in zip(FEATURES, SHORT):
        profile[f"raw_{short}"] = grouped[feature].first().to_numpy()
    metrics = ["global_nll", "mahalanobis_d2", "global_nll_train_percentile",
               "mahalanobis_d2_train_percentile", "max_abs_conditional_z"]
    for field in metrics:
        profile[f"{field}_median"] = grouped[field].median().to_numpy()
        profile[f"{field}_min"] = grouped[field].min().to_numpy()
        profile[f"{field}_max"] = grouped[field].max().to_numpy()
    flags = ["gmm_nll_q90", "gmm_d2_q90", "gmm_nll_q95", "gmm_d2_q95",
             "conditional_z_2_5", "physical_global_candidate",
             "indentation_candidate", "insufficient_candidate",
             "crack_related_candidate", "any_physical_candidate"]
    for field in flags:
        support = grouped[field].sum().astype(int).to_numpy()
        profile[f"{field}_support"] = support
        profile[f"{field}_positive_ge1"] = support >= 1
        profile[f"{field}_positive_ge3"] = support >= 3
        profile[f"{field}_positive_eq4"] = support == 4
    dominant = grouped["gmm_dominant_variable"].agg(
        lambda values: values.value_counts().sort_index().idxmax())
    dominant_support = grouped["gmm_dominant_variable"].agg(
        lambda values: int(values.value_counts().max()))
    profile["gmm_dominant_variable_mode"] = dominant.to_numpy()
    profile["gmm_dominant_variable_mode_support"] = dominant_support.to_numpy()
    return profile


def physical_sensitivity(profile: pd.DataFrame):
    d2_field = "mahalanobis_d2_median"
    z_field = "max_abs_conditional_z_median"
    d2_q90 = float(profile[d2_field].quantile(.90))
    z_q90 = float(profile[z_field].quantile(.90))
    table = profile[["relative_position", d2_field, z_field,
                     "physical_global_candidate_support",
                     "physical_global_candidate_positive_ge1",
                     "physical_global_candidate_positive_ge3",
                     "physical_global_candidate_positive_eq4",
                     "any_physical_candidate_support",
                     "any_physical_candidate_positive_ge3"]].copy()
    table["motif_relative_d2_q90"] = table[d2_field].ge(d2_q90)
    table["motif_relative_z_q90"] = table[z_field].ge(z_q90)
    table["motif_relative_physical_sensitivity"] = (
        table.any_physical_candidate_positive_ge3 &
        (table.motif_relative_d2_q90 | table.motif_relative_z_q90))
    existing = table.physical_global_candidate_positive_ge3
    sensitivity = table.motif_relative_physical_sensitivity
    table["comparison"] = np.select(
        [existing & sensitivity, existing & ~sensitivity, ~existing & sensitivity],
        ["both", "existing_only", "sensitivity_only"], default="neither")
    union = int((existing | sensitivity).sum())
    summary = {
        "motif_d2_median_q90": d2_q90,
        "motif_max_abs_z_median_q90": z_q90,
        "existing_primary_positions": int(existing.sum()),
        "motif_relative_sensitivity_positions": int(sensitivity.sum()),
        "intersection_positions": int((existing & sensitivity).sum()),
        "union_positions": union,
        "jaccard": float((existing & sensitivity).sum() / union) if union else None,
    }
    return table, summary


def episode_tables(repeat: pd.DataFrame, profile: pd.DataFrame):
    methods = ["gmm_nll_q90", "gmm_d2_q90", "conditional_z_2_5",
               "physical_global_candidate"]
    rows = []
    for occurrence_id, part in repeat.groupby("occurrence_id", sort=True):
        ordered = part.sort_values("relative_position")
        for method in methods:
            for start, end, length in runs(ordered[method].to_numpy(bool)):
                rows.append({"scope": "occurrence", "occurrence_id": occurrence_id,
                             "method": method, "support_rule": "row_candidate",
                             "start_position": start, "end_position": end,
                             "length": length})
    for method in methods:
        support = profile[f"{method}_support"].to_numpy()
        for rule, threshold in (("ge1", 1), ("ge3_primary", 3), ("eq4", 4)):
            mask = support == 4 if rule == "eq4" else support >= threshold
            for start, end, length in runs(mask):
                rows.append({"scope": "motif", "occurrence_id": "all_four",
                             "method": method, "support_rule": rule,
                             "start_position": start, "end_position": end,
                             "length": length})
    episodes = pd.DataFrame(rows)
    envelopes = []
    primary_masks = np.column_stack([
        profile[f"{method}_positive_ge3"].to_numpy(bool) for method in methods])
    for start, end, length in runs(primary_masks.any(axis=1)):
        inside = slice(start - 1, end)
        envelopes.append({
            "scope": "motif", "support_rule": "ge3_primary_union_context_only",
            "start_position": start, "end_position": end, "length": length,
            "methods_present": "|".join(
                method for method, values in zip(methods, primary_masks.T)
                if values[inside].any()),
        })
    return episodes, pd.DataFrame(envelopes)


def result_comparison(repeat: pd.DataFrame, result: pd.DataFrame):
    quality = result.pivot_table(index="date", columns="defect type", values="defect",
                                 aggfunc="first").rename(
        columns={1: "result_type1", 2: "result_type2", 3: "result_type3"})
    rows = []
    for occurrence_id, part in repeat.groupby("occurrence_id", sort=True):
        date = part.date.iloc[0]
        row = {"occurrence_id": occurrence_id, "date": date, "rows": len(part),
               "raw4_identical_across_occurrences": True,
               "gmm_scores_reference_dependent": True}
        for field in ("global_nll", "mahalanobis_d2", "max_abs_conditional_z"):
            row[f"{field}_mean"] = float(part[field].mean())
            row[f"{field}_median"] = float(part[field].median())
        for field in ("gmm_nll_q90", "gmm_d2_q90", "conditional_z_2_5",
                      "physical_global_candidate"):
            row[f"{field}_rate"] = float(part[field].mean())
        for result_field in ("result_type1", "result_type2", "result_type3"):
            row[result_field] = (quality.at[date, result_field]
                                 if date in quality.index and result_field in quality else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw, result, lodo, physical, in_sample, blocks = load_inputs()
    (core, occurrence, relative, parent_union, parent_other, outside_parent,
     occurrence_audit, parent_blocks) = repeat_and_parent_audit(raw, blocks)
    score = assemble_scores(raw, lodo, physical, in_sample)
    decomposition, difference, masks = group_decomposition(
        score, core, parent_other, outside_parent)
    components = component_overlap(score, masks)

    repeat = score.loc[core].copy()
    repeat["occurrence_id"] = occurrence[core]
    repeat["relative_position"] = relative[core]
    repeat["parent_repeat_union"] = parent_union[core]
    repeat["parent_repeat_other"] = parent_other[core]
    repeat["outside_parent"] = outside_parent[core]
    repeat = repeat.sort_values(["occurrence_id", "relative_position"])
    profile = relative_profile(repeat)
    physical_table, physical_summary = physical_sensitivity(profile)
    episodes, envelopes = episode_tables(repeat, profile)
    result_table = result_comparison(repeat, result)

    write_csv(repeat, "repeat298_mapping.csv")
    write_csv(occurrence_audit, "repeat298_occurrence_audit.csv")
    write_csv(parent_blocks, "repeat298_parent_blocks.csv")
    write_csv(decomposition, "repeat298_gmm_decomposition.csv")
    write_csv(difference, "repeat298_gmm_difference.csv")
    write_csv(components, "repeat298_component_overlap.csv")
    write_csv(profile, "repeat298_relative_profile.csv")
    write_csv(physical_table, "physical_screening_motif_sensitivity.csv")
    write_csv(episodes, "repeat298_episodes.csv")
    write_csv(envelopes, "repeat298_contextual_envelopes.csv")
    write_csv(result_table, "result_exploratory_comparison.csv")

    ae_exports = find_ae_exports()
    ae_status = "available" if ae_exports else "not_run_missing_input"
    ae_audit = pd.DataFrame([{
        "status": ae_status,
        "required_columns": "excel_row|date|ae_total_error|ae_candidate",
        "feature_error_columns": "ae_error_F|ae_error_I|ae_error_V|ae_error_t",
        "matching_exports": len(ae_exports),
        "note": ("candidate export found; integration requires documented model metadata"
                 if ae_exports else "no repository export satisfies the minimum row contract"),
    }])
    write_csv(ae_audit, "ae_input_audit.csv")

    group_rows = {row.group: int(row.rows) for row in decomposition.itertuples()}
    primary_positions = {
        method: int(profile[f"{method}_positive_ge3"].sum())
        for method in ("gmm_nll_q90", "gmm_d2_q90", "conditional_z_2_5",
                       "physical_global_candidate")
    }
    primary_episode_counts = {
        method: int(((episodes.scope == "motif") &
                     (episodes.support_rule == "ge3_primary") &
                     (episodes.method == method)).sum())
        for method in primary_positions
    }
    summary = {
        "status": "partial_completed_missing_ae" if not ae_exports else "ae_input_review_required",
        "raw_rows": len(raw), "repeat_rows": int(core.sum()),
        "relative_positions": len(profile), "occurrences": len(OCCURRENCES),
        "raw4_identical": bool(occurrence_audit.raw4_identical_to_first.all()),
        "parent_candidate_blocks": len(parent_blocks),
        "B_rows": group_rows["B-all"],
        "repeat298_core_rows": group_rows["B-repeat298-core"],
        "B_nonrepeat_rows": group_rows["B-nonrepeat"],
        "parent_repeat_other_rows": group_rows["parent-repeat-other"],
        "outside_parent_rows": group_rows["outside-parent"],
        "primary_positive_positions": primary_positions,
        "primary_episode_counts": primary_episode_counts,
        "physical_sensitivity": physical_summary,
        "ae_status": ae_status, "ae_exports": ae_exports,
        "agreement_status": "not_run_missing_input" if not ae_exports else "not_run_metadata_review",
        "dominant_variable_agreement_status": "not_run_missing_input" if not ae_exports else "not_run_metadata_review",
        "consensus_episode_status": "not_run_missing_input" if not ae_exports else "not_run_metadata_review",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    input_paths = {
        "raw": RAW_PATH,
        "exact_repeated_blocks": (HERE.parent / "ysh-001" / "outputs" /
                                  "exact_repeated_blocks.csv"),
        "lodo_row_scores": HERE.parent / "ysh-007" / "outputs" / "lodo_row_scores.csv",
        "in_sample_scores": HERE.parent / "ysh-007" / "outputs" / "in_sample_scores.csv",
        "physical_row_followup": (HERE.parent / "ysh-007" / "outputs" /
                                   "physical_row_followup.csv"),
    }
    manifest = {
        "experiment": "YSH-008", "status": summary["status"],
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
        "code_sha256": {"analyze.py": sha256(HERE / "analyze.py")},
        "plan_sha256": sha256(HERE / "실험계획.md"),
        "stages": {"008A": "completed", "008B": "completed",
                   "008C": summary["agreement_status"],
                   "008D": summary["dominant_variable_agreement_status"],
                   "008E": "completed", "008F": "gmm_only_missing_ae",
                   "008G": "completed_descriptive_only"},
        "ae_exports": ae_exports,
    }
    (OUT / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
