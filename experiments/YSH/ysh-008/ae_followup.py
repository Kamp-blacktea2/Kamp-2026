from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "outputs"
YSH005 = HERE.parent / "ysh-005" / "outputs"
RAW_PATH = ROOT / "data" / "Welding_Data_Set_01.xlsx"
RAW_SHA256 = "d514d6aaa121630c04d7d51c97a56025e78e2c86868813f7722ba5db922c1f33"
FEATURES = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
SHORT = ["F", "I", "V", "t"]
GMM_METHODS = ["gmm_nll_q90", "gmm_d2_q90", "conditional_z_2_5",
               "physical_global_candidate"]
RULES = (("ge1", 1), ("ge3_primary", 3), ("eq4", 4))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, float_format="%.15g")


def apply_rule(support: np.ndarray, rule: str, threshold: int) -> np.ndarray:
    return support == 4 if rule == "eq4" else support >= threshold


def runs(mask: np.ndarray):
    positions = np.flatnonzero(mask) + 1
    if not len(positions):
        return []
    cuts = np.flatnonzero(np.diff(positions) > 1) + 1
    return [(int(x[0]), int(x[-1]), int(len(x))) for x in np.split(positions, cuts)]


def dominant_labels(errors: np.ndarray) -> np.ndarray:
    maximum = errors.max(axis=1, keepdims=True)
    tied = np.isclose(errors, maximum, rtol=0, atol=1e-15).sum(axis=1) > 1
    labels = np.array(SHORT, dtype=object)[errors.argmax(axis=1)]
    labels[tied] = "tie"
    return labels


def load_ae():
    if sha256(RAW_PATH) != RAW_SHA256:
        raise ValueError("Raw SHA-256 mismatch")
    raw = pd.read_excel(RAW_PATH, sheet_name="Raw data")
    result = pd.read_excel(RAW_PATH, sheet_name="result")
    raw["excel_row"] = np.arange(len(raw)) + 2
    raw["date"] = pd.to_datetime(raw["working time"]).dt.strftime("%Y-%m-%d")
    result["date"] = pd.to_datetime(result["working time"]).dt.strftime("%Y-%m-%d")
    train = raw.date.le("2020-03-31").to_numpy()
    if train.sum() != 8470:
        raise AssertionError("YSH-005 train rows differ")
    fit = json.loads((YSH005 / "ae_fit.json").read_text(encoding="utf-8"))
    status = json.loads((YSH005 / "ae_status.json").read_text(encoding="utf-8"))
    model_rows = pd.read_csv(YSH005 / "tables" / "row_models.csv")
    stored_rows = pd.read_csv(YSH005 / "tables" / "row_scores.csv")
    model_row = model_rows.loc[model_rows.model.eq("ae_42")].iloc[0]
    artifact = np.load(YSH005 / "models" / "ae_42_reconstruction.npz")
    scaled_input, reconstruction = artifact["input"], artifact["output"]
    if scaled_input.shape != (11939, 4) or reconstruction.shape != (11939, 4):
        raise AssertionError("YSH-005 AE reconstruction shape")
    x = raw[FEATURES].to_numpy(float)
    train_min, train_max = x[train].min(axis=0), x[train].max(axis=0)
    direct_scaled = (x - train_min) / (train_max - train_min)
    if not np.allclose(train_min, fit["min"], atol=1e-12, rtol=0):
        raise AssertionError("train-only scaler minimum")
    if not np.allclose(train_max, fit["max"], atol=1e-12, rtol=0):
        raise AssertionError("train-only scaler maximum")
    if not np.allclose(direct_scaled, scaled_input, atol=2e-7, rtol=0):
        raise AssertionError("saved AE input differs from train-only MinMax")
    # Preserve the original YSH-005 PyTorch float32 subtraction/square/mean order.
    feature_error = ((reconstruction - scaled_input) ** 2).astype(float)
    total_error = ((reconstruction - scaled_input) ** 2).mean(axis=1).astype(float)
    threshold = total_error[train].mean() + 8 * total_error[train].std(ddof=0)
    if not np.isclose(threshold, model_row.threshold, atol=1e-12, rtol=0):
        raise AssertionError("AE threshold mismatch")
    if not np.allclose(total_error, stored_rows.ae_42_score, atol=1e-12, rtol=0):
        raise AssertionError("AE total score mismatch")
    candidate = total_error >= threshold
    if not np.array_equal(candidate, stored_rows.ae_42_flag.to_numpy(bool)):
        raise AssertionError("AE candidate mismatch")
    if status.get("status") != "complete":
        raise AssertionError("YSH-005 AE artifact not complete")
    ae = raw[["excel_row", "date"]].copy()
    ae["ae_total_error"] = total_error
    ae["ae_candidate"] = candidate
    for i, name in enumerate(SHORT):
        ae[f"ae_error_{name}"] = feature_error[:, i]
    ae["ae_dominant_variable"] = dominant_labels(feature_error)
    metadata = {
        "source": "YSH-005 saved seed42 reconstruction",
        "features": FEATURES, "scaler": "train-only MinMaxScaler",
        "train_last_date": "2020-03-31", "train_rows": int(train.sum()),
        "architecture": "4-3-2-3-4 RReLU", "optimizer": "Adam",
        "learning_rate": 0.01, "batch_size": 64, "epochs": 50, "seed": 42,
        "loss": "mean squared error in scaled feature space",
        "feature_error": "squared error in each scaled feature",
        "threshold_rule": "train mean + 8 * population std",
        "threshold": float(threshold), "train_candidates": int(candidate[train].sum()),
        "test_candidates": int(candidate[~train].sum()),
        "artifact_sha256": sha256(YSH005 / "models" / "ae_42_reconstruction.npz"),
    }
    return raw, result, ae, metadata


def repeat_sanity(ae: pd.DataFrame, mapping: pd.DataFrame):
    repeat = mapping.merge(ae, on=["excel_row", "date"], validate="1:1")
    fields = ["ae_total_error", "ae_error_F", "ae_error_I", "ae_error_V",
              "ae_error_t"]
    rows = []
    for position, part in repeat.groupby("relative_position", sort=True):
        row = {"relative_position": int(position),
               "ae_candidate_support": int(part.ae_candidate.sum())}
        for field in fields:
            row[f"{field}_min"] = float(part[field].min())
            row[f"{field}_max"] = float(part[field].max())
            row[f"{field}_range"] = float(part[field].max() - part[field].min())
        row["all_errors_exactly_identical"] = all(
            part[field].nunique(dropna=False) == 1 for field in fields)
        row["candidate_identical"] = part.ae_candidate.nunique(dropna=False) == 1
        rows.append(row)
    sanity = pd.DataFrame(rows)
    if not sanity.all_errors_exactly_identical.all() or not sanity.candidate_identical.all():
        raise AssertionError("identical Raw4 has differing fixed row-wise AE errors")
    return repeat, sanity


def agreement_tables(repeat: pd.DataFrame):
    grouped = repeat.groupby("relative_position", sort=True)
    position = pd.DataFrame({"relative_position": np.arange(1, 299)})
    position["ae_total_error"] = grouped.ae_total_error.first().to_numpy()
    position["ae_candidate_support"] = grouped.ae_candidate.sum().astype(int).to_numpy()
    for name in SHORT:
        position[f"ae_error_{name}"] = grouped[f"ae_error_{name}"].first().to_numpy()
    position["ae_dominant_variable"] = grouped.ae_dominant_variable.first().to_numpy()
    for field in ("global_nll", "mahalanobis_d2", "max_abs_conditional_z"):
        position[f"{field}_median"] = grouped[field].median().to_numpy()
    for method in GMM_METHODS:
        position[f"{method}_support"] = grouped[method].sum().astype(int).to_numpy()
    summary_rows = []
    for rule, threshold in RULES:
        ae_positive = apply_rule(position.ae_candidate_support.to_numpy(), rule, threshold)
        for method in GMM_METHODS:
            gmm_positive = apply_rule(position[f"{method}_support"].to_numpy(),
                                      rule, threshold)
            tp = int((ae_positive & gmm_positive).sum())
            fp = int((~ae_positive & gmm_positive).sum())
            fn = int((ae_positive & ~gmm_positive).sum())
            tn = int((~ae_positive & ~gmm_positive).sum())
            union, denom = tp + fp + fn, 2 * tp + fp + fn
            summary_rows.append({
                "support_rule": rule, "gmm_method": method, "positions": 298,
                "ae_positive": int(ae_positive.sum()),
                "gmm_positive": int(gmm_positive.sum()),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "intersection": tp, "union": union,
                "jaccard": tp / union if union else np.nan,
                "positive_agreement": 2 * tp / denom if denom else np.nan,
                "overall_agreement": (tp + tn) / 298,
                "cohen_kappa": cohen_kappa_score(ae_positive, gmm_positive),
            })
    return position, pd.DataFrame(summary_rows)


def score_correlations(repeat: pd.DataFrame, position: pd.DataFrame):
    rows = []
    fields = ["global_nll", "mahalanobis_d2", "max_abs_conditional_z"]
    for field in fields:
        rho = spearmanr(position.ae_total_error,
                        position[f"{field}_median"]).statistic
        rows.append({"scope": "motif_position", "occurrence_id": "all_four",
                     "gmm_score": field, "n": 298, "spearman": rho})
    for occurrence_id, part in repeat.groupby("occurrence_id", sort=True):
        for field in fields:
            rho = spearmanr(part.ae_total_error, part[field]).statistic
            rows.append({"scope": "occurrence", "occurrence_id": occurrence_id,
                         "gmm_score": field, "n": len(part), "spearman": rho})
    return pd.DataFrame(rows)


def mode_or_tie(values: pd.Series) -> str:
    counts = values.value_counts()
    winners = counts[counts.eq(counts.max())].index.tolist()
    return winners[0] if len(winners) == 1 else "tie"


def dominant_agreement(repeat: pd.DataFrame, position: pd.DataFrame):
    rows = []
    for rule, threshold in RULES:
        ae_positive = apply_rule(position.ae_candidate_support.to_numpy(), rule, threshold)
        for method in GMM_METHODS:
            gmm_positive = apply_rule(position[f"{method}_support"].to_numpy(),
                                      rule, threshold)
            for pos in position.loc[ae_positive & gmm_positive,
                                    "relative_position"].astype(int):
                part = repeat[(repeat.relative_position == pos) & repeat[method]]
                gmm_dominant = mode_or_tie(part.gmm_dominant_variable)
                ae_dominant = position.loc[
                    position.relative_position.eq(pos), "ae_dominant_variable"].iloc[0]
                rows.append({
                    "support_rule": rule, "gmm_method": method,
                    "relative_position": pos,
                    "ae_dominant_variable": ae_dominant,
                    "gmm_dominant_variable": gmm_dominant,
                    "gmm_candidate_occurrences": len(part),
                    "dominant_match": (ae_dominant == gmm_dominant and
                                       ae_dominant != "tie"),
                })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    confusion = (detail.groupby(["support_rule", "gmm_method",
                                 "ae_dominant_variable", "gmm_dominant_variable"])
                 .size().rename("positions").reset_index())
    return detail, confusion


def episode_tables(repeat: pd.DataFrame, position: pd.DataFrame):
    rows, envelopes = [], []

    def add(scope, occurrence_id, episode_type, method, rule, mask):
        for start, end, length in runs(np.asarray(mask, dtype=bool)):
            rows.append({"scope": scope, "occurrence_id": occurrence_id,
                         "episode_type": episode_type, "gmm_method": method,
                         "support_rule": rule, "start_position": start,
                         "end_position": end, "length": length})

    for occurrence_id, part in repeat.groupby("occurrence_id", sort=True):
        ordered = part.sort_values("relative_position")
        ae_mask = ordered.ae_candidate.to_numpy(bool)
        add("occurrence", occurrence_id, "AE", "ae_candidate", "row_candidate", ae_mask)
        for method in GMM_METHODS:
            gmm_mask = ordered[method].to_numpy(bool)
            add("occurrence", occurrence_id, "GMM", method, "row_candidate", gmm_mask)
            add("occurrence", occurrence_id, "consensus", method, "row_candidate",
                ae_mask & gmm_mask)
    for rule, threshold in RULES:
        ae_mask = apply_rule(position.ae_candidate_support.to_numpy(), rule, threshold)
        add("motif", "all_four", "AE", "ae_candidate", rule, ae_mask)
        for method in GMM_METHODS:
            gmm_mask = apply_rule(position[f"{method}_support"].to_numpy(),
                                  rule, threshold)
            add("motif", "all_four", "GMM", method, rule, gmm_mask)
            add("motif", "all_four", "consensus", method, rule, ae_mask & gmm_mask)
            for start, end, length in runs(ae_mask | gmm_mask):
                envelopes.append({
                    "scope": "motif", "gmm_method": method,
                    "support_rule": rule, "definition": "AE_or_GMM_context_only",
                    "start_position": start, "end_position": end, "length": length,
                })
    return pd.DataFrame(rows), pd.DataFrame(envelopes)


def result_table(repeat: pd.DataFrame, result: pd.DataFrame):
    quality = result.pivot_table(index="date", columns="defect type", values="defect",
                                 aggfunc="first").rename(
        columns={1: "result_type1", 2: "result_type2", 3: "result_type3"})
    rows = []
    for occurrence_id, part in repeat.groupby("occurrence_id", sort=True):
        date = part.date.iloc[0]
        row = {"occurrence_id": occurrence_id, "date": date, "rows": len(part),
               "ae_total_error_mean": float(part.ae_total_error.mean()),
               "ae_total_error_median": float(part.ae_total_error.median()),
               "ae_candidate_count": int(part.ae_candidate.sum()),
               "ae_candidate_rate": float(part.ae_candidate.mean()),
               "raw4_and_ae_errors_identical_across_occurrences": True}
        for field in ("result_type1", "result_type2", "result_type3"):
            row[field] = (quality.at[date, field]
                          if date in quality.index and field in quality else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    raw, result, ae, metadata = load_ae()
    mapping = pd.read_csv(OUT / "repeat298_mapping.csv")
    repeat, sanity = repeat_sanity(ae, mapping)
    position, agreement = agreement_tables(repeat)
    correlations = score_correlations(repeat, position)
    dominant_detail, dominant_confusion = dominant_agreement(repeat, position)
    episodes, envelopes = episode_tables(repeat, position)
    result_comparison = result_table(repeat, result)

    if dominant_detail.empty:
        dominant_summary = pd.DataFrame(columns=[
            "support_rule", "gmm_method", "common_positions", "matches",
            "match_rate", "ae_mode_baseline"])
    else:
        rows = []
        for (rule, method), part in dominant_detail.groupby(
                ["support_rule", "gmm_method"], sort=True):
            counts = part.ae_dominant_variable.value_counts()
            rows.append({"support_rule": rule, "gmm_method": method,
                         "common_positions": len(part),
                         "matches": int(part.dominant_match.sum()),
                         "match_rate": float(part.dominant_match.mean()),
                         "ae_mode_baseline": float(counts.max() / len(part))})
        dominant_summary = pd.DataFrame(rows)

    write_csv(ae, "ae_row_scores.csv")
    write_csv(sanity, "ae_repeat298_sanity.csv")
    write_csv(repeat, "ae_gmm_agreement_rows.csv")
    write_csv(position, "ae_gmm_relative_profile.csv")
    write_csv(agreement, "ae_gmm_agreement_summary.csv")
    write_csv(correlations, "ae_gmm_score_correlations.csv")
    write_csv(dominant_detail, "dominant_variable_agreement_rows.csv")
    write_csv(dominant_confusion, "dominant_variable_confusion.csv")
    write_csv(dominant_summary, "dominant_variable_agreement_summary.csv")
    write_csv(episodes, "ae_gmm_episodes.csv")
    write_csv(envelopes, "ae_gmm_contextual_envelopes.csv")
    write_csv(result_comparison, "ae_result_exploratory_comparison.csv")
    (OUT / "ae_model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    primary = agreement.loc[agreement.support_rule.eq("ge3_primary")]
    primary_agreement = {
        row.gmm_method: {"gmm_positive": int(row.gmm_positive),
                         "intersection": int(row.intersection),
                         "jaccard": float(row.jaccard),
                         "cohen_kappa": float(row.cohen_kappa)}
        for row in primary.itertuples()
    }
    primary_dom = dominant_summary.loc[
        dominant_summary.support_rule.eq("ge3_primary")]
    dominant_result = {
        row.gmm_method: {"common_positions": int(row.common_positions),
                         "matches": int(row.matches),
                         "match_rate": float(row.match_rate),
                         "ae_mode_baseline": float(row.ae_mode_baseline)}
        for row in primary_dom.itertuples()
    }
    ae_positions = int((position.ae_candidate_support >= 3).sum())
    summary = {
        "status": "completed", "ae_model": "YSH-005 ae_42",
        "threshold": metadata["threshold"],
        "all_row_candidates": int(ae.ae_candidate.sum()),
        "repeat_candidate_rows": int(repeat.ae_candidate.sum()),
        "repeat_candidate_positions": ae_positions,
        "repeat_candidate_position_rate": ae_positions / 298,
        "repeat_errors_identical": bool(sanity.all_errors_exactly_identical.all()),
        "max_occurrence_error_range": float(sanity.filter(like="_range").max().max()),
        "primary_agreement": primary_agreement,
        "primary_dominant_agreement": dominant_result,
        "ae_primary_episode_count": int(((episodes.scope == "motif") &
                                         (episodes.episode_type == "AE") &
                                         (episodes.support_rule == "ge3_primary")).sum()),
    }
    (OUT / "ae_followup_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    input_paths = {
        "raw": RAW_PATH,
        "ysh005_model_code": HERE.parent / "ysh-005" / "model_candidates.py",
        "ysh005_config": HERE.parent / "ysh-005" / "config.json",
        "ysh005_ae_fit": YSH005 / "ae_fit.json",
        "ysh005_row_models": YSH005 / "tables" / "row_models.csv",
        "ysh005_reconstruction": YSH005 / "models" / "ae_42_reconstruction.npz",
        "ysh008_gmm_mapping": OUT / "repeat298_mapping.csv",
    }
    manifest = {
        "experiment": "YSH-008 AE follow-up", "status": "completed",
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
        "code_sha256": {"ae_followup.py": sha256(HERE / "ae_followup.py")},
        "model_selection_used_result": False,
        "threshold_selection_used_result": False,
        "ensemble_score_created": False, "result_joined_last": True,
        "stages": {"008C": "completed", "008D": "completed",
                   "008F_AE": "completed", "008F_consensus": "completed",
                   "008G_AE": "completed_descriptive_only"},
    }
    (OUT / "ae_followup_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
