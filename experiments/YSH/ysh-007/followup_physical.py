"""Post-hoc physical signatures and ordered-neighbor audit for YSH-007.

Uses the saved Raw4 GMMs; never fits or changes the selected models.
All signatures and window boundaries are exploratory, not defect/cycle labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import OUT, SHORT, load_inputs, read_config, sha256, write_csv


WIDTHS = (3, 4, 8, 16)
SIGNATURES = ("indentation", "insufficient", "crack_related")


def conditional_z(model, scaler, x: np.ndarray) -> np.ndarray:
    """Signed component-conditional residuals in model-standardized coordinates."""
    z = scaler.transform(x)
    assigned = model.predict_proba(z).argmax(axis=1)
    delta = z - model.means_[assigned]
    if model.covariance_type == "diag":
        precision = model.precisions_[assigned]
        return delta * np.sqrt(precision)
    precision = model.precisions_[assigned]
    numerator = np.einsum("nij,nj->ni", precision, delta)
    diagonal = np.diagonal(precision, axis1=1, axis2=2)
    return numerator / np.sqrt(diagonal)


def add_proxies(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    i_a = out["I"].to_numpy(float) * 1000
    t_s = out["t"].to_numpy(float) / 1000
    f = out["F"].to_numpy(float)
    if np.any(i_a <= 0) or np.any(t_s <= 0) or np.any(f <= 0):
        raise ValueError("Nonpositive I/t/F: proxy denominators require review")
    out["R_proxy_ohm"] = out["V"] / i_a
    out["P_proxy_w"] = out["V"] * i_a
    out["E_proxy_j"] = out["P_proxy_w"] * t_s
    out["H_proxy_a2s"] = i_a * i_a * t_s
    out["E_over_F_proxy"] = out["E_proxy_j"] / f
    out["H_over_F_proxy"] = out["H_proxy_a2s"] / f
    out["F_times_I"] = out["F"] * out["I"]
    out["F_times_R_proxy"] = out["F"] * out["R_proxy_ohm"]
    return out


def add_signatures(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Date-held-out, Result-blind cutoffs; not fitted model features."""
    out = rows.copy()
    audit = []
    cols = ("F", "I", "t", "E_proxy_j", "H_proxy_a2s")
    for date in sorted(out.date.unique()):
        train = out[out.date.ne(date)]
        held = out.date.eq(date)
        quant = train[list(cols)].quantile([.10, .25, .75, .90])
        q = lambda col, p: float(quant.loc[p, col])
        for col in cols:
            for percentile in (.10, .25, .75, .90):
                audit.append({"held_out_date": date, "column": col,
                              "quantile": percentile, "train_cutoff": q(col, percentile),
                              "train_rows": len(train)})
        # These are deliberately broad physical screening hypotheses.
        out.loc[held, "indentation_candidate"] = (
            (out.loc[held, "I"] >= q("I", .75)) &
            (out.loc[held, "t"] >= q("t", .75)) &
            (out.loc[held, "E_proxy_j"] >= q("E_proxy_j", .90)))
        out.loc[held, "insufficient_candidate"] = (
            ((out.loc[held, "I"] <= q("I", .25)) &
             (out.loc[held, "E_proxy_j"] <= q("E_proxy_j", .10))) |
            ((out.loc[held, "F"] >= q("F", .75)) &
             (out.loc[held, "E_proxy_j"] <= q("E_proxy_j", .25))))
        out.loc[held, "crack_related_candidate"] = (
            (out.loc[held, "E_proxy_j"] >= q("E_proxy_j", .90)) &
            (out.loc[held, "F"] >= q("F", .75)) &
            ((out.loc[held, "I"] >= q("I", .75)) |
             (out.loc[held, "t"] >= q("t", .75))))
    for name in SIGNATURES:
        out[f"{name}_candidate"] = out[f"{name}_candidate"].astype(bool)
    out["any_physical_candidate"] = out[[f"{s}_candidate" for s in SIGNATURES]].any(axis=1)
    out["high_internal_d2"] = out.mahalanobis_d2_train_percentile >= .90
    out["very_high_internal_d2"] = out.mahalanobis_d2_train_percentile >= .95
    out["high_global_nll"] = out.global_nll_train_percentile >= .90
    out["max_abs_conditional_z"] = out[[f"conditional_z_{s}" for s in SHORT]].abs().max(axis=1)
    out["row_review_candidate"] = out.any_physical_candidate & (
        out.high_internal_d2 | (out.max_abs_conditional_z >= 2.5))
    return out, pd.DataFrame(audit)


def assign_segments(rows: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    seg = segments[segments.condition.eq("all")].sort_values("excel_start")
    out["safe_segment"] = ""
    out["segment_start"] = -1
    out["segment_end"] = -1
    for record in seg.itertuples(index=False):
        mask = out.excel_row.between(record.excel_start, record.excel_end)
        out.loc[mask, "safe_segment"] = str(record.segment)
        out.loc[mask, "segment_start"] = int(record.excel_start)
        out.loc[mask, "segment_end"] = int(record.excel_end)
    if out.safe_segment.eq("").any():
        raise ValueError("Unmapped safe segment")
    return out


def component_profiles(rows: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    # One full-data reference label system, strictly descriptive in-sample.
    joined = rows.merge(full[["excel_row", "assigned_component"]], on="excel_row",
                        validate="one_to_one", suffixes=("", "_full"))
    cols = ["F", "I", "V", "t", "R_proxy_ohm", "E_proxy_j", "H_proxy_a2s"]
    out = []
    for component, frame in joined.groupby("assigned_component_full"):
        rec = {"component_full": int(component), "rows": len(frame),
               "A_share": float(frame.regime.eq("A").mean()),
               "B_share": float(frame.regime.eq("B").mean())}
        for col in cols:
            rec.update({f"{col}_{metric}": float(value) for metric, value in
                        zip(("p10", "median", "p90"), frame[col].quantile([.1, .5, .9]))})
        for sig in SIGNATURES:
            rec[f"{sig}_share"] = float(frame[f"{sig}_candidate"].mean())
        rec["lodo_high_d2_share"] = float(frame.high_internal_d2.mean())
        rec["lodo_high_nll_share"] = float(frame.high_global_nll.mean())
        out.append(rec)
    return pd.DataFrame(out)


def daily(rows: pd.DataFrame, quality: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for (date, component), frame in rows.groupby(["date", "assigned_component"]):
        rec = {"date": date, "regime": frame.regime.iloc[0],
               "component_lodo": int(component), "rows": len(frame),
               "review_rows": int(frame.row_review_candidate.sum()),
               "high_d2_rows": int(frame.high_internal_d2.sum()),
               "high_nll_rows": int(frame.high_global_nll.sum())}
        for sig in SIGNATURES:
            flag = frame[f"{sig}_candidate"]
            rec[f"{sig}_rows"] = int(flag.sum())
            rec[f"{sig}_high_d2_rows"] = int((flag & frame.high_internal_d2).sum())
        records.append(rec)
    component = pd.DataFrame(records)
    day = component.groupby(["date", "regime"], as_index=False).sum(numeric_only=True)
    day = day.drop(columns="component_lodo")
    day = day.merge(quality[["date", "type1", "type2", "type3"]],
                    on="date", validate="one_to_one")
    return component, day


def runs(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = rows.sort_values("excel_row").copy()
    switch = (out.date.ne(out.date.shift()) |
              out.safe_segment.ne(out.safe_segment.shift()) |
              out.assigned_component.ne(out.assigned_component.shift()) |
              out.excel_row.ne(out.excel_row.shift() + 1))
    out["component_run_id"] = switch.cumsum().astype(int)
    summary = out.groupby("component_run_id").agg(
        date=("date", "first"), regime=("regime", "first"),
        safe_segment=("safe_segment", "first"),
        component_lodo=("assigned_component", "first"),
        start_excel_row=("excel_row", "min"), end_excel_row=("excel_row", "max"),
        n_rows=("excel_row", "size"), review_rows=("row_review_candidate", "sum"),
        high_d2_rows=("high_internal_d2", "sum"),
        E_proxy_median=("E_proxy_j", "median"),
        max_abs_conditional_z=("max_abs_conditional_z", "max"))
    for sig in SIGNATURES:
        summary[f"{sig}_rows"] = out.groupby("component_run_id")[f"{sig}_candidate"].sum()
    summary = summary.reset_index()
    for side, shift in (("previous", 1), ("next", -1)):
        candidate = summary.shift(shift)
        same = summary.date.eq(candidate.date) & summary.safe_segment.eq(candidate.safe_segment)
        summary[f"{side}_run_id"] = candidate.component_run_id.where(same).astype("Int64")
        summary[f"{side}_component"] = candidate.component_lodo.where(same).astype("Int64")
        summary[f"{side}_high_d2_rows"] = candidate.high_d2_rows.where(same).astype("Int64")
    return out, summary


def periodic_audit(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (date, segment), frame in rows.groupby(["date", "safe_segment"]):
        a = frame.sort_values("excel_row").assigned_component.to_numpy()
        if len(a) < 40:
            continue
        for lag in range(2, 33):
            records.append({"date": date, "regime": frame.regime.iloc[0],
                            "safe_segment": str(segment), "segment_rows": len(a),
                            "lag": lag, "same_component_rate": float(np.mean(a[lag:] == a[:-lag])),
                            "pairs": len(a) - lag})
    out = pd.DataFrame(records)
    out["local_peak_excess"] = np.nan
    for (_, _), frame in out.groupby(["date", "safe_segment"]):
        ordered = frame.sort_values("lag")
        v = ordered.same_component_rate.to_numpy()
        excess = v[1:-1] - (v[:-2] + v[2:]) / 2
        out.loc[ordered.index[1:-1], "local_peak_excess"] = excess
    return out


def neighborhood(rows: pd.DataFrame) -> pd.DataFrame:
    """All candidate-row contexts, with explicit width/phase sensitivity."""
    indexed = rows.set_index("excel_row", verify_integrity=True)
    candidates = rows[rows.row_review_candidate]
    records = []
    for row in candidates.itertuples(index=False):
        for width in WIDTHS:
            for phase in (range(width) if width in (3, 4) else (0, width // 2)):
                relative = row.excel_row - row.segment_start
                block = (relative - phase) // width
                start = row.segment_start + phase + block * width
                end = start + width - 1
                if start < row.segment_start or end > row.segment_end:
                    continue
                rec = {"anchor_excel_row": row.excel_row, "date": row.date,
                       "regime": row.regime, "component_lodo": row.assigned_component,
                       "component_run_id": row.component_run_id,
                       "width": width, "phase": phase,
                       "current_start": start, "current_end": end}
                for label, lo, hi in (("previous", start-width, start-1),
                                      ("current", start, end),
                                      ("next", end+1, end+width)):
                    if lo < row.segment_start or hi > row.segment_end:
                        rec[f"{label}_available"] = False
                        continue
                    sub = indexed.loc[lo:hi]
                    if len(sub) != width or not sub.safe_segment.eq(row.safe_segment).all():
                        raise ValueError("Context crossed a safe segment")
                    rec[f"{label}_available"] = True
                    rec[f"{label}_component_mode"] = int(sub.assigned_component.mode().iloc[0])
                    rec[f"{label}_high_d2_share"] = float(sub.high_internal_d2.mean())
                    rec[f"{label}_review_share"] = float(sub.row_review_candidate.mean())
                    rec[f"{label}_E_proxy_median"] = float(sub.E_proxy_j.median())
                    for sig in SIGNATURES:
                        rec[f"{label}_{sig}_share"] = float(sub[f"{sig}_candidate"].mean())
                records.append(rec)
    return pd.DataFrame(records)


def main() -> None:
    config = read_config()
    raw, quality, segments, _, _, digest = load_inputs(config)
    lodo_path = OUT / "lodo_row_scores.csv"
    full_path = OUT / "in_sample_scores.csv"
    lodo = pd.read_csv(lodo_path)
    full = pd.read_csv(full_path)
    cols = dict(zip(config["features"], SHORT))
    rows = raw[["excel_row", "date", "regime", *config["features"]]].rename(columns=cols)
    rows = rows.merge(lodo.drop(columns=["date", "regime", "phase", "held_out_date"]),
                      on="excel_row", validate="one_to_one")
    if len(rows) != 11939 or not np.array_equal(rows.excel_row, lodo.sort_values("excel_row").excel_row):
        raise ValueError("Original-to-score row mapping failed")
    scored_order = lodo.sort_values("excel_row")
    if not np.array_equal(rows["date"].to_numpy(), scored_order["date"].to_numpy()):
        raise ValueError("Raw/LODO date mapping failed")
    if not np.array_equal(rows["regime"].to_numpy(), scored_order["regime"].to_numpy()):
        raise ValueError("Raw/LODO regime mapping failed")
    rows = add_proxies(rows)
    for held in config["dates"]:
        mask = rows.date.eq(held)
        saved = joblib.load(OUT / "models_lodo" / f"{held}.joblib")
        model, scaler = saved["model"], saved["scaler"]
        z = conditional_z(model, scaler, rows.loc[mask, SHORT].to_numpy(float))
        for j, name in enumerate(SHORT):
            rows.loc[mask, f"conditional_z_{name}"] = z[:, j]
    rows, cutoffs = add_signatures(rows)
    rows = assign_segments(rows, segments)
    rows, run_table = runs(rows)
    component, day = daily(rows, quality)
    profile = component_profiles(rows, full)
    period = periodic_audit(rows)
    neighbors = neighborhood(rows)
    write_csv(rows, "physical_row_followup.csv")
    write_csv(cutoffs, "physical_signature_cutoffs.csv")
    write_csv(profile, "physical_component_profiles.csv")
    write_csv(component, "physical_daily_component.csv")
    write_csv(day, "physical_daily_result.csv")
    write_csv(run_table, "physical_component_runs.csv")
    write_csv(period, "physical_sequence_lags.csv")
    write_csv(neighbors, "physical_neighbor_windows.csv")
    summary = {
        "status": "completed_exploratory", "model_refit": False,
        "raw_sha256": digest,
        "lodo_scores_sha256": sha256(lodo_path),
        "full_scores_sha256": sha256(full_path),
        "rows": len(rows), "safe_segments": int(rows.safe_segment.nunique()),
        "component_runs": len(run_table),
        "component_run_median_rows": float(run_table.n_rows.median()),
        "component_run_p95_rows": float(run_table.n_rows.quantile(.95)),
        "row_review_candidates": int(rows.row_review_candidate.sum()),
        "signature_counts": {s: int(rows[f"{s}_candidate"].sum()) for s in SIGNATURES},
        "candidate_contexts": len(neighbors),
        "candidate_context_widths": list(WIDTHS),
        "conditional_z_note": "component-conditional diagnostic; not calibrated defect probability",
        "cycle_note": "component runs and fixed windows are candidate contexts, not verified physical cycles",
    }
    (OUT / "physical_followup_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
