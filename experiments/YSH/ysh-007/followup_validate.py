"""Independent checks and sensitivity summaries for YSH-007 physical follow-up."""
from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import HERE, OUT, read_config, write_csv, sha256

def safe_rho(x, y):
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(spearmanr(x, y).statistic)

def main():
    cfg = read_config()
    rows = pd.read_csv(OUT / "physical_row_followup.csv").sort_values("excel_row")
    daily = pd.read_csv(OUT / "physical_daily_result.csv")
    windows = pd.read_csv(OUT / "physical_neighbor_windows.csv")
    assert len(rows) == 11939 and rows.excel_row.is_unique
    assert set(rows.date) == set(cfg["dates"])
    assert np.allclose(rows.P_proxy_w, rows.V * rows.I * 1000)
    assert np.allclose(rows.E_proxy_j, rows.P_proxy_w * rows.t / 1000)
    assert np.allclose(rows.H_proxy_a2s, (rows.I * 1000) ** 2 * rows.t / 1000)
    assert np.allclose(rows.R_proxy_ohm, rows.V / (rows.I * 1000))
    assert (windows.current_start <= windows.anchor_excel_row).all()
    assert (windows.current_end >= windows.anchor_excel_row).all()
    assert windows.groupby(["anchor_excel_row", "width", "phase"]).size().eq(1).all()
    assert daily.rows.sum() == len(rows)
    assert daily.review_rows.sum() == rows.row_review_candidate.sum()

    # Independently reconstruct the conditional normal residual from covariance.
    conditional_error = 0.0
    for held in cfg["dates"]:
        saved = joblib.load(OUT / "models_lodo" / f"{held}.joblib")
        model, scaler = saved["model"], saved["scaler"]
        assert model.covariance_type == "full"
        sample = rows[rows.date.eq(held)].head(5)
        standardized = scaler.transform(sample[["F", "I", "V", "t"]].to_numpy(float))
        assigned = model.predict_proba(standardized).argmax(axis=1)
        for i, (_, record) in enumerate(sample.iterrows()):
            k = assigned[i]
            cov = model.covariances_[k]
            mu = model.means_[k]
            for j, name in enumerate(("F", "I", "V", "t")):
                other = [m for m in range(4) if m != j]
                cross = cov[j, other]
                minor = cov[np.ix_(other, other)]
                conditional_mean = mu[j] + cross @ np.linalg.solve(
                    minor, standardized[i, other] - mu[other])
                conditional_var = cov[j, j] - cross @ np.linalg.solve(minor, cov[other, j])
                direct = (standardized[i, j] - conditional_mean) / np.sqrt(conditional_var)
                conditional_error = max(conditional_error, abs(direct - record[f"conditional_z_{name}"]))
    assert conditional_error < 1e-7
    # Original positions are retained. Unique-endpoint pairs are a sensitivity
    # check; dropping duplicate rows would create artificial adjacencies.
    rows["unique4"] = ~rows.duplicated(["date", "F", "I", "V", "t"])
    rng = np.random.default_rng(42)
    records = []
    for (date, seg), frame in rows.groupby(["date", "safe_segment"]):
        a = frame.assigned_component.to_numpy()
        unique = frame.unique4.to_numpy()
        if len(a) < 40:
            continue
        bounds = np.r_[0, np.flatnonzero(a[1:] != a[:-1]) + 1, len(a)]
        runs = [a[bounds[j]:bounds[j+1]] for j in range(len(bounds)-1)]
        null_peak = []
        for _ in range(100):
            shuffled = np.concatenate([runs[j] for j in rng.permutation(len(runs))])
            null_peak.append(float(np.mean(shuffled[3:] == shuffled[:-3]) -
                                   (np.mean(shuffled[2:] == shuffled[:-2]) +
                                    np.mean(shuffled[4:] == shuffled[:-4])) / 2))
        for lag in range(2, 33):
            match = a[lag:] == a[:-lag]
            endpoint = unique[lag:] & unique[:-lag]
            records.append({
                "date": date, "regime": frame.regime.iloc[0], "safe_segment": seg,
                "lag": lag, "pairs": len(match), "same": int(match.sum()),
                "unique_endpoint_pairs": int(endpoint.sum()),
                "unique_endpoint_same": int((match & endpoint).sum()),
                "run_shuffle_peak3_mean": float(np.mean(null_peak)),
                "run_shuffle_peak3_p95": float(np.quantile(null_peak, .95)),
            })
    lag = pd.DataFrame(records)
    lag["same_rate"] = lag.same / lag.pairs
    lag["unique_endpoint_same_rate"] = lag.unique_endpoint_same / lag.unique_endpoint_pairs
    write_csv(lag, "physical_sequence_sensitivity.csv")

    # Each anchor contributes once per width after averaging eligible phases.
    columns = [c for c in windows if c.endswith("_share")]
    neighbor = windows.groupby(["anchor_excel_row", "date", "regime", "width"], as_index=False)[columns].mean()
    coverage = windows.groupby(["anchor_excel_row", "width"]).agg(
        valid_phases=("phase", "size"), previous_available=("previous_available", "sum"),
        next_available=("next_available", "sum")).reset_index()
    neighbor = neighbor.merge(coverage, on=["anchor_excel_row", "width"], validate="one_to_one")
    write_csv(neighbor, "physical_neighbor_summary.csv")

    # Observed Result entries only; missing records remain missing.
    metrics = ["review_rows", "indentation_rows", "insufficient_rows", "crack_related_rows",
               "indentation_high_d2_rows", "insufficient_high_d2_rows", "crack_related_high_d2_rows"]
    for col in metrics:
        daily[col + "_share"] = daily[col] / daily.rows
    out = []
    for scope in ("all", "A", "B"):
        part = daily if scope == "all" else daily[daily.regime.eq(scope)]
        for result in ("type1", "type2", "type3"):
            for metric in metrics:
                xcol = metric + "_share"
                valid = part[result].notna() & part[xcol].notna()
                x, y = part.loc[valid, xcol].to_numpy(), part.loc[valid, result].to_numpy()
                rho = safe_rho(x, y)
                loo = [safe_rho(np.delete(x, k), np.delete(y, k)) for k in range(len(x))] if len(x) >= 4 else []
                finite = [v for v in loo if np.isfinite(v)]
                out.append({"scope": scope, "result_type": result, "metric": xcol,
                            "n_dates": len(x), "rho": rho,
                            "loo_min": min(finite) if finite else np.nan,
                            "loo_max": max(finite) if finite else np.nan,
                            "loo_valid": len(finite)})
    assoc = pd.DataFrame(out)
    write_csv(assoc, "physical_result_association.csv")

    unique_daily = rows[rows.unique4].groupby(["date", "regime"], as_index=False).agg(
        rows=("excel_row", "size"), review_rows=("row_review_candidate", "sum"),
        indentation_rows=("indentation_candidate", "sum"),
        insufficient_rows=("insufficient_candidate", "sum"),
        crack_related_rows=("crack_related_candidate", "sum"))
    for col in ("review", "indentation", "insufficient", "crack_related"):
        unique_daily[col + "_share"] = unique_daily[col + "_rows"] / unique_daily.rows
    write_csv(unique_daily, "physical_daily_unique4.csv")

    lag3 = lag[lag.lag.eq(3)].copy()
    obs = lag.pivot_table(index=["date", "safe_segment"], columns="lag", values="same_rate")
    obs_peak = obs[3] - (obs[2] + obs[4]) / 2
    lag3 = lag3.set_index(["date", "safe_segment"]).join(obs_peak.rename("observed_peak3")).reset_index()
    cycle = pd.read_csv(OUT / "physical_cycle_probe.csv")
    local = pd.read_csv(OUT / "physical_cycle_local_windows.csv")
    assert set(cycle.date) == set(cfg["dates"])
    assert len(cycle[cycle.lag.eq(3)]) == 9
    assert len(local) > 0
    summary = {
        "status": "passed",
        "conditional_z_max_abs_error": float(conditional_error),
        "followup_code_sha256": {name: sha256(HERE / name) for name in (
            "followup_physical.py", "followup_cycle_probe.py",
            "followup_validate.py", "followup_figures.py")},
        "cycle_probe_sha256": sha256(OUT / "physical_cycle_probe.csv"),
        "local_cycle_windows": len(local),
        "local_peak3_and_phase3_windows": int(((local.lag3_peak > .1) &
                                               (local.phase3_adjusted_mi > .1)).sum()),
        "global_phase3_adjusted_mi_min": float(cycle.phase3_adjusted_mi.min()),
        "global_phase3_adjusted_mi_max": float(cycle.phase3_adjusted_mi.max()),
        "row_followup_sha256": sha256(OUT / "physical_row_followup.csv"),
        "rows": len(rows),
        "unique4_within_date": int(rows.unique4.sum()),
        "review_rows": int(rows.row_review_candidate.sum()),
        "unique_review_rows": int(rows.loc[rows.unique4, "row_review_candidate"].sum()),
        "candidate_unique_raw4_combinations": int(rows.loc[rows.row_review_candidate, ["F","I","V","t"]].drop_duplicates().shape[0]),
        "lag3_rate": float(lag[lag.lag.eq(3)].same.sum() / lag[lag.lag.eq(3)].pairs.sum()),
        "lag3_unique_endpoint_rate": float(lag[lag.lag.eq(3)].unique_endpoint_same.sum() / lag[lag.lag.eq(3)].unique_endpoint_pairs.sum()),
        "lag3_peak_above_run_shuffle_p95_segments": int((lag3.observed_peak3 > lag3.run_shuffle_peak3_p95).sum()),
        "lag3_tested_segments": len(lag3),
        "context_anchors": int(neighbor.anchor_excel_row.nunique()),
        "result_observed_dates": {name: int(daily[name].notna().sum()) for name in ("type1", "type2", "type3")},
    }
    (OUT / "physical_followup_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Observed lag3 peak by segment:")
    print(lag3[["date","safe_segment","observed_peak3","run_shuffle_peak3_p95"]].round(3).to_string(index=False))
    print("Result association:")
    print(assoc[(assoc.scope == "all") & assoc.metric.isin(
        ["indentation_rows_share","insufficient_rows_share","crack_related_rows_share"]
    )][["result_type","metric","n_dates","rho","loo_min","loo_max"]].round(3).to_string(index=False))

if __name__ == "__main__":
    main()