from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from common import (HERE, OUT, SHORT, align_labels, apply_pool, component_means_raw,
                    d2_and_residual, fit_model, load_inputs, raw_loglik, read_config,
                    score_frame, upper_pool, write_csv)


def residual_fit(white: np.ndarray, k: int, seed: int, config: dict):
    args = config["residual"]
    return GaussianMixture(n_components=k, covariance_type="full", random_state=seed,
                           reg_covar=args["reg_covar"], max_iter=args["max_iter"],
                           n_init=args["n_init"]).fit(white)


def residual_profile(labels: np.ndarray, raw_res: np.ndarray, k: int):
    means, counts = [], []
    for i in range(k):
        part = raw_res[labels == i]
        counts.append(len(part))
        means.append(np.mean(part, axis=0) if len(part) else np.zeros(raw_res.shape[1]))
    return np.asarray(means), counts


def residual_select(raw, config, selected):
    x = raw[config["features"]].to_numpy(float)
    dates = raw["date"].to_numpy()
    rows, stability = [], []
    for fold, (val, test) in enumerate(config["folds"], start=1):
        ti, vi = (dates != val) & (dates != test), dates == val
        bs, bm = fit_model(x[ti], selected["scaler"], selected["K"],
                           selected["covariance_type"], config["final_seed"], config["gmm"])
        _, assigned_t, d2_t, white_t, _ = d2_and_residual(bm, bs, x[ti])
        _, assigned_v, d2_v, white_v, _ = d2_and_residual(bm, bs, x[vi])
        pool_t, thresholds = upper_pool(d2_t, assigned_t, "global")
        pool_v = apply_pool(d2_v, assigned_v, "global", thresholds)
        for k in config["residual"]["K"]:
            preds = []
            for seed in config["seeds"]:
                row = {"fold": fold, "validation_date": val, "K_error": k,
                       "seed": seed, "train_candidates": int(pool_t.sum()),
                       "val_candidates": int(pool_v.sum()), "status": "ok"}
                try:
                    em = residual_fit(white_t[pool_t], k, seed, config)
                    row["train_bic"] = em.bic(white_t[pool_t])
                    row["val_loglik"] = (em.score(white_v[pool_v]) if pool_v.sum()
                                         else np.nan)
                    row["converged"] = em.converged_
                    pred = em.predict(white_v[pool_v]) if pool_v.sum() else np.array([])
                    preds.append((seed, pred))
                    if not em.converged_:
                        row["status"] = "not_converged"
                except Exception as exc:
                    row.update({"status": "failed", "error": repr(exc),
                                "val_loglik": np.nan})
                rows.append(row)
            if len(preds) >= 2 and len(preds[0][1]) >= 2:
                for seed, pred in preds[1:]:
                    stability.append({"fold": fold, "K_error": k,
                                      "seed_ref": preds[0][0], "seed_other": seed,
                                      "val_ari": adjusted_rand_score(preds[0][1], pred)})
    frame = pd.DataFrame(rows)
    write_csv(frame, "residual_pattern_selection.csv")
    write_csv(pd.DataFrame(stability), "residual_seed_stability.csv")
    grouped = (frame[(frame.status == "ok") & frame.val_loglik.notna()]
               .groupby("K_error").agg(mean_val_loglik=("val_loglik", "mean"),
                                       mean_train_bic=("train_bic", "mean"),
                                       valid_runs=("val_loglik", "size"))
               .reset_index().sort_values(["valid_runs", "mean_val_loglik",
                                            "mean_train_bic"], ascending=[False, False, True]))
    write_csv(grouped, "residual_pattern_global_selection.csv")
    if grouped.empty:
        raise RuntimeError("No residual GMM candidate has validation support")
    chosen = int(grouped.iloc[0]["K_error"])
    print("residual K_error", chosen, "valid runs", int(grouped.iloc[0]["valid_runs"]), flush=True)
    return chosen


def residual_lodo(raw, config, selected, k_error):
    x = raw[config["features"]].to_numpy(float)
    dates = raw["date"].to_numpy()
    full_obj = joblib.load(OUT / "model_full_descriptive.joblib")
    fs, fm = full_obj["scaler"], full_obj["model"]
    _, fa, fd, fw, fr = d2_and_residual(fm, fs, x)
    fp, _ = upper_pool(fd, fa, "global")
    full_err = residual_fit(fw[fp], k_error, config["residual"]["seed"], config)
    full_profile, full_counts = residual_profile(full_err.predict(fw[fp]),
                                                  fr[fp], k_error)
    profiles, daily, rows_out = [], [], []
    for c in range(k_error):
        profiles.append({"phase": "reference_in_sample", "date": "all", "policy": "global",
                         "pattern": c, "support": full_counts[c],
                         **{f"residual_{SHORT[j]}": full_profile[c, j] for j in range(4)}})
    for fold, held in enumerate(config["dates"], start=1):
        ti, hi = dates != held, dates == held
        saved = joblib.load(OUT / "models_lodo" / f"{held}.joblib")
        scaler, model = saved["scaler"], saved["model"]
        _, ta, td, tw, tr = d2_and_residual(model, scaler, x[ti])
        _, ha, hd, hw, hr = d2_and_residual(model, scaler, x[hi])
        for policy in ("global", "component"):
            pool_t, thresholds = upper_pool(td, ta, policy)
            pool_h = apply_pool(hd, ha, policy, thresholds)
            if pool_t.sum() < max(30, k_error*10):
                daily.append({"date": held, "regime": raw.loc[hi, "regime"].iloc[0],
                              "fold": fold, "policy": policy, "status": "too_few_train",
                              "train_candidates": int(pool_t.sum()),
                              "held_candidates": int(pool_h.sum()),
                              "total_rows": int(hi.sum())})
                continue
            try:
                em = residual_fit(tw[pool_t], k_error, config["residual"]["seed"], config)
                labels_t = em.predict(tw[pool_t])
                train_profile, train_count = residual_profile(labels_t, tr[pool_t], k_error)
                mapping = align_labels(full_profile, train_profile,
                                       np.std(x, axis=0))
                for c in range(k_error):
                    profiles.append({"phase": "lodo", "date": held, "fold": fold,
                                     "policy": policy, "pattern": int(mapping[c]),
                                     "model_pattern": c, "support": train_count[c],
                                     "train_candidates": int(pool_t.sum()),
                                     "held_candidates": int(pool_h.sum()),
                                     **{f"residual_{SHORT[j]}": train_profile[c, j]
                                        for j in range(4)}})
                h_labels = mapping[em.predict(hw[pool_h])] if pool_h.any() else np.array([], dtype=int)
                row = {"date": held, "regime": raw.loc[hi, "regime"].iloc[0],
                       "fold": fold, "policy": policy, "status": "ok",
                       "train_candidates": int(pool_t.sum()),
                       "held_candidates": int(pool_h.sum()),
                       "total_rows": int(hi.sum())}
                for c in range(k_error):
                    count = int((h_labels == c).sum())
                    row[f"pattern_{c}_count"] = count
                    row[f"pattern_{c}_share_all"] = count / int(hi.sum())
                    row[f"pattern_{c}_share_candidates"] = (
                        count / int(pool_h.sum()) if pool_h.sum() else np.nan)
                daily.append(row)
                if pool_h.any():
                    details = raw.loc[hi, ["excel_row", "date", "regime"]].loc[pool_h].copy()
                    details["fold"] = fold
                    details["policy"] = policy
                    details["pattern"] = h_labels
                    details["mahalanobis_d2"] = hd[pool_h]
                    rows_out.append(details)
            except Exception as exc:
                daily.append({"date": held, "regime": raw.loc[hi, "regime"].iloc[0],
                              "fold": fold, "policy": policy, "status": "failed",
                              "error": repr(exc), "train_candidates": int(pool_t.sum()),
                              "held_candidates": int(pool_h.sum()),
                              "total_rows": int(hi.sum())})
    write_csv(pd.DataFrame(profiles), "residual_pattern_models.csv")
    frame = pd.DataFrame(daily)
    write_csv(frame, "daily_pattern_share.csv")
    write_csv(frame[frame.policy == "component"], "residual_candidate_sensitivity.csv")
    write_csv(pd.concat(rows_out, ignore_index=True) if rows_out else pd.DataFrame(),
              "lodo_pattern_rows.csv")
    return frame


def force_dominance(raw, config, selected):
    x = raw[config["features"]].to_numpy(float)
    xv = x[:, 1:]
    dates = raw["date"].to_numpy()
    k = selected["K"]
    fs, fm = fit_model(xv, selected["scaler"], k, selected["covariance_type"],
                       config["final_seed"], config["gmm"])
    ref = component_means_raw(fm, fs)
    scale = np.std(xv, axis=0)
    ivt_rows = []
    for held in config["dates"]:
        ti, hi = dates != held, dates == held
        ss, mm = fit_model(xv[ti], selected["scaler"], k, selected["covariance_type"],
                            config["final_seed"], config["gmm"])
        mapping = align_labels(ref, component_means_raw(mm, ss), scale)
        score = score_frame(mm, ss, xv[hi], raw.loc[hi], mapping)
        shares = [score[f"posterior_C{c}"].mean() for c in range(k)]
        ivt_rows.append({"date": held, "regime": score["regime"].iloc[0],
                         "feature_set": "IVt", "status": "ok",
                         **{f"C{c}_share": shares[c] for c in range(k)}})
    lodo = pd.read_csv(OUT / "daily_component_share.csv")
    raw4 = lodo[lodo.phase == "lodo"]
    for _, row in raw4.iterrows():
        ivt_rows.append({"date": row["date"], "regime": row["regime"],
                         "feature_set": "Raw4", "status": "ok",
                         **{f"C{c}_share": row[f"C{c}_posterior_share"] for c in range(k)}})
    frame = pd.DataFrame(ivt_rows)
    summary = []
    for fs_name, part in frame.groupby("feature_set"):
        a = part[part.regime == "A"][[f"C{c}_share" for c in range(k)]].mean().to_numpy(float)
        b = part[part.regime == "B"][[f"C{c}_share" for c in range(k)]].mean().to_numpy(float)
        summary.append({"feature_set": fs_name, "A_dates": int((part.regime == "A").sum()),
                        "B_dates": int((part.regime == "B").sum()),
                        "A_B_total_variation": float(np.abs(a-b).sum()/2)})
    write_csv(frame, "force_dominance.csv")
    write_csv(pd.DataFrame(summary), "force_dominance_summary.csv")


def repeat_sensitivity(raw, config, selected, controlled_mask, unique_mask):
    x = raw[config["features"]].to_numpy(float)
    dates = raw["date"].to_numpy()
    base = pd.read_csv(OUT / "daily_score_summary.csv")
    main = base[base.phase == "lodo"].set_index("date")
    results = []
    masks = {"unique4": unique_mask, "controlled100": controlled_mask}
    for condition, keep in masks.items():
        present = [d for d in config["dates"] if ((dates == d) & keep).any()]
        for held in present:
            hi = (dates == held) & keep
            ti = (dates != held) & keep
            train_dates = [d for d in present if d != held]
            row = {"condition": condition, "date": held, "regime":
                   raw.loc[hi, "regime"].iloc[0], "rows": int(hi.sum()),
                   "train_rows": int(ti.sum()), "train_dates": len(train_dates),
                   "train_high_force_rows": int((x[ti, 0] > 3).sum()),
                   "held_high_force_rows": int((x[hi, 0] > 3).sum())}
            if ti.sum() < max(100, selected["K"]*20) or len(train_dates) < 2:
                row["status"] = "insufficient_train"
                results.append(row)
                continue
            try:
                scaler, model = fit_model(x[ti], selected["scaler"], selected["K"],
                                           selected["covariance_type"],
                                           config["final_seed"], config["gmm"])
                scores = score_frame(model, scaler, x[hi], raw.loc[hi])
                row.update({"status": ("unsupported_high_force" if row["train_high_force_rows"] == 0 and row["held_high_force_rows"] > 0 else "ok"), "nll_p95": scores.global_nll.quantile(.95),
                            "d2_p95": scores.mahalanobis_d2.quantile(.95),
                            "main_nll_p95": main.loc[held, "global_nll_p95"],
                            "main_d2_p95": main.loc[held, "mahalanobis_d2_p95"]})
                train_values = set(map(tuple, x[ti]))
                row["exact4_train_share"] = sum(tuple(v) in train_values for v in x[hi])/int(hi.sum())
            except Exception as exc:
                row.update({"status": "failed", "error": repr(exc)})
            results.append(row)
        print("repeat", condition, "dates", len(present), flush=True)
    write_csv(pd.DataFrame(results), "repeat_sensitivity.csv")


def safe_spearman(x, y):
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(spearmanr(x, y).statistic)


def result_association(quality, config, k_error):
    scores = pd.read_csv(OUT / "daily_score_summary.csv")
    scores = scores[scores.phase == "lodo"]
    pattern = pd.read_csv(OUT / "daily_pattern_share.csv")
    pattern = pattern[(pattern.policy == "global") & (pattern.status == "ok")]
    merged = quality.merge(scores, on="date", how="left").merge(
        pattern.drop(columns=["regime"], errors="ignore"), on="date", how="left")
    metrics = ["global_nll_p95", "mahalanobis_d2_p95",
               "global_nll_train_percentile_p95",
               "mahalanobis_d2_train_percentile_p95"]
    metrics += [f"pattern_{c}_share_all" for c in range(k_error) if
                f"pattern_{c}_share_all" in merged]
    out = []
    for scenario in ("observed", "hypothetical_zero"):
        q = merged.copy()
        if scenario == "hypothetical_zero":
            q[["type1", "type2", "type3"]] = q[["type1", "type2", "type3"]].fillna(0)
        for scope in ("all", "A", "B"):
            d = q if scope == "all" else q[q.regime == scope]
            for defect in ("type1", "type2", "type3"):
                for target in ("count", "per1000"):
                    y = d[defect] if target == "count" else d[defect]*1000/d["raw_rows"]
                    for metric in metrics:
                        valid = y.notna() & d[metric].notna()
                        xx, yy = d.loc[valid, metric].to_numpy(float), y[valid].to_numpy(float)
                        rho = safe_spearman(xx, yy)
                        loo = [safe_spearman(np.delete(xx, i), np.delete(yy, i))
                               for i in range(len(xx))] if len(xx) >= 4 else []
                        finite = [v for v in loo if np.isfinite(v)]
                        out.append({"scenario": scenario, "scope": scope,
                                    "defect_type": defect, "target": target,
                                    "metric": metric, "n_dates": int(valid.sum()),
                                    "dates": "|".join(d.loc[valid, "date"]),
                                    "rho": rho,
                                    "loo_min": min(finite) if finite else np.nan,
                                    "loo_max": max(finite) if finite else np.nan,
                                    "loo_valid": len(finite)})
    write_csv(merged, "regime_confounding.csv")
    frame = pd.DataFrame(out)
    write_csv(frame[frame.scenario == "observed"], "result_association_main.csv")
    write_csv(frame[frame.scenario == "hypothetical_zero"],
              "result_association_sensitivity.csv")


def run_postprocess():
    config = read_config()
    raw, quality, segments, controlled, unique, digest = load_inputs(config)
    selected = json.loads((OUT / "selected_config.json").read_text(encoding="utf-8"))
    with threadpool_limits(limits=config["threads"]):
        k_error = residual_select(raw, config, selected)
        (OUT / "selected_residual.json").write_text(
            json.dumps({"K_error": k_error, "policy": "global", "seed": config["residual"]["seed"]},
                       indent=2), encoding="utf-8")
        residual_lodo(raw, config, selected, k_error)
        force_dominance(raw, config, selected)
        repeat_sensitivity(raw, config, selected, controlled, unique)
    result_association(quality, config, k_error)
    print("postprocess complete", flush=True)


if __name__ == "__main__":
    run_postprocess()



