from __future__ import annotations

import json
import platform
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from common import (HERE, OUT, SHORT, align_labels, component_means_raw,
                    fit_model, load_inputs, raw_bic_aic, raw_loglik, read_config,
                    rows_to_daily, score_frame, sha256, train_percentile, write_csv)


def audit(raw: pd.DataFrame, quality: pd.DataFrame, unique_mask, controlled_mask):
    fields = read_config()["features"]
    rows = []
    for date, frame in raw.groupby("date", sort=True):
        row = {"date": date, "regime": frame["regime"].iloc[0],
               "raw_rows": len(frame),
               "unique4_rows": int(unique_mask[frame.index].sum()),
               "controlled100_rows": int(controlled_mask[frame.index].sum()),
               "high_force_rows": int((frame[fields[0]] > 3).sum()),
               "high_force_share": float((frame[fields[0]] > 3).mean())}
        for c, name in zip(fields, SHORT):
            x = frame[c]
            row.update({f"{name}_{metric}": value for metric, value in
                        {"median": x.median(), "iqr": x.quantile(.75)-x.quantile(.25),
                         "p10": x.quantile(.10), "p90": x.quantile(.90),
                         "min": x.min(), "max": x.max()}.items()})
        rows.append(row)
    write_csv(pd.DataFrame(rows), "daily_raw4.csv")
    write_csv(quality, "daily_quality.csv")


def build_splits(raw, config):
    rows = []
    overlap = []
    feature_cols = config["features"]
    for fold, (val, test) in enumerate(config["folds"], start=1):
        train_dates = [d for d in config["dates"] if d not in (val, test)]
        train = raw["date"].isin(train_dates)
        train_values = set(map(tuple, raw.loc[train, feature_cols].to_numpy()))
        for split, dates in (("train", train_dates), ("validation", [val]), ("test", [test])):
            part = raw[raw["date"].isin(dates)]
            for d in dates:
                frame = part[part["date"].eq(d)]
                rows.append({"phase": "selection", "fold": fold, "split": split, "date": d,
                             "regime": frame["regime"].iloc[0], "rows": len(frame)})
            if split != "train":
                values = list(map(tuple, part[feature_cols].to_numpy()))
                overlap.append({"fold": fold, "split": split, "date": dates[0],
                                "rows": len(values),
                                "train_exact4_rows": sum(v in train_values for v in values)})
    for fold, held in enumerate(config["dates"], start=1):
        for d in config["dates"]:
            frame = raw[raw["date"].eq(d)]
            rows.append({"phase": "lodo", "fold": fold,
                         "split": "held_out" if d == held else "train",
                         "date": d, "regime": frame["regime"].iloc[0], "rows": len(frame)})
    write_csv(pd.DataFrame(rows), "splits.csv")
    write_csv(pd.DataFrame(overlap), "split_overlap.csv")


def select_models(raw, config):
    x = raw[config["features"]].to_numpy(dtype=float)
    dates = raw["date"].to_numpy()
    rows, profiles = [], []
    total = len(config["folds"])*len(config["scalers"])*len(config["covariance_types"])*len(config["K"])*len(config["seeds"])
    done = 0
    for fold, (val, test) in enumerate(config["folds"], start=1):
        train_idx = (dates != val) & (dates != test)
        val_idx = dates == val
        xt, xv = x[train_idx], x[val_idx]
        for scaler_name in config["scalers"]:
            for cov in config["covariance_types"]:
                for k in config["K"]:
                    for seed in config["seeds"]:
                        row = {"fold": fold, "validation_date": val, "test_date": test,
                               "scaler": scaler_name, "covariance_type": cov,
                               "K": k, "seed": seed, "train_rows": len(xt),
                               "validation_rows": len(xv), "status": "ok",
                               "test_loglik": np.nan}
                        try:
                            with warnings.catch_warnings(record=True) as warning_list:
                                warnings.simplefilter("always", ConvergenceWarning)
                                scaler, model = fit_model(xt, scaler_name, k, cov,
                                                          seed, config["gmm"])
                            bic, aic = raw_bic_aic(model, scaler, xt)
                            row.update({"train_bic": bic, "train_aic": aic,
                                        "train_loglik": raw_loglik(model, scaler, xt).mean(),
                                        "val_loglik": raw_loglik(model, scaler, xv).mean(),
                                        "converged": model.converged_,
                                        "iterations": model.n_iter_,
                                        "min_weight": model.weights_.min(),
                                        "warnings": " | ".join(str(w.message) for w in warning_list)})
                            if k > 1:
                                try:
                                    z = scaler.transform(xt)
                                    row["silhouette_sample300"] = silhouette_score(
                                        z, model.predict(z), sample_size=min(300, len(z)),
                                        random_state=seed)
                                except ValueError:
                                    row["silhouette_sample300"] = np.nan
                            else:
                                row["silhouette_sample300"] = np.nan
                            means = component_means_raw(model, scaler)
                            for component in range(k):
                                profiles.append({"phase": "selection", "fold": fold,
                                                 "scaler": scaler_name, "covariance_type": cov,
                                                 "K": k, "seed": seed, "component": component,
                                                 "weight": model.weights_[component],
                                                 **{f"mean_{name}": means[component, j]
                                                    for j, name in enumerate(SHORT)}})
                            if not model.converged_:
                                row["status"] = "not_converged"
                        except Exception as exc:
                            row.update({"status": "failed", "error": repr(exc)})
                        rows.append(row)
                        done += 1
        print(f"selection fold {fold}/5 complete ({done}/{total})", flush=True)
        write_csv(pd.DataFrame(rows), "gmm_model_selection.csv")
    frame = pd.DataFrame(rows)
    write_csv(pd.DataFrame(profiles), "component_profiles.csv")
    ok = frame[(frame.status == "ok") & np.isfinite(frame.val_loglik)]
    grouped = ok.groupby(["scaler", "covariance_type", "K"]).agg(
        mean_val_loglik=("val_loglik", "mean"),
        mean_train_bic=("train_bic", "mean"),
        completed=("val_loglik", "size")).reset_index()
    needed = len(config["folds"])*len(config["seeds"])
    grouped["eligible"] = grouped["completed"].eq(needed)
    write_csv(grouped.sort_values("mean_val_loglik", ascending=False), "global_model_selection.csv")
    eligible = grouped[grouped.eligible].sort_values(
        ["mean_val_loglik", "mean_train_bic"], ascending=[False, True])
    if eligible.empty:
        raise RuntimeError("No complete GMM setting")
    selected = eligible.iloc[0][["scaler", "covariance_type", "K"]].to_dict()
    selected["K"] = int(selected["K"])
    fold_best = (ok.groupby(["fold", "scaler", "covariance_type", "K"], as_index=False)
                 ["val_loglik"].mean().sort_values("val_loglik", ascending=False)
                 .drop_duplicates("fold").sort_values("fold"))
    write_csv(fold_best, "fold_best_k.csv")
    return selected


def profile_row(phase, fold, date, scaler, model, config, mapping=None):
    means = component_means_raw(model, scaler)
    rows = []
    for k in range(model.n_components):
        rows.append({"phase": phase, "fold": fold, "date": date,
                     "component": int(mapping[k]) if mapping is not None else k,
                     "model_component": k, "weight": model.weights_[k],
                     **{f"mean_{name}": means[k, j] for j, name in enumerate(SHORT[:means.shape[1]])}})
    return rows


def summarize_component(frame: pd.DataFrame, k: int, phase: str):
    cols = [f"posterior_C{i}" for i in range(k)]
    rows = []
    for date, part in frame.groupby("date"):
        row = {"phase": phase, "date": date, "regime": part["regime"].iloc[0],
               "rows": len(part), "entropy_mean": part["entropy"].mean(),
               "posterior_max_mean": part["posterior_max"].mean()}
        hard = part["assigned_component"].value_counts(normalize=True)
        for i in range(k):
            row[f"C{i}_posterior_share"] = part[f"posterior_C{i}"].mean()
            row[f"C{i}_hard_share"] = hard.get(i, 0)
        row["dominant_component"] = int(np.argmax([row[f"C{i}_posterior_share"] for i in range(k)]))
        rows.append(row)
    return pd.DataFrame(rows)


def final_scoring(raw, config, selected):
    x = raw[config["features"]].to_numpy(float)
    dates = raw["date"].to_numpy()
    seed = config["final_seed"]
    scaler_name, cov, k = selected["scaler"], selected["covariance_type"], selected["K"]
    full_scaler, full_model = fit_model(x, scaler_name, k, cov, seed, config["gmm"])
    ref_means = component_means_raw(full_model, full_scaler)
    ref_scale = np.std(x, axis=0)
    OUT.mkdir(exist_ok=True)
    (OUT / "models_lodo").mkdir(exist_ok=True)
    joblib.dump({"scaler": full_scaler, "model": full_model},
                OUT / "model_full_descriptive.joblib")
    full = score_frame(full_model, full_scaler, x, raw)
    full["phase"] = "in_sample"
    write_csv(full, "in_sample_scores.csv")
    rows = []
    fit_params = []
    profile_rows = profile_row("in_sample", 0, "all", full_scaler, full_model, config)
    for fold, held in enumerate(config["dates"], start=1):
        ti = dates != held
        hi = dates == held
        scaler, model = fit_model(x[ti], scaler_name, k, cov, seed, config["gmm"])
        mapping = align_labels(ref_means, component_means_raw(model, scaler), ref_scale)
        held_frame = score_frame(model, scaler, x[hi], raw.loc[hi], mapping)
        train_frame = score_frame(model, scaler, x[ti], raw.loc[ti], mapping)
        held_frame["global_nll_train_percentile"] = train_percentile(
            train_frame["global_nll"].to_numpy(), held_frame["global_nll"].to_numpy())
        held_frame["mahalanobis_d2_train_percentile"] = train_percentile(
            train_frame["mahalanobis_d2"].to_numpy(), held_frame["mahalanobis_d2"].to_numpy())
        held_frame["phase"] = "lodo"
        held_frame["fold"] = fold
        held_frame["held_out_date"] = held
        rows.append(held_frame)
        fit_params.append({"fold": fold, "held_out_date": held, "scaler": scaler_name,
                           "covariance_type": cov, "K": k, "seed": seed,
                           "train_rows": int(ti.sum()), "held_out_rows": int(hi.sum()),
                           "train_dates": "|".join(d for d in config["dates"] if d != held),
                           **{f"center_{SHORT[j]}": scaler.center_[j] if hasattr(scaler, "center_")
                              else scaler.mean_[j] for j in range(4)},
                           **{f"scale_{SHORT[j]}": scaler.scale_[j] for j in range(4)}})
        profile_rows.extend(profile_row("lodo", fold, held, scaler, model, config, mapping))
        joblib.dump({"scaler": scaler, "model": model, "mapping": mapping,
                     "train_dates": [d for d in config["dates"] if d != held]},
                    OUT / "models_lodo" / f"{held}.joblib")
        print(f"LODO {fold}/9 {held}", flush=True)
    lodo = pd.concat(rows, ignore_index=True)
    if lodo["excel_row"].nunique() != len(raw):
        raise ValueError("LODO coverage not one per original row")
    write_csv(lodo, "lodo_row_scores.csv")
    write_csv(pd.DataFrame(fit_params), "lodo_fit_audit.csv")
    write_csv(pd.DataFrame(profile_rows), "selected_component_profiles.csv")
    daily_comp = pd.concat([summarize_component(lodo, k, "lodo"),
                            summarize_component(full, k, "in_sample")], ignore_index=True)
    write_csv(daily_comp, "daily_component_share.csv")
    fields = ["global_nll", "mahalanobis_d2", "posterior_max"]
    for name in ["global_nll_train_percentile", "mahalanobis_d2_train_percentile"]:
        if name in lodo:
            fields.append(name)
    daily = rows_to_daily(lodo, fields)
    daily["phase"] = "lodo"
    insample = rows_to_daily(full, ["global_nll", "mahalanobis_d2", "posterior_max"])
    insample["phase"] = "in_sample"
    write_csv(pd.concat([daily, insample], ignore_index=True), "daily_score_summary.csv")
    return lodo, full, full_scaler, full_model


def selected_folds(raw, config, selected):
    x = raw[config["features"]].to_numpy(float)
    dates = raw["date"].to_numpy()
    rows, stability = [], []
    for fold, (val, test) in enumerate(config["folds"], start=1):
        train = (dates != val) & (dates != test)
        model_runs = []
        for seed in config["seeds"]:
            scaler, model = fit_model(x[train], selected["scaler"], selected["K"],
                                      selected["covariance_type"], seed, config["gmm"])
            row = {"fold": fold, "seed": seed, "validation_date": val, "test_date": test,
                   "train_rows": int(train.sum()),
                   "val_loglik": raw_loglik(model, scaler, x[dates == val]).mean(),
                   "test_loglik": raw_loglik(model, scaler, x[dates == test]).mean(),
                   "converged": model.converged_}
            rows.append(row)
            model_runs.append((scaler, model))
        # Component label matching is tested through held-out prediction agreement.
        from sklearn.metrics import adjusted_rand_score
        target = x[dates == test]
        ref = model_runs[0][1].predict(model_runs[0][0].transform(target))
        for j in range(1, len(model_runs)):
            cur = model_runs[j][1].predict(model_runs[j][0].transform(target))
            stability.append({"fold": fold, "seed_ref": config["seeds"][0],
                              "seed_other": config["seeds"][j],
                              "test_date": test, "test_ari": adjusted_rand_score(ref, cur)})
    write_csv(pd.DataFrame(rows), "selected_fold_scores.csv")
    write_csv(pd.DataFrame(stability), "component_seed_stability.csv")


def main():
    config = read_config()
    with threadpool_limits(limits=config["threads"]):
        raw, quality, segments, controlled, unique, digest = load_inputs(config)
        OUT.mkdir(exist_ok=True)
        audit(raw, quality, unique, controlled)
        build_splits(raw, config)
        selected = select_models(raw, config)
        print("global selection", selected, flush=True)
        (OUT / "selected_config.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
        selected_folds(raw, config, selected)
        final_scoring(raw, config, selected)
    from postprocess import run_postprocess
    run_postprocess()
    manifest = {
        "experiment": "YSH-007",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": digest,
        "config_sha256": sha256(HERE / "config.json"),
        "plan_sha256": sha256(HERE / "실험계획.md"),
        "code_sha256": {p.name: sha256(p) for p in
                        [HERE / "common.py", HERE / "analyze.py", HERE / "postprocess.py"]},
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "selected_config": selected, "status": "analysis_complete_verification_pending"
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print("analysis complete", flush=True)


if __name__ == "__main__":
    main()

