from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal

from common import HERE, OUT, read_config, sha256

ROOT = HERE.parents[2]


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def run():
    cfg = read_config()
    raw_path = ROOT / "data" / "Welding_Data_Set_01.xlsx"
    check(sha256(raw_path) == cfg["raw_sha256"], "Raw digest")
    raw = pd.read_excel(raw_path, sheet_name="Raw data")
    result = pd.read_excel(raw_path, sheet_name="result")
    raw["excel_row"] = np.arange(len(raw)) + 2
    raw["date"] = pd.to_datetime(raw["working time"]).dt.strftime("%Y-%m-%d")
    x = raw[cfg["features"]].to_numpy(float)
    d = raw["date"].to_numpy()
    scored = pd.read_csv(OUT / "lodo_row_scores.csv")
    selection = pd.read_csv(OUT / "gmm_model_selection.csv")
    global_rank = pd.read_csv(OUT / "global_model_selection.csv")
    audit = pd.read_csv(OUT / "lodo_fit_audit.csv")
    patterns = pd.read_csv(OUT / "daily_pattern_share.csv")
    selected = json.loads((OUT / "selected_config.json").read_text())
    check(len(raw) == len(scored) == 11939, "LODO row count")
    check(scored.excel_row.is_unique, "Each Raw row scored once")
    check(np.array_equal(np.sort(scored.excel_row), raw.excel_row), "LODO coverage")
    check(len(selection) == 360 and (selection.status == "ok").all(), "Selection grid")
    expected = selection.groupby(["scaler", "covariance_type", "K"]).val_loglik.mean()
    for r in global_rank.itertuples():
        actual = expected.loc[r.scaler, r.covariance_type, r.K]
        check(abs(actual-r.mean_val_loglik) < 1e-9, "Global equal-fold selection")
    top = global_rank.sort_values(["mean_val_loglik", "mean_train_bic"],
                                  ascending=[False, True]).iloc[0]
    check((top.scaler, top.covariance_type, int(top.K)) ==
          (selected["scaler"], selected["covariance_type"], selected["K"]),
          "Selected setting")
    check(len(audit) == 9, "Nine LODO fits")
    overlap = {}
    full_payload = joblib.load(OUT / "model_full_descriptive.joblib")
    reference = full_payload["scaler"].inverse_transform(full_payload["model"].means_)
    reference_scale = x.std(axis=0)
    density_error = 0.0
    d2_error = 0.0
    for held in cfg["dates"]:
        train = d != held
        test = d == held
        fit = audit[audit.held_out_date == held].iloc[0]
        check(int(fit.train_rows) == int(train.sum()) and
              int(fit.held_out_rows) == int(test.sum()), "LODO split counts")
        check(set(fit.train_dates.split("|")) == set(d[train]), "LODO train dates")
        part = scored[scored.date == held]
        check(len(part) == test.sum() and (part.held_out_date == held).all(),
              "Held-out scoring only")
        payload = joblib.load(OUT / "models_lodo" / f"{held}.joblib")
        scaler, model = payload["scaler"], payload["model"]
        check(set(payload["train_dates"]) == set(d[train]), "Saved model dates")
        check(model.n_components == selected["K"], "Frozen K")
        candidate = scaler.inverse_transform(model.means_)
        cost = np.linalg.norm((candidate[:, None, :] - reference[None, :, :]) /
                              reference_scale[None, None, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        check(np.array_equal(payload["mapping"][rows], cols),
              "Independent component alignment")
        check(model.covariance_type == selected["covariance_type"], "Frozen covariance")
        expected_mean = x[train].mean(axis=0)
        expected_scale = x[train].std(axis=0)
        check(np.allclose(scaler.mean_, expected_mean, rtol=0, atol=1e-10),
              "Scaler fitted on eight training dates")
        check(np.allclose(scaler.scale_, expected_scale, rtol=0, atol=1e-10),
              "Scaler scale on eight training dates")
        train_values = set(map(tuple, x[train]))
        overlap[held] = {
            "held_rows": int(test.sum()),
            "exact4_seen_in_train": int(sum(tuple(row) in train_values for row in x[test])),
        }
        sample = part.sort_values("excel_row").iloc[np.linspace(
            0, len(part)-1, min(20, len(part)), dtype=int)]
        idx = sample.excel_row.to_numpy(int)-2
        scale = np.diag(scaler.scale_)
        raw_means = scaler.inverse_transform(model.means_)
        raw_cov = np.einsum("ij,kjl,lm->kim", scale, model.covariances_, scale)
        log_terms = np.column_stack([
            np.log(model.weights_[k]) +
            multivariate_normal.logpdf(x[idx], mean=raw_means[k], cov=raw_cov[k])
            for k in range(model.n_components)])
        direct_nll = -logsumexp(log_terms, axis=1)
        density_error = max(density_error, float(np.max(np.abs(
            direct_nll-sample.global_nll.to_numpy()))))
        posterior = np.exp(log_terms-logsumexp(log_terms, axis=1)[:, None])
        assigned = posterior.argmax(axis=1)
        direct_d2 = np.array([
            (x[i]-raw_means[k]) @ np.linalg.solve(raw_cov[k], x[i]-raw_means[k])
            for i, k in zip(idx, assigned)])
        d2_error = max(d2_error, float(np.max(np.abs(
            direct_d2-sample.mahalanobis_d2.to_numpy()))))
        check(np.allclose(sample[[f"posterior_C{k}" for k in range(model.n_components)]
                          ].sum(axis=1), 1, atol=1e-8), "Posterior sums")
    check(density_error < 1e-5, "Independent raw density")
    check(d2_error < 1e-5, "Independent Mahalanobis d2")
    for r in patterns.itertuples():
        check(sum(getattr(r, f"pattern_{k}_count") for k in range(5)) ==
              r.held_candidates, "Residual candidate coverage")
    check(len(patterns) == 18, "Both residual policies on all dates")
    result["date"] = pd.to_datetime(result["working time"]).dt.strftime("%Y-%m-%d")
    check(result.loc[result.date == "2020-03-27", "defect"].empty,
          "March 27 Result missing rows preserved")
    check(result[(result.date == "2020-03-31") &
                 (result["defect type"] == 3)].empty,
          "March 31 type3 missing row preserved")
    check(result[(result.date == "2020-04-07") &
                 (result["defect type"] == 1)]["defect"].iloc[0] == 0,
          "April 7 explicit zero preserved")
    report = {
        "status": "passed",
        "raw_sha256": cfg["raw_sha256"],
        "selection_rows": len(selection),
        "lodo_rows": len(scored),
        "lodo_dates": len(audit),
        "max_density_abs_error": density_error,
        "max_mahalanobis_d2_abs_error": d2_error,
        "lodo_exact4_overlap": overlap,
        "result_missingness_checked": True,
        "residual_policies_checked": True,
        "component_alignment_checked": True,
        "segments_sha256": sha256(HERE.parent / "ysh-003" / "outputs" /
                                  "tables" / "segments.csv"),
        "key_output_sha256": {name: sha256(OUT / name) for name in (
            "gmm_model_selection.csv", "lodo_row_scores.csv",
            "daily_pattern_share.csv", "result_association_main.csv")},
    }
    (OUT / "verification.json").write_text(json.dumps(report, indent=2),
                                            encoding="utf-8")
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "plan_sha256_pre_execution" not in manifest:
        manifest["plan_sha256_pre_execution"] = manifest["plan_sha256"]
    manifest["plan_sha256_current"] = sha256(HERE / "실험계획.md")
    manifest["result_sha256"] = sha256(HERE / "실험결과.md")
    notebook_path = HERE / "review_ysh007.ipynb"
    if notebook_path.is_file():
        manifest["review_notebook_sha256"] = sha256(notebook_path)
    manifest["requirements_lock_sha256"] = sha256(ROOT / "requirements.txt")
    manifest["status"] = "verified"
    manifest["code_sha256"]["verify.py"] = sha256(HERE / "verify.py")
    manifest["code_sha256"]["report.py"] = sha256(HERE / "report.py")
    manifest["verification_sha256"] = sha256(OUT / "verification.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()






