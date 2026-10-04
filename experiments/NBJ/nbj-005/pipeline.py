"""Frozen NBJ-004 K-Means/IF bundle, inference, and NBJ-005 verification."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "4"

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplcache"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ZIP = Path(r"C:\Users\skqja\Downloads\2. 용접기 AI 데이터셋 (1).zip")
MEMBER = "2. 용접기 AI 데이터셋/Welding Data Set_01.xlsx"
ZIP_HASH = "54a41de1e3d0c60cf47d66d707e1e8ff78dfd203b5024b478f9113c5a4435873"
EXCEL_HASH = "d514d6aaa121630c04d7d51c97a56025e78e2c86868813f7722ba5db922c1f33"
FEATURES = ("weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)")
OUTPUT_FEATURES = ("force_bar", "current_kA", "voltage_V", "weld_time_ms")
THRESHOLD = 0.6079581703468012
THRESHOLD99_METADATA = 0.6681064037197512
RTOL = 1e-10
ATOL = 1e-12
BASE = HERE.parent / "nbj-004" / "outputs"
MODEL_PATH = OUT / "models" / "pipeline.joblib"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_raw() -> tuple[pd.DataFrame, dict]:
    zipped = ZIP.read_bytes()
    if sha256(zipped) != ZIP_HASH:
        raise ValueError("ZIP hash differs from NBJ-004")
    with zipfile.ZipFile(io.BytesIO(zipped)) as archive:
        excel = archive.read(MEMBER)
    if sha256(excel) != EXCEL_HASH:
        raise ValueError("Excel hash differs from NBJ-004")
    raw = pd.read_excel(io.BytesIO(excel), sheet_name="Raw data")
    if raw.shape != (11939, 10) or tuple(raw.columns[6:]) != FEATURES:
        raise ValueError("Raw shape or sensor columns differ from NBJ-004")
    raw["excel_row"] = np.arange(2, len(raw) + 2)
    raw["date"] = pd.to_datetime(raw["working time"]).dt.strftime("%Y-%m-%d")
    raw["train"] = raw["date"] <= "2020-03-31"
    if (raw.train.sum(), (~raw.train).sum(), raw.date.nunique()) != (8470, 3469, 9):
        raise ValueError("Train/follow-up split differs from NBJ-004")
    if raw.loc[:, list(FEATURES)].isna().any().any() or not np.isfinite(raw.loc[:, list(FEATURES)].to_numpy(float)).all():
        raise ValueError("Invalid original sensor values")
    return raw, {"zip_path": str(ZIP), "zip_sha256": ZIP_HASH, "zip_member": MEMBER,
                 "excel_sha256": EXCEL_HASH, "sheet": "Raw data"}


def fit_bundle(raw: pd.DataFrame) -> tuple[dict, dict]:
    x = raw.loc[:, list(FEATURES)].to_numpy(float)
    train = raw.train.to_numpy(bool)
    x_train = x[train]
    k_scaler = StandardScaler().fit(x_train)
    if_scaler = StandardScaler().fit(x_train)
    k_model = KMeans(n_clusters=2, init="k-means++", n_init=20, max_iter=300,
                     tol=1e-4, algorithm="lloyd", random_state=42).fit(k_scaler.transform(x_train))
    if_model = IsolationForest(n_estimators=300, max_samples=min(256, len(x_train)),
                               max_features=1.0, bootstrap=False, contamination="auto",
                               random_state=42, n_jobs=4).fit(if_scaler.transform(x_train))
    train_scores = -if_model.score_samples(if_scaler.transform(x_train))
    recalculated = float(np.quantile(train_scores, .95, method="linear"))
    bundle = {
        "feature_names": FEATURES,
        "output_feature_names": OUTPUT_FEATURES,
        "feature_units": ("bar", "kA", "V", "ms"),
        "k_scaler": k_scaler,
        "k_model": k_model,
        "if_scaler": if_scaler,
        "if_model": if_model,
        "if_threshold": THRESHOLD,
        "if_threshold_operator": ">=",
        "if_threshold99_metadata": THRESHOLD99_METADATA,
        "train_scores_sorted": np.sort(train_scores),
        "regime_status": "provisional_elbow_conflict",
        "seed": 42,
    }
    return bundle, {"recalculated_threshold95": recalculated, "train_reference_n": len(train_scores),
                    "train_inertia": float(k_model.inertia_)}


def _validated_frame(records: pd.DataFrame | list[dict] | dict, bundle: dict) -> pd.DataFrame:
    if isinstance(records, dict):
        records = [records]
    if isinstance(records, list):
        if not records or any(not isinstance(record, dict) for record in records):
            raise ValueError("Input must contain named sensor records")
        frame = pd.DataFrame(records)
    elif isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        raise TypeError("Input must be a named record, list of named records, or DataFrame")
    if frame.empty:
        raise ValueError("Input is empty")
    required = set(bundle["feature_names"])
    allowed = required | {"input_id", "date"}
    if not required.issubset(frame.columns) or set(frame.columns) - allowed:
        raise ValueError(f"Expected sensor names {bundle['feature_names']} plus input_id/date metadata")
    if "input_id" not in frame or frame.input_id.isna().any():
        raise ValueError("input_id is required")
    values = frame.loc[:, list(bundle["feature_names"])]
    if any(not pd.api.types.is_numeric_dtype(values[col]) for col in values):
        raise ValueError("Sensor values must be numeric in the declared units")
    x = values.to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("Sensor values must be finite")
    return frame


def infer(bundle: dict, records: pd.DataFrame | list[dict] | dict) -> pd.DataFrame:
    """Run frozen inference on named bar/kA/V/ms fields; never fit during inference."""
    frame = _validated_frame(records, bundle)
    x = frame.loc[:, list(bundle["feature_names"])].to_numpy(float)
    kz = bundle["k_scaler"].transform(x)
    cluster = bundle["k_model"].predict(kz)
    distance = np.linalg.norm(kz - bundle["k_model"].cluster_centers_[cluster], axis=1)
    score = -bundle["if_model"].score_samples(bundle["if_scaler"].transform(x))
    ref = bundle["train_scores_sorted"]
    percentile = 100.0 * np.searchsorted(ref, score, side="right") / len(ref)
    result = pd.DataFrame({"input_id": frame.input_id.to_numpy(),
                           "date": frame.date.to_numpy() if "date" in frame else [None] * len(frame)})
    for source, destination in zip(bundle["feature_names"], bundle["output_feature_names"]):
        result[destination] = frame[source].to_numpy()
    result["cluster_id"] = cluster
    result["centroid_distance_scaled"] = distance
    result["if_anomaly_score"] = score
    result["train_anomaly_percentile"] = percentile
    result["candidate_flag"] = score >= bundle["if_threshold"]
    return result


def check_close(name: str, new: np.ndarray, old: np.ndarray, checks: dict,
                *, exact: bool = False) -> None:
    new, old = np.asarray(new), np.asarray(old)
    if exact:
        equal = np.array_equal(new, old)
        maximum = None
        mismatch = int(np.count_nonzero(new != old)) if new.shape == old.shape else None
    else:
        equal = bool(np.allclose(new, old, rtol=RTOL, atol=ATOL, equal_nan=False))
        maximum = float(np.max(np.abs(new - old))) if new.size and new.shape == old.shape else None
        mismatch = int(np.count_nonzero(~np.isclose(new, old, rtol=RTOL, atol=ATOL))) if new.shape == old.shape else None
    checks[name] = {"pass": bool(equal), "mismatch_n": mismatch, "max_abs_difference": maximum}


def verify(bundle: dict, reloaded: dict, raw: pd.DataFrame, fit_info: dict) -> tuple[pd.DataFrame, dict]:
    checks: dict = {}
    baseline_k = pd.read_csv(BASE / "row_clusters.csv")
    baseline_if = pd.read_csv(BASE / "if_row_scores.csv")
    baseline_k_diag = pd.read_csv(BASE / "k_diagnostics.csv")
    baseline_daily = pd.read_csv(BASE / "daily_clusters.csv")
    baseline_if_daily = pd.read_csv(BASE / "if_daily.csv")
    baseline_manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    rows = raw.loc[:, list(FEATURES)].copy()
    rows.insert(0, "date", raw.date.to_numpy())
    rows.insert(0, "input_id", [f"{EXCEL_HASH}:Raw data:{r}" for r in raw.excel_row])
    before = infer(bundle, rows)
    repeated = infer(bundle, rows)
    after = infer(reloaded, rows)
    check_close("repeat_all_fields_exact", before.to_numpy(), repeated.to_numpy(), checks, exact=True)
    check_close("reload_all_fields_exact", before.to_numpy(), after.to_numpy(), checks, exact=True)
    for key in ("feature_names", "output_feature_names", "feature_units", "if_threshold", "if_threshold_operator", "seed", "regime_status"):
        check_close(f"reload_{key}_exact", np.asarray([str(bundle[key])]), np.asarray([str(reloaded[key])]), checks, exact=True)
    for key in ("k_scaler", "if_scaler"):
        for attr in ("mean_", "scale_", "var_", "n_samples_seen_"):
            check_close(f"reload_{key}_{attr}_exact", getattr(bundle[key], attr), getattr(reloaded[key], attr), checks, exact=True)
    check_close("reload_centers_exact", bundle["k_model"].cluster_centers_, reloaded["k_model"].cluster_centers_, checks, exact=True)
    check_close("reload_train_reference_exact", bundle["train_scores_sorted"], reloaded["train_scores_sorted"], checks, exact=True)
    check_close("split_excel_row_exact", raw.excel_row, baseline_k.excel_row, checks, exact=True)
    check_close("split_date_exact", raw.date, baseline_k.date, checks, exact=True)
    check_close("split_train_exact", raw.train, baseline_k.train, checks, exact=True)
    check_close("if_row_key_exact", raw.excel_row, baseline_if.excel_row, checks, exact=True)
    check_close("if_date_exact", raw.date, baseline_if.date, checks, exact=True)
    check_close("if_train_exact", raw.train, baseline_if.train, checks, exact=True)
    check_close("cluster_id_exact", before.cluster_id, baseline_k.cluster, checks, exact=True)
    check_close("centroid_distance_csv_tolerance", before.centroid_distance_scaled, baseline_k.distance_to_center_scaled, checks)
    check_close("if_score_csv_tolerance", before.if_anomaly_score, baseline_if.if_42_score, checks)
    check_close("candidate_exact", before.candidate_flag, baseline_if.if_42_flag95, checks, exact=True)
    check_close("inertia_csv_tolerance", [fit_info["train_inertia"]],
                baseline_k_diag.loc[baseline_k_diag.K.eq(2), "inertia"], checks)
    check_close("threshold_manifest_tolerance", [fit_info["recalculated_threshold95"]],
                [baseline_manifest["IF"]["thresholds"][0]["threshold95"]], checks)
    check_close("threshold_fixed_exact", [bundle["if_threshold"]], [THRESHOLD], checks, exact=True)
    x_train = raw.loc[raw.train, list(FEATURES)].to_numpy(float)
    for key in ("k_scaler", "if_scaler"):
        check_close(f"{key}_train_mean", bundle[key].mean_, x_train.mean(axis=0), checks)
        check_close(f"{key}_train_scale", bundle[key].scale_, x_train.std(axis=0), checks)
        check_close(f"{key}_train_n_exact", [bundle[key].n_samples_seen_], [len(x_train)], checks, exact=True)
    check_close("scalers_same_mean_exact", bundle["k_scaler"].mean_, bundle["if_scaler"].mean_, checks, exact=True)
    check_close("scalers_same_scale_exact", bundle["k_scaler"].scale_, bundle["if_scaler"].scale_, checks, exact=True)
    check_close("train_reference_all_scores_exact", bundle["train_scores_sorted"],
                np.sort(before.loc[raw.train, "if_anomaly_score"].to_numpy()), checks, exact=True)
    shuffled = rows[["input_id", "date", FEATURES[3], FEATURES[2], FEATURES[1], FEATURES[0]]]
    check_close("shuffled_named_features_exact", before.to_numpy(), infer(bundle, shuffled).to_numpy(), checks, exact=True)
    for name, bad in (("missing_feature", rows.drop(columns=[FEATURES[0]]).iloc[:1]),
                      ("ambiguous_positional", [1, 2, 3, 4]),
                      ("nonfinite", rows.iloc[:1].assign(**{FEATURES[0]: np.nan}))):
        try:
            infer(bundle, bad)
            rejected = False
        except (TypeError, ValueError):
            rejected = True
        checks[name] = {"pass": rejected, "mismatch_n": 0 if rejected else 1, "max_abs_difference": None}
    scores = before.if_anomaly_score.to_numpy()
    threshold = bundle["if_threshold"]
    below = np.flatnonzero(scores < threshold)
    same = np.flatnonzero(scores == threshold)
    above = np.flatnonzero(scores > threshold)
    chosen = [0]
    cases = {"first_excel_row": int(raw.excel_row.iloc[0]),
             "equal_threshold": "no_observed_row" if not len(same) else int(raw.excel_row.iloc[same[0]])}
    for label, locations, direction in (("below", below, -1), ("above", above, 1)):
        if len(locations):
            local = locations[np.argmax(scores[locations])] if direction < 0 else locations[np.argmin(scores[locations])]
            chosen.append(int(local))
            cases[f"{label}_excel_row"] = int(raw.excel_row.iloc[local])
    if len(same):
        chosen.append(int(same[0]))
    for index in dict.fromkeys(chosen):
        single = infer(bundle, rows.iloc[[index]])
        old = before.iloc[[index]]
        discrete = ["input_id", "date", "cluster_id", "candidate_flag"] + list(OUTPUT_FEATURES)
        check_close(f"single_batch_row_{int(raw.excel_row.iloc[index])}_discrete", single[discrete].to_numpy(),
                    old[discrete].to_numpy(), checks, exact=True)
        for field in ("centroid_distance_scaled", "if_anomaly_score", "train_anomaly_percentile"):
            check_close(f"single_batch_row_{int(raw.excel_row.iloc[index])}_{field}", single[field], old[field], checks)
    ordered = np.argsort(scores, kind="stable")
    percent = before.train_anomaly_percentile.to_numpy()
    checks["percentile_nondecreasing"] = {"pass": bool(np.all(np.diff(percent[ordered]) >= 0)), "mismatch_n": None, "max_abs_difference": None}
    expected_percent = 100.0 * np.searchsorted(bundle["train_scores_sorted"], scores, side="right") / len(x_train)
    check_close("percentile_definition_exact", percent, expected_percent, checks, exact=True)
    checks["percentile_tie_consistent"] = {"pass": bool(np.all(percent[ordered][1:][np.diff(scores[ordered]) == 0] == percent[ordered][:-1][np.diff(scores[ordered]) == 0])), "mismatch_n": None, "max_abs_difference": None}
    checks["percentile_bounds"] = {"pass": bool(np.all((percent >= 0) & (percent <= 100))), "mismatch_n": None, "max_abs_difference": None}
    checks["percentile_reference_unchanged"] = {"pass": bool(np.array_equal(bundle["train_scores_sorted"], reloaded["train_scores_sorted"])), "mismatch_n": None, "max_abs_difference": None}
    checks["train_followup_counts"] = {"pass": bool((raw.train.sum(), (~raw.train).sum()) == (8470, 3469)), "mismatch_n": None, "max_abs_difference": None}
    table = before.copy()
    table["excel_row"] = raw.excel_row.to_numpy()
    table["split"] = np.where(raw.train.to_numpy(), "train", "follow-up")
    cluster_daily = table.groupby(["date", "cluster_id"], as_index=False).size().rename(columns={"size": "n"})
    matched = cluster_daily.merge(baseline_daily, left_on=["date", "cluster_id"], right_on=["date", "cluster"], how="outer", indicator=True)
    checks["daily_cluster_counts_exact"] = {"pass": bool(matched._merge.eq("both").all() and matched.n_x.eq(matched.n_y).all()), "mismatch_n": None, "max_abs_difference": None}
    daily_if = table.groupby("date").candidate_flag.sum()
    check_close("daily_if_candidates_exact", daily_if.to_numpy(), baseline_if_daily.set_index("date").loc[daily_if.index, "flagged"].to_numpy(), checks, exact=True)
    checks["cluster_counts"] = {split: table.loc[table.split.eq(split)].cluster_id.value_counts().sort_index().to_dict() for split in ("train", "follow-up")}
    checks["if_candidate_counts"] = {split: int(table.loc[table.split.eq(split), "candidate_flag"].sum()) for split in ("train", "follow-up")}
    checks["single_row_cases"] = cases
    return table, checks


def charts(table: pd.DataFrame, threshold: float) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, col, label in zip(axes.flat, OUTPUT_FEATURES, ("Force (bar)", "Current (kA)", "Voltage (V)", "Weld Time (ms)")):
        data = [table.loc[table.cluster_id.eq(cluster), col] for cluster in (0, 1)]
        ax.boxplot(data, tick_labels=["Cluster 0", "Cluster 1"], showfliers=False)
        ax.set_title(label)
        ax.set_ylabel(label)
    fig.suptitle("Sensor values by provisional cluster")
    fig.savefig(OUT / "cluster_sensor_distribution.png", dpi=140)
    plt.close(fig)
    daily = table.groupby(["date", "cluster_id"]).size().unstack(fill_value=0)
    share = daily.div(daily.sum(axis=1), axis=0) * 100
    ax = share.plot(kind="bar", stacked=True, figsize=(11, 5), color=["#377eb8", "#e69f00"])
    ax.set_title("Daily provisional cluster shares")
    ax.set_xlabel("Date")
    ax.set_ylabel("Share of records (%)")
    ax.legend(title="Cluster")
    ax.figure.tight_layout()
    ax.figure.savefig(OUT / "daily_cluster_share.png", dpi=140)
    plt.close(ax.figure)
    fig, ax = plt.subplots(figsize=(9, 5))
    for split, color in (("train", "#377eb8"), ("follow-up", "#e69f00")):
        ax.hist(table.loc[table.split.eq(split), "if_anomaly_score"], bins=45, alpha=.55, label=split, color=color)
    ax.axvline(threshold, color="black", linestyle="--", label=f"Fixed threshold {threshold:.8f}")
    ax.set(title="Isolation Forest anomaly score distribution", xlabel="IF anomaly score (-score_samples, unitless)", ylabel="Records")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "if_score_distribution.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([table.loc[table.cluster_id.eq(cluster), "if_anomaly_score"] for cluster in (0, 1)],
               tick_labels=["Cluster 0", "Cluster 1"], showfliers=False)
    ax.axhline(threshold, color="black", linestyle="--", label="Fixed threshold")
    ax.set(title="Isolation Forest score by provisional cluster", xlabel="Cluster", ylabel="IF anomaly score (unitless)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "if_score_by_cluster.png", dpi=140)
    plt.close(fig)


def execute() -> None:
    raw, source = load_raw()
    bundle, fit_info = fit_bundle(raw)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    reloaded = joblib.load(MODEL_PATH)
    table, checks = verify(bundle, reloaded, raw, fit_info)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "inference_rows.csv", index=False, encoding="utf-8-sig", float_format="%.15g")
    table.groupby(["split", "cluster_id"], as_index=False).agg(records=("input_id", "size"), candidates=("candidate_flag", "sum"), score_median=("if_anomaly_score", "median")).to_csv(OUT / "split_cluster_summary.csv", index=False, encoding="utf-8-sig")
    table.groupby(["date", "cluster_id"], as_index=False).agg(records=("input_id", "size"), candidates=("candidate_flag", "sum"), score_median=("if_anomaly_score", "median")).to_csv(OUT / "daily_summary.csv", index=False, encoding="utf-8-sig")
    charts(table, THRESHOLD)
    preflight = json.loads((OUT / "preflight.json").read_text(encoding="utf-8"))
    if not preflight["pass"] or preflight["sys_executable"] != sys.executable:
        raise RuntimeError("Recorded preflight does not match the running interpreter")
    metadata = {"source": source, "run_command": f'"{sys.executable}" -B experiments/NBJ/nbj-005/pipeline.py run',
                "python": sys.version, "python_executable": sys.executable,
                "pythonpath": os.environ.get("PYTHONPATH"), "preflight": preflight,
                "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                "joblib": joblib.__version__, "matplotlib": matplotlib.__version__, "threads": 4,
                "feature_order": FEATURES, "feature_units": bundle["feature_units"], "train_rows": int(raw.train.sum()),
                "followup_rows": int((~raw.train).sum()), "train_cutoff": "2020-03-31", "seed": 42,
                "kmeans": bundle["k_model"].get_params(), "isolation_forest": bundle["if_model"].get_params(),
                "threshold": THRESHOLD, "threshold99_metadata": THRESHOLD99_METADATA,
                "threshold_definition": "train score 95% linear quantile; candidate if score >= frozen NBJ-004 threshold",
                "score_definition": "-score_samples(IF scaler transform(X))",
                "distance_definition": "Euclidean distance to assigned K-Means center in standardized space",
                "percentile_definition": "100 * count(sorted train scores <= score) / train_n; side=right",
                "regime_status": bundle["regime_status"], "fit": fit_info,
                "bundle_sha256": sha256(MODEL_PATH.read_bytes()), "code_sha256": sha256(Path(__file__).read_bytes()),
                "verification_passed": all(item["pass"] for item in checks.values() if isinstance(item, dict) and "pass" in item)}
    (MODEL_PATH.parent / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT / "verification.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    example = table.iloc[0].drop(labels=["excel_row", "split"]).to_dict()
    print(json.dumps({"verification_passed": metadata["verification_passed"], "failed_checks": [name for name, info in checks.items() if isinstance(info, dict) and info.get("pass") is False], "example": example, "fit": fit_info}, ensure_ascii=False, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "infer"))
    parser.add_argument("--input-json", help="For infer: named input record/list JSON, or path to JSON file")
    args = parser.parse_args()
    if args.command == "run":
        execute()
    else:
        if args.input_json is None:
            parser.error("infer requires --input-json")
        supplied = args.input_json.strip()
        payload = json.loads(supplied if supplied.startswith(("[", "{")) else Path(supplied).read_text(encoding="utf-8"))
        print(infer(joblib.load(MODEL_PATH), payload).to_json(orient="records", force_ascii=False))


if __name__ == "__main__":
    main()
