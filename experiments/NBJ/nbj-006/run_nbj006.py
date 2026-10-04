"""NBJ-006 notebook code cells in their original order.
Generated mechanically; only IPython display import is replaced by stdout output.
Run from the repository root with the NBJ-005 interpreter and PYTHONPATH.
This file has been syntax-checked, not executed.
Notebook SHA-256: 52cf061ef112f3276e8f3f4704cdc3a753b83d55c9d6fd8e88dc4f5e5866a581
"""

# ## 1. Imports / environment
# source_sha256: 90fa2ef0f2f6a12923136dd167d6a8cea5b4a4c4ac4e85a22fe602786dea0247
# NBJ006_BEGIN_CELL 2
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "4"
import hashlib, io, json, sys, zipfile
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, silhouette_score
def display(obj):
    """Print an object without changing its data."""
    if hasattr(obj, "to_string"):
        print(obj.to_string())
    else:
        print(obj)

ROOT = Path.cwd()
if not (ROOT / "experiments" / "NBJ").is_dir():
    raise RuntimeError("저장소 루트에서 Notebook을 실행하세요.")
HERE = ROOT / "experiments" / "NBJ" / "nbj-006"
BASE = ROOT / "experiments" / "NBJ" / "nbj-005"
META_PATH = BASE / "outputs" / "models" / "metadata.json"
MODEL_PATH = BASE / "outputs" / "models" / "pipeline.joblib"
BASE_ROWS = BASE / "outputs" / "inference_rows.csv"
QUALITY_PATH = ROOT / "experiments" / "NBJ" / "nbj-004" / "outputs" / "quality_daily.csv"
SEEDS = [0, 1, 2, 3, 42]
RTOL, ATOL = 1e-10, 1e-12
print({"python": sys.version, "executable": sys.executable, "pythonpath": os.environ.get("PYTHONPATH"),
       "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__})
# NBJ006_END_CELL 2

# ## 2. NBJ-005 metadata 읽기
# source_sha256: bd5295aab0041fa920735e1101386d0cddf75f66834f3b72d35d9161e8af0d03
# NBJ006_BEGIN_CELL 4
if not all(p.is_file() for p in (META_PATH, MODEL_PATH, BASE_ROWS)):
    raise FileNotFoundError("NBJ-005 metadata, bundle 또는 inference_rows.csv가 없습니다.")
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
if sys.executable != meta["python_executable"]:
    raise RuntimeError(f'Python executable mismatch: {sys.executable!r} != {meta["python_executable"]!r}')
if os.environ.get("PYTHONPATH") != meta["pythonpath"]:
    raise RuntimeError(f'PYTHONPATH mismatch: {os.environ.get("PYTHONPATH")!r} != {meta["pythonpath"]!r}')
sha256 = lambda b: hashlib.sha256(b).hexdigest()
assert meta["verification_passed"] is True
assert sha256(MODEL_PATH.read_bytes()) == meta["bundle_sha256"]
assert sha256((BASE / "pipeline.py").read_bytes()) == meta["code_sha256"]
assert (np.__version__, pd.__version__, sklearn.__version__) == (
    meta["numpy"], meta["pandas"], meta["sklearn"])
assert sys.version == meta["python"]
assert int(meta["threads"]) == 4
assert meta["feature_order"] == ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
assert meta["train_cutoff"] == "2020-03-31" and (meta["train_rows"], meta["followup_rows"]) == (8470, 3469)
assert meta["kmeans"]["n_clusters"] == 2 and meta["seed"] == 42
assert meta["threshold_definition"].startswith("train score 95% linear quantile")
assert meta["score_definition"] == "-score_samples(IF scaler transform(X))"
RAW4 = list(meta["feature_order"])
ZIP_PATH = Path(meta["source"]["zip_path"])
print({"source": meta["source"], "kmeans": meta["kmeans"],
       "isolation_forest": meta["isolation_forest"], "threshold": meta["threshold"]})
# NBJ006_END_CELL 4

# ## 3. 원본 hash 및 schema 검증
# source_sha256: 1089cb66b156cb7e197590c4ca84c08256716346c361326723e5830006f3bc4a
# NBJ006_BEGIN_CELL 6
if not ZIP_PATH.is_file():
    raise FileNotFoundError(ZIP_PATH)
zip_bytes = ZIP_PATH.read_bytes()
assert sha256(zip_bytes) == meta["source"]["zip_sha256"]
with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
    excel_bytes = archive.read(meta["source"]["zip_member"])
assert sha256(excel_bytes) == meta["source"]["excel_sha256"]
raw = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=meta["source"]["sheet"])
quality_raw = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="result")
assert raw.shape == (11939, 10) and list(raw.columns[6:]) == RAW4
assert len(quality_raw) == 23
assert raw[RAW4].notna().all().all()
assert np.isfinite(raw[RAW4].to_numpy(float)).all()
raw["excel_row"] = np.arange(2, len(raw) + 2)
raw["date"] = pd.to_datetime(raw["working time"], errors="raise").dt.strftime("%Y-%m-%d")
raw["input_id"] = [f'{meta["source"]["excel_sha256"]}:Raw data:{i}' for i in raw["excel_row"]]
assert raw["date"].nunique() == 9 and raw["excel_row"].is_unique
print({"raw_rows": len(raw), "result_rows": len(quality_raw), "dates": raw["date"].nunique()})
# NBJ006_END_CELL 6

# ## 4. train/follow-up split 재현
# source_sha256: 31e3795984d4e6b8caa91b485a4a8b58271159106cd44b2551725958bd3bee66
# NBJ006_BEGIN_CELL 8
train_mask = raw["date"].le(meta["train_cutoff"]).to_numpy(bool)
split = np.where(train_mask, "train", "follow-up")
assert (int(train_mask.sum()), int((~train_mask).sum())) == (8470, 3469)
assert np.all(np.diff(raw["excel_row"].to_numpy()) == 1)
display(pd.DataFrame({"split": split}).value_counts().rename("rows"))
# NBJ006_END_CELL 8

# ## 5. Raw 4 feature 확인
# source_sha256: 11c121aed1e53b96030588dc390f0a1c5df72b68f7dc8d988fc202ba3a385d03
# NBJ006_BEGIN_CELL 10
display(raw[RAW4].describe().T)
display(raw.groupby("date", sort=True).size().rename("raw_rows"))
# NBJ006_END_CELL 10

# ## 6. physics proxy 생성
# source_sha256: 995d085cf75878bd60cd8d1dbc057d05689dadf675d1030d19070f3a288b5c12
# NBJ006_BEGIN_CELL 12
force, current, voltage, weld_time = RAW4
if raw[current].eq(0).any():
    raise ValueError("Current 0 발견: R_proxy 분모 불가")
features = raw[RAW4].copy()
features["R_proxy"] = raw[voltage] / raw[current]
features["VI_t_proxy"] = raw[voltage] * raw[current] * raw[weld_time]
features["I2_t_proxy"] = raw[current] ** 2 * raw[weld_time]
DERIVED = ["R_proxy", "VI_t_proxy", "I2_t_proxy"]
if not np.isfinite(features.to_numpy(float)).all():
    raise ValueError("비유한 파생값 발견")
assert features.index.equals(raw.index)
# NBJ006_END_CELL 12

# ## 7. derived feature 기술통계
# source_sha256: e4e281a0b698181396caae70fe1664fa4069d440f5adf29d9c7011d98f06be15
# NBJ006_BEGIN_CELL 14
train_features = features.loc[train_mask]
display(train_features.describe().T)
display(raw.loc[train_mask, weld_time].value_counts(dropna=False).sort_index().rename("train_rows"))
# NBJ006_END_CELL 14

# ## 8. Pearson/Spearman redundancy 분석
# source_sha256: 4a7f0eee2079cc241707da77b5279102d0bf7e666006aa5c942d70b2bfbeebde
# NBJ006_BEGIN_CELL 16
pearson = train_features.corr(method="pearson")
spearman = train_features.corr(method="spearman")
display(pearson)
display(spearman)
display(spearman.loc[DERIVED, DERIVED])
fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
for ax, matrix, title in zip(axes, (pearson, spearman), ("Train Pearson", "Train Spearman")):
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set(xticks=range(len(matrix)), yticks=range(len(matrix)),
           xticklabels=matrix.columns, yticklabels=matrix.index, title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.colorbar(im, ax=ax, label="Correlation")
plt.show()
# NBJ006_END_CELL 16

# ## 9. feature set 정의
# source_sha256: 5bc0a836b0fa243a7b6b52d1155c084e83c819d4616d696c0a4acf574e53cbe3
# NBJ006_BEGIN_CELL 18
FEATURE_SETS = {
    "FS0_RAW4": RAW4,
    "FS1_R": RAW4 + ["R_proxy"],
    "FS2_ENERGY": RAW4 + ["VI_t_proxy"],
    "FS3_I2T": RAW4 + ["I2_t_proxy"],
    "FS4_ALL": RAW4 + DERIVED,
}
assert [len(v) for v in FEATURE_SETS.values()] == [4, 5, 5, 5, 7]
assert all(v[:4] == RAW4 for v in FEATURE_SETS.values())
display(pd.DataFrame({"set": list(FEATURE_SETS), "features": list(FEATURE_SETS.values())}))
# NBJ006_END_CELL 18

# ## 10. FS0 seed=42 NBJ-005 regression test
# source_sha256: c48a67d931bca4a97a0a5a7d41e026ddeb57972ba8c36a24c9ba1f556b708902
# NBJ006_BEGIN_CELL 20
def fit_k(set_name, seed):
    x = features[FEATURE_SETS[set_name]].to_numpy(float)
    scaler = StandardScaler().fit(x[train_mask])
    model = KMeans(**{**meta["kmeans"], "random_state": seed})
    model.fit(scaler.transform(x[train_mask]))
    z = scaler.transform(x)
    labels = model.predict(z)
    distance = np.linalg.norm(z - model.cluster_centers_[labels], axis=1)
    return {"scaler": scaler, "model": model, "labels": labels,
            "distance": distance, "train_z": z[train_mask]}

def fit_if(set_name, seed):
    x = features[FEATURE_SETS[set_name]].to_numpy(float)
    scaler = StandardScaler().fit(x[train_mask])
    model = IsolationForest(**{**meta["isolation_forest"], "random_state": seed})
    model.fit(scaler.transform(x[train_mask]))
    score = -model.score_samples(scaler.transform(x))
    calculated = float(np.quantile(score[train_mask], .95, method="linear"))
    threshold = float(meta["threshold"]) if (set_name, seed) == ("FS0_RAW4", 42) else calculated
    flag = score >= threshold
    return {"scaler": scaler, "model": model, "score": score,
            "threshold_calculated": calculated, "threshold": threshold, "flag": flag}

REGRESSION_GATE_PASSED = False
k_runs = {("FS0_RAW4", 42): fit_k("FS0_RAW4", 42)}
if_runs = {("FS0_RAW4", 42): fit_if("FS0_RAW4", 42)}
k0, i0 = k_runs[("FS0_RAW4", 42)], if_runs[("FS0_RAW4", 42)]
saved = pd.read_csv(BASE_ROWS)
assert len(saved) == len(raw)
assert np.array_equal(raw["input_id"].to_numpy(), saved["input_id"].to_numpy())
assert np.array_equal(raw["excel_row"].to_numpy(), saved["excel_row"].to_numpy())
assert np.array_equal(raw["date"].to_numpy(), saved["date"].to_numpy())
assert np.array_equal(split, saved["split"].to_numpy())
assert np.array_equal(k0["labels"], saved["cluster_id"].to_numpy())
assert np.allclose(k0["distance"], saved["centroid_distance_scaled"], rtol=RTOL, atol=ATOL)
assert np.allclose(i0["score"], saved["if_anomaly_score"], rtol=RTOL, atol=ATOL)
assert np.array_equal(i0["flag"], saved["candidate_flag"].to_numpy(bool))
assert np.isclose(i0["threshold_calculated"], meta["threshold"], rtol=RTOL, atol=ATOL)
assert np.array_equal(np.bincount(k0["labels"], minlength=2), 
                      np.bincount(saved["cluster_id"], minlength=2))
for which in ("train", "follow-up"):
    mask = split == which
    assert int(mask.sum()) == int((saved["split"] == which).sum())
    assert np.array_equal(np.bincount(k0["labels"][mask], minlength=2),
                          np.bincount(saved.loc[mask, "cluster_id"], minlength=2))
    assert int(i0["flag"][mask].sum()) == int(saved.loc[mask, "candidate_flag"].sum())
check = pd.DataFrame({"date": raw["date"], "cluster_id": k0["labels"], "flag": i0["flag"]})
new_group_check = check[["date", "cluster_id"]].copy()
old_group_check = saved[["date", "cluster_id"]].copy()
new_group_check["cluster_id"] = new_group_check["cluster_id"].astype("int64")
old_group_check["cluster_id"] = old_group_check["cluster_id"].astype("int64")
new_cluster = new_group_check.groupby(["date", "cluster_id"]).size()
old_cluster = old_group_check.groupby(["date", "cluster_id"]).size()
pd.testing.assert_series_equal(new_cluster, old_cluster)
new_if = check.groupby("date")["flag"].sum()
old_if = saved.groupby("date")["candidate_flag"].sum()
pd.testing.assert_series_equal(new_if, old_if, check_names=False)
REGRESSION_GATE_PASSED = True
print("NBJ-005 FS0 seed=42 regression gate: PASS")
# NBJ006_END_CELL 20

# ## 11. K-Means multi-seed fitting
# source_sha256: cc4c158e795425211badfd203a95f792b01a1e054f38b1835055461a5417b3f9
# NBJ006_BEGIN_CELL 22
assert REGRESSION_GATE_PASSED is True, "NBJ-005 regression gate failed or was not run"
assert ("FS0_RAW4", 42) in k_runs and ("FS0_RAW4", 42) in if_runs
for set_name in FEATURE_SETS:
    for seed in SEEDS:
        if (set_name, seed) not in k_runs:
            k_runs[(set_name, seed)] = fit_k(set_name, seed)
assert len(k_runs) == 25
# NBJ006_END_CELL 22

# ## 12. K-Means stability 분석
# source_sha256: 00c75c6bc6bebe4ed797975d785094b36bfd75cf1508a09f42120b1a45bf80cd
# NBJ006_BEGIN_CELL 24
k_rows, k_pairs, k_profiles = [], [], []
for set_name, cols in FEATURE_SETS.items():
    for seed in SEEDS:
        run = k_runs[(set_name, seed)]
        labels = run["labels"]
        sil = float(silhouette_score(run["train_z"], labels[train_mask]))
        for part in ("train", "follow-up"):
            mask = split == part
            for cluster in (0, 1):
                k_rows.append({"set": set_name, "seed": seed, "split": part,
                               "cluster_id": cluster, "rows": int(np.sum(labels[mask] == cluster)),
                               "train_silhouette": sil})
    for a, b in combinations(SEEDS, 2):
        k_pairs.append({"set": set_name, "seed_a": a, "seed_b": b,
                        "ari": adjusted_rand_score(k_runs[(set_name, a)]["labels"],
                                                   k_runs[(set_name, b)]["labels"])})
    labels42 = k_runs[(set_name, 42)]["labels"]
    for cluster in (0, 1):
        frame = features.loc[labels42 == cluster, cols]
        for col in cols:
            k_profiles.append({"set": set_name, "cluster_id": cluster, "feature": col,
                               "n": len(frame), "median": frame[col].median(),
                               "iqr": frame[col].quantile(.75) - frame[col].quantile(.25)})
k_summary = pd.DataFrame(k_rows)
k_seed_pairs = pd.DataFrame(k_pairs)
k_profile = pd.DataFrame(k_profiles)
daily_cluster_rows = []
for set_name in FEATURE_SETS:
    for seed in SEEDS:
        frame = pd.DataFrame({"date": raw["date"], "cluster": k_runs[(set_name, seed)]["labels"]})
        counts = frame.groupby(["date", "cluster"]).size()
        for (date, cluster), n in counts.items():
            daily_cluster_rows.append({"set": set_name, "seed": seed, "date": date,
                                       "cluster_id": cluster, "rows": int(n),
                                       "share_pct": 100 * n / int((raw["date"] == date).sum())})
daily_cluster = pd.DataFrame(daily_cluster_rows)
display(k_summary)
display(k_seed_pairs)
display(k_profile)
display(daily_cluster)
# NBJ006_END_CELL 24

# ## 13. IF multi-seed fitting
# source_sha256: 5e30c111de9e879a6b335c7b32f1f05cf929cb08e420128d84adcf7b10adfcc5
# NBJ006_BEGIN_CELL 26
for set_name in FEATURE_SETS:
    for seed in SEEDS:
        if (set_name, seed) not in if_runs:
            if_runs[(set_name, seed)] = fit_if(set_name, seed)
assert len(if_runs) == 25
if_rows, if_daily_rows = [], []
for (set_name, seed), run in if_runs.items():
    for part in ("train", "follow-up"):
        mask = split == part
        if_rows.append({"set": set_name, "seed": seed, "split": part,
                        "threshold": run["threshold"], "candidate_n": int(run["flag"][mask].sum()),
                        "candidate_pct": float(run["flag"][mask].mean() * 100),
                        "score_median": float(np.median(run["score"][mask]))})
    daily = pd.DataFrame({"date": raw["date"], "score": run["score"], "flag": run["flag"]})
    for date, group in daily.groupby("date"):
        if_daily_rows.append({"set": set_name, "seed": seed, "date": date,
                              "candidate_n": int(group["flag"].sum()),
                              "candidate_pct": float(group["flag"].mean() * 100),
                              "score_median": float(group["score"].median())})
if_summary = pd.DataFrame(if_rows)
if_daily = pd.DataFrame(if_daily_rows)
display(if_summary)
display(if_daily)
# NBJ006_END_CELL 26

# ## 14. IF score/candidate stability 분석
# source_sha256: be893d1cc0dd661bc95021077a2e24ba63137fb6e751606e0098c16f23c3ed2b
# NBJ006_BEGIN_CELL 28
def jaccard(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = int(np.logical_or(a, b).sum())
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)

if_pairs = []
for set_name in FEATURE_SETS:
    for a, b in combinations(SEEDS, 2):
        left, right = if_runs[(set_name, a)], if_runs[(set_name, b)]
        if_pairs.append({"set": set_name, "seed_a": a, "seed_b": b,
                         "score_spearman": pd.Series(left["score"]).corr(pd.Series(right["score"]), method="spearman"),
                         "candidate_jaccard": jaccard(left["flag"], right["flag"])})
if_seed_pairs = pd.DataFrame(if_pairs)
display(if_seed_pairs)
# NBJ006_END_CELL 28

# ## 15. FS0 vs physics feature set 비교
# source_sha256: fb8ebe2ab2f29211c2dc0eb6c18c9a9e48000a951a6c8931ee50d934f1a64df7
# NBJ006_BEGIN_CELL 30
anchor_k = k_runs[("FS0_RAW4", 42)]
anchor_if = if_runs[("FS0_RAW4", 42)]
fs0_compare = []
changed_masks = {}
for set_name in FEATURE_SETS:
    kr, ir = k_runs[(set_name, 42)], if_runs[(set_name, 42)]
    kept = anchor_if["flag"] & ir["flag"]
    lost = anchor_if["flag"] & ~ir["flag"]
    gained = ~anchor_if["flag"] & ir["flag"]
    changed_masks[set_name] = {"kept": kept, "lost": lost, "gained": gained}
    fs0_compare.append({"set": set_name,
                        "cluster_ari_vs_fs0": adjusted_rand_score(anchor_k["labels"], kr["labels"]),
                        "score_spearman_vs_fs0": pd.Series(anchor_if["score"]).corr(pd.Series(ir["score"]), method="spearman"),
                        "candidate_overlap_n": int(kept.sum()),
                        "candidate_jaccard_vs_fs0": jaccard(anchor_if["flag"], ir["flag"]),
                        "kept_n": int(kept.sum()), "lost_n": int(lost.sum()), "gained_n": int(gained.sum())})
fs0_comparison = pd.DataFrame(fs0_compare)
display(fs0_comparison)
# NBJ006_END_CELL 30

# ## 16. 날짜별 exploratory quality relation
# source_sha256: 252d35e4a2748ececf0ceb3acf4063aa5f87c9dd7a8ed2d817b1537bf85db276
# NBJ006_BEGIN_CELL 32
EXPECTED_RESULT_COLUMNS = ["idx", "Machine_Name", "Item No", "working time", "defect", "defect type", "Unnamed: 6"]
if list(quality_raw.columns) != EXPECTED_RESULT_COLUMNS:
    raise ValueError("result schema changed; denominator meaning must be rechecked before quality analysis")
if not QUALITY_PATH.is_file():
    raise FileNotFoundError(QUALITY_PATH)
prior_quality = pd.read_csv(QUALITY_PATH).set_index("date")
raw_daily_rows = raw.groupby("date").size()
assert set(prior_quality.index) == set(raw_daily_rows.index)
assert np.array_equal(prior_quality.loc[raw_daily_rows.index, "raw_records"].to_numpy(), raw_daily_rows.to_numpy())
quality_raw["date"] = pd.to_datetime(quality_raw["working time"], errors="raise").dt.strftime("%Y-%m-%d")
quality = quality_raw.groupby("date").agg(recorded_defects=("defect", "sum"),
                                          recorded_types=("defect type", "nunique"))
quality["raw_rows"] = raw_daily_rows.reindex(quality.index)
assert "2020-03-27" not in quality.index
assert int(quality.loc["2020-03-31", "recorded_types"]) == 2
assert np.array_equal(quality["recorded_types"].to_numpy(), prior_quality.loc[quality.index, "recorded_types"].to_numpy())
assert np.array_equal(quality["recorded_defects"].to_numpy(), prior_quality.loc[quality.index, "recorded_sum_partial"].to_numpy())
quality = quality.loc[quality["recorded_types"].eq(3)].copy()
assert len(quality) == 7
quality["defects_per_1000_raw_records"] = 1000 * quality["recorded_defects"] / quality["raw_rows"]
assert np.allclose(quality["defects_per_1000_raw_records"], prior_quality.loc[quality.index, "complete_sum_per1000records"], rtol=RTOL, atol=ATOL)
print("Quality denominator: Raw daily row count, not confirmed inspected/product count; seven complete recorded dates")
alignment_rows, aligned_cluster_by_set = [], {}
reference = anchor_k["labels"]
for set_name in FEATURE_SETS:
    labels = k_runs[(set_name, 42)]["labels"]
    identity_n = int(np.sum(labels == reference))
    swap_n = int(np.sum((1 - labels) == reference))
    mapping = {0: 0, 1: 1} if identity_n >= swap_n else {0: 1, 1: 0}
    aligned = np.array([mapping[int(label)] for label in labels], dtype=int)
    aligned_cluster_by_set[set_name] = aligned
    alignment_rows.append({"set": set_name, "seed": 42, "mapping": mapping,
                           "identity_overlap_n": identity_n, "swap_overlap_n": swap_n,
                           "selected_overlap_n": int(np.sum(aligned == reference)),
                           "tie_rule": "identity"})
cluster_alignment = pd.DataFrame(alignment_rows)
display(cluster_alignment)
quality_rows = []
for set_name in FEATURE_SETS:
    ir = if_runs[(set_name, 42)]
    frame = pd.DataFrame({"date": raw["date"], "score": ir["score"],
                          "flag": ir["flag"], "aligned_cluster": aligned_cluster_by_set[set_name]})
    daily = frame.groupby("date").agg(score_median=("score", "median"),
                                      candidate_share=("flag", "mean"),
                                      aligned_cluster1_share=("aligned_cluster", "mean"))
    daily[["candidate_share", "aligned_cluster1_share"]] *= 100
    joined = daily.join(quality[["recorded_defects", "defects_per_1000_raw_records"]], how="inner")
    for indicator in ("score_median", "candidate_share", "aligned_cluster1_share"):
        for target in ("recorded_defects", "defects_per_1000_raw_records"):
            quality_rows.append({"set": set_name, "indicator": indicator, "target": target,
                                 "n_dates": len(joined),
                                 "spearman": joined[indicator].corr(joined[target], method="spearman")})
quality_exploratory = pd.DataFrame(quality_rows)
display(quality_exploratory)
# NBJ006_END_CELL 32

# ## 17. candidate changed-row 분석
# source_sha256: 85b26427a04afda59a15584a4d30e59be61f16d9a4f21a7507a8e72cf469b1ce
# NBJ006_BEGIN_CELL 34
changed_rows = []
changed_profiles = []
for set_name in FEATURE_SETS:
    for status, mask in changed_masks[set_name].items():
        selected = raw.loc[mask, ["excel_row", "date"]].copy()
        selected["set"] = set_name
        selected["status"] = status
        changed_rows.append(selected)
        for col in features.columns:
            values = features.loc[mask, col]
            changed_profiles.append({"set": set_name, "status": status, "feature": col,
                                     "n": len(values), "median": values.median(),
                                     "iqr": values.quantile(.75) - values.quantile(.25)})
changed_row_table = pd.concat(changed_rows, ignore_index=True)
changed_profile_table = pd.DataFrame(changed_profiles)
display(changed_profile_table)
# NBJ006_END_CELL 34

# ## 18. 최종 비교표
# source_sha256: ddb9a54e9887014bb93a4ece3d11a8b524c60199856a099295d8ba3cbae7e5ef
# NBJ006_BEGIN_CELL 36
def pair_stat(table, col):
    return table.groupby("set")[col].agg(["median", "min"]).rename(
        columns={"median": col + "_median", "min": col + "_min"})

comparison = pd.DataFrame(index=FEATURE_SETS)
comparison.index.name = "set"
comparison["feature_n"] = [len(v) for v in FEATURE_SETS.values()]
comparison["train_silhouette"] = (k_summary.query("seed == 42 and split == 'train'")
                                   .drop_duplicates("set").set_index("set")["train_silhouette"])
comparison = comparison.join(pair_stat(k_seed_pairs, "ari"))
comparison = comparison.join(pair_stat(if_seed_pairs, "score_spearman"))
comparison = comparison.join(pair_stat(if_seed_pairs, "candidate_jaccard"))
comparison = comparison.join(fs0_comparison.set_index("set")[["cluster_ari_vs_fs0",
    "score_spearman_vs_fs0", "candidate_overlap_n", "candidate_jaccard_vs_fs0",
    "kept_n", "lost_n", "gained_n"]])
for part, label in (("train", "train_candidate_pct"), ("follow-up", "followup_candidate_pct")):
    comparison[label] = if_summary.query("seed == 42 and split == @part").set_index("set")["candidate_pct"]
comparison["train_candidate_pct_role"] = "sanity check only; not for model ranking"
comparison["redundancy_max_abs_spearman"] = [
    0.0 if len(cols) == 4 else spearman.loc[[c for c in cols if c in DERIVED], [x for x in cols if x not in DERIVED]].abs().max().max()
    for cols in FEATURE_SETS.values()
]
comparison["changed_candidate_conditions"] = "검토 후 changed_profile_table을 근거로 기술"
comparison["interpretation_advantage"] = "검토 전"
comparison["interpretation_limit"] = "검토 전"
display(comparison)
# NBJ006_END_CELL 36

