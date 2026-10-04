"""Execute the frozen NBJ-004 K-Means/IF plan against the read-only KAMP ZIP."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import zipfile
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "4"
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs" / ".mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs"
DEFAULT_ZIP = Path(r"C:\Users\skqja\Downloads\2. 용접기 AI 데이터셋 (1).zip")
INNER = "2. 용접기 AI 데이터셋/Welding Data Set_01.xlsx"
EXPECTED_ZIP = "54a41de1e3d0c60cf47d66d707e1e8ff78dfd203b5024b478f9113c5a4435873"
EXPECTED_EXCEL = "d514d6aaa121630c04d7d51c97a56025e78e2c86868813f7722ba5db922c1f33"
COLS = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
SHORT = ["force_bar", "current_kA", "voltage_V", "weld_time_ms"]
SEEDS = (42, 7, 2026)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUTPUT / name, index=False, encoding="utf-8-sig", float_format="%.15g")


def load(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    zbytes = zip_path.read_bytes()
    assert digest(zbytes) == EXPECTED_ZIP, "ZIP SHA-256 differs from the frozen plan"
    with zipfile.ZipFile(io.BytesIO(zbytes)) as archive:
        excel = archive.read(INNER)
    assert digest(excel) == EXPECTED_EXCEL, "Workbook SHA-256 differs from the frozen plan"
    book = pd.ExcelFile(io.BytesIO(excel))
    assert book.sheet_names == ["Raw data", "result", "data set"]
    raw = pd.read_excel(book, sheet_name="Raw data")
    quality = pd.read_excel(book, sheet_name="result")
    assert raw.shape == (11939, 10) and quality.shape == (23, 7)
    assert list(raw.columns[6:]) == COLS
    assert not raw.isna().any().any() and not raw[COLS].isna().any().any()
    raw = raw.copy()
    raw["excel_row"] = np.arange(2, len(raw) + 2)
    raw["date"] = pd.to_datetime(raw["working time"]).dt.strftime("%Y-%m-%d")
    raw["train"] = raw["date"] <= "2020-03-31"
    assert raw["train"].sum() == 8470 and (~raw["train"]).sum() == 3469
    assert raw["date"].nunique() == 9 and raw["excel_row"].is_unique
    assert raw["Machine_Name"].nunique() == raw["Item No"].nunique() == 1
    assert np.isfinite(raw[COLS].to_numpy(float)).all()
    quality = quality.copy()
    quality["date"] = pd.to_datetime(quality["working time"]).dt.strftime("%Y-%m-%d")
    assert quality["date"].nunique() == 8
    assert not quality.duplicated(["date", "Machine_Name", "Item No", "defect type"]).any()
    return raw, quality, {"zip_sha256": EXPECTED_ZIP, "excel_sha256": EXPECTED_EXCEL,
                          "zip_path": str(zip_path), "zip_member": INNER}


def kmeans(raw: pd.DataFrame, x: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = raw["train"].to_numpy(bool)
    train_x = x[train]
    scaler = StandardScaler().fit(train_x)
    z = scaler.transform(x)
    assert np.allclose(z[train].mean(axis=0), 0, atol=1e-9)
    rng = np.random.default_rng(42)
    sample = np.sort(rng.choice(np.flatnonzero(train), size=min(5000, train.sum()), replace=False))
    sample_is_fixed = digest(sample.tobytes())
    models: dict[int, KMeans] = {}
    records = []
    inertia_prior = None
    min_n = max(50, math.ceil(0.01 * int(train.sum())))
    for k in range(1, 7):
        fits = {}
        for seed in ((42,) if k == 1 else SEEDS):
            model = KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=300,
                           tol=1e-4, algorithm="lloyd", random_state=seed).fit(z[train])
            fits[seed] = model
        main = fits[42]
        models[k] = main
        labels = main.labels_
        counts = np.bincount(labels, minlength=k)
        sil = float(silhouette_score(z[sample], main.predict(z[sample]))) if k > 1 else np.nan
        medians = np.array([np.median(train_x[labels == i], axis=0) for i in range(k)])
        if k > 1:
            distance = np.abs(medians[:, None, :] - medians[None, :, :]) / scaler.scale_
            np.fill_diagonal(distance[:, :, 0], np.nan)
            distinguishable = bool(np.all(np.nanmax(distance, axis=(1, 2)) >= 0.5))
        else:
            distinguishable = False
        aris = {seed: adjusted_rand_score(labels, fits[seed].labels_) for seed in (7, 2026)} if k > 1 else {}
        eligible = (k > 1 and min(counts) >= min_n and sil >= .25
                    and all(value >= .80 for value in aris.values()) and distinguishable)
        records.append({"K": k, "inertia": main.inertia_,
                        "relative_inertia_drop": (inertia_prior - main.inertia_) / inertia_prior
                        if inertia_prior is not None else np.nan,
                        "silhouette_sample5000": sil, "min_cluster_train": min(counts),
                        "min_size_required": min_n, "ari_seed7": aris.get(7, np.nan),
                        "ari_seed2026": aris.get(2026, np.nan),
                        "distinct_median_half_sd": distinguishable, "eligible": eligible})
        inertia_prior = main.inertia_
        print(f"K={k}: inertia={main.inertia_:.3f} silhouette={sil:.4f} eligible={eligible}", flush=True)
    table = pd.DataFrame(records)
    valid = table.loc[table.eligible]
    chosen = None if valid.empty else int(valid.loc[valid.silhouette_sample5000 >= valid.silhouette_sample5000.max() - .02, "K"].min())
    elbow = None
    if chosen is not None and chosen < 6:
        before = table.loc[table.K == chosen, "relative_inertia_drop"].iloc[0]
        after = table.loc[table.K == chosen + 1, "relative_inertia_drop"].iloc[0]
        elbow = bool(after <= 0.5 * before)
    table["selected_by_rule"] = table.K.eq(chosen) if chosen is not None else False
    save(table, "k_diagnostics.csv")

    # If no K meets the frozen criteria, retain the highest-silhouette K solely
    # as a diagnostic partition; it is never called a confirmed regime.
    diagnostic_k = int(table.loc[table.K > 1].sort_values(["silhouette_sample5000", "K"], ascending=[False, True]).iloc[0].K)
    used_k = chosen if chosen is not None else diagnostic_k
    main = models[used_k]
    all_labels = main.predict(z)
    assignment = raw[["excel_row", "date", "train"]].copy()
    assignment["cluster"] = all_labels
    assignment["distance_to_center_scaled"] = np.linalg.norm(z - main.cluster_centers_[all_labels], axis=1)
    for col, short in zip(COLS, SHORT):
        assignment[short] = raw[col].to_numpy()
    assignment["force_outside_2_2p6"] = ~raw[COLS[0]].between(2.0, 2.6)
    assignment["force_gt3"] = raw[COLS[0]] > 3.0
    save(assignment, "row_clusters.csv")

    profiles = []
    for split, mask in (("all", np.ones(len(raw), bool)), ("train", train), ("later", ~train)):
        for cluster in range(used_k):
            part = assignment.loc[mask & (all_labels == cluster)]
            row = {"split": split, "cluster": cluster, "n": len(part), "share": len(part) / mask.sum(),
                   "force_outside_2_2p6_n": int(part.force_outside_2_2p6.sum()),
                   "force_gt3_n": int(part.force_gt3.sum())}
            for col in SHORT:
                row.update({f"{col}_{name}": value for name, value in zip(
                    ("min", "q1", "median", "q3", "max"), np.quantile(part[col], [0, .25, .5, .75, 1]))})
            profiles.append(row)
    save(pd.DataFrame(profiles), "cluster_profiles.csv")
    daily = assignment.groupby(["date", "cluster"], as_index=False).size().rename(columns={"size": "n"})
    daily["daily_records"] = daily.groupby("date").n.transform("sum")
    daily["share"] = daily.n / daily.daily_records
    save(daily, "daily_clusters.csv")

    # Fixed-K sensitivity; predict every original row and align labels by
    # maximum overlap on the common original training rows.
    sensitivity = []
    sensitivity_assignments = {}
    if chosen is not None:
        for setting in ("robust", "unique4"):
            if setting == "robust":
                fit_rows = np.flatnonzero(train)
                alt_scaler = RobustScaler().fit(x[fit_rows])
            else:
                fit_rows = np.flatnonzero(train)[~pd.DataFrame(train_x).duplicated().to_numpy()]
                alt_scaler = StandardScaler().fit(x[fit_rows])
            altz = alt_scaler.transform(x)
            alt = KMeans(n_clusters=chosen, init="k-means++", n_init=20, max_iter=300,
                         tol=1e-4, algorithm="lloyd", random_state=42).fit(altz[fit_rows])
            alt_labels = alt.predict(altz)
            contingency = np.zeros((chosen, chosen), dtype=int)
            for a, b in zip(all_labels[train], alt_labels[train]):
                contingency[a, b] += 1
            rr, cc = linear_sum_assignment(-contingency)
            mapping = dict(zip(cc, rr))
            aligned = np.array([mapping[v] for v in alt_labels])
            sensitivity_assignments[setting] = aligned
            for split, mask in (("all", np.ones(len(raw), bool)), ("train", train), ("later", ~train)):
                sensitivity.append({"setting": setting, "split": split, "K": chosen,
                                    "fit_n": len(fit_rows), "ari": adjusted_rand_score(all_labels[mask], alt_labels[mask]),
                                    "aligned_disagreement_share": np.mean(aligned[mask] != all_labels[mask])})
        save(pd.DataFrame(sensitivity), "k_sensitivity.csv")
        save(pd.DataFrame({"excel_row": raw.excel_row, "cluster": all_labels,
                           "robust_aligned": sensitivity_assignments["robust"],
                           "unique4_aligned": sensitivity_assignments["unique4"]}), "sensitivity_assignments.csv")

    selection = {"selected_K": chosen, "diagnostic_K": diagnostic_k, "used_K": used_k,
                 "regime_status": "no_eligible_K" if chosen is None else
                 ("provisional_elbow_unavailable" if elbow is None else
                  "provisional_elbow_conflict" if not elbow else "eligible_with_elbow"),
                 "elbow_support": elbow, "silhouette_sample_row_index_sha256": sample_is_fixed,
                 "silhouette_sample_n": len(sample), "sensitivity": sensitivity}
    return assignment, daily, selection


def isolation_forest(raw: pd.DataFrame, x: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = raw.train.to_numpy(bool)
    scaler = StandardScaler().fit(x[train])
    z = scaler.transform(x)
    out = raw[["excel_row", "date", "train"]].copy()
    info = []
    for seed in SEEDS:
        model = IsolationForest(n_estimators=300, max_samples=min(256, train.sum()),
                                max_features=1.0, bootstrap=False, contamination="auto",
                                random_state=seed, n_jobs=4).fit(z[train])
        score = -model.score_samples(z)
        threshold95 = float(np.quantile(score[train], .95, method="linear"))
        threshold99 = float(np.quantile(score[train], .99, method="linear"))
        out[f"if_{seed}_score"] = score
        out[f"if_{seed}_flag95"] = score >= threshold95
        out[f"if_{seed}_flag99"] = score >= threshold99
        info.append({"seed": seed, "threshold95": threshold95, "threshold99": threshold99,
                     "train_flags95": int((score[train] >= threshold95).sum()),
                     "later_flags95": int((score[~train] >= threshold95).sum()),
                     "train_flags99": int((score[train] >= threshold99).sum()),
                     "later_flags99": int((score[~train] >= threshold99).sum())})
    save(out, "if_row_scores.csv")
    base = out.if_42_score
    stable = []
    for seed in (7, 2026):
        a = out.if_42_flag95.to_numpy(bool)
        b = out[f"if_{seed}_flag95"].to_numpy(bool)
        stable.append({"seed": seed, "spearman_score": float(spearmanr(base, out[f"if_{seed}_score"]).statistic),
                       "jaccard_flag95": float(np.sum(a & b) / np.sum(a | b))})
    save(pd.DataFrame(info), "if_thresholds.csv")
    save(pd.DataFrame(stable), "if_seed_stability.csv")
    sensor = raw[["excel_row", "date", "train"] + COLS].copy()
    sensor["if_score"] = out.if_42_score
    sensor["if_flag95"] = out.if_42_flag95
    sensor["force_outside_2_2p6"] = ~sensor[COLS[0]].between(2, 2.6)
    sensor["force_gt3"] = sensor[COLS[0]] > 3
    by_sensor = []
    for split, mask in (("all", np.ones(len(raw), bool)), ("train", train), ("later", ~train)):
        for flag in (False, True):
            part = sensor.loc[mask & (sensor.if_flag95 == flag)]
            row = {"split": split, "if_flag95": flag, "n": len(part),
                   "score_median": part.if_score.median(),
                   "force_outside_2_2p6_n": int(part.force_outside_2_2p6.sum()),
                   "force_gt3_n": int(part.force_gt3.sum())}
            for col, short in zip(COLS, SHORT):
                row.update({f"{short}_{name}": val for name, val in zip(
                    ("q1", "median", "q3"), part[col].quantile([.25, .5, .75]))})
            by_sensor.append(row)
    save(pd.DataFrame(by_sensor), "if_sensor_profiles.csv")
    daily = sensor.groupby("date").agg(records=("excel_row", "size"), score_median=("if_score", "median"),
                                       score_p95=("if_score", lambda v: v.quantile(.95)),
                                       flagged=("if_flag95", "sum"),
                                       force_outside_2_2p6=("force_outside_2_2p6", "sum"),
                                       force_gt3=("force_gt3", "sum")).reset_index()
    daily["if_flag_share"] = daily.flagged / daily.records
    save(daily, "if_daily.csv")
    return out, daily, {"thresholds": info, "seed_stability": stable}


def cluster_if_comparison(raw: pd.DataFrame, clusters: pd.DataFrame, if_rows: pd.DataFrame) -> None:
    joined = clusters[["excel_row", "date", "train", "cluster"]].merge(
        if_rows[["excel_row", "if_42_score", "if_42_flag95"]], on="excel_row", validate="one_to_one")
    rows = []
    for split, mask in (("all", np.ones(len(raw), bool)), ("train", raw.train.to_numpy(bool)),
                        ("later", ~raw.train.to_numpy(bool))):
        for cluster, part in joined.loc[mask].groupby("cluster"):
            rows.append({"split": split, "cluster": int(cluster), "records": len(part),
                         "if_flags": int(part.if_42_flag95.sum()),
                         "if_share_within_cluster": float(part.if_42_flag95.mean()),
                         "if_score_median": float(part.if_42_score.median()),
                         "if_score_p95": float(part.if_42_score.quantile(.95))})
    save(pd.DataFrame(rows), "cluster_if.csv")


def get_ae(raw: pd.DataFrame, path: Path | None) -> tuple[pd.DataFrame | None, str]:
    if path is None or not path.exists():
        return None, "YSH-005 행별 AE 점수 파일과 모델이 없음; 계획 10.1절에 따라 비교 보류"
    score = pd.read_csv(path)
    needed = {"excel_row", "date", "train", "ae_42_score", "ae_42_flag"}
    if not needed.issubset(score.columns):
        return None, f"AE 파일 필수 열 부족: {sorted(needed - set(score.columns))}"
    if len(score) != len(raw) or score.excel_row.duplicated().any():
        return None, "AE 행 수 또는 원본행 키가 일치하지 않음"
    check = raw[["excel_row", "date", "train"]].merge(score, on="excel_row", validate="one_to_one", suffixes=("_raw", "_ae"))
    if len(check) != len(raw) or not check.date_raw.eq(check.date_ae).all() or not check.train_raw.eq(check.train_ae).all():
        return None, "AE 날짜·학습마스크가 원본과 일치하지 않음"
    return check, "기존 AE 행별 점수 사용"


def ae_comparison(raw: pd.DataFrame, clusters: pd.DataFrame, if_rows: pd.DataFrame,
                  ae: pd.DataFrame | None, selection: dict) -> dict:
    if ae is None:
        return {"status": "unavailable", "reason": "기존 AE 행별 점수·모델 미확보"}
    a = ae.ae_42_flag.to_numpy(bool)
    b = if_rows.if_42_flag95.to_numpy(bool)
    score = ae.ae_42_score.to_numpy(float)
    assert np.isfinite(score).all() and len(a) == len(raw)
    groups = np.select([a & b, a & ~b, ~a & b, ~a & ~b], [1, 2, 3, 4])
    joined = clusters.copy()
    joined["ae_score"] = score
    joined["ae_flag"] = a
    joined["if_score"] = if_rows.if_42_score.to_numpy(float)
    joined["if_flag"] = b
    joined["group"] = groups
    save(joined, "ae_if_groups_rows.csv")
    profiles = []
    for split, mask in (("all", np.ones(len(raw), bool)), ("train", raw.train.to_numpy(bool)),
                        ("later", ~raw.train.to_numpy(bool))):
        for group in (1, 2, 3, 4):
            part = joined.loc[mask & (groups == group)]
            item = {"split": split, "group": group, "n": len(part),
                    "force_outside_2_2p6_n": int(part.force_outside_2_2p6.sum()),
                    "force_gt3_n": int(part.force_gt3.sum())}
            for col in SHORT:
                item.update({f"{col}_{name}": val for name, val in zip(
                    ("q1", "median", "q3"), part[col].quantile([.25, .5, .75]))})
            profiles.append(item)
    save(pd.DataFrame(profiles), "ae_if_group_profiles.csv")
    save(joined.groupby(["date", "group"], as_index=False).size().rename(columns={"size": "n"}), "ae_if_group_daily.csv")
    save(joined.groupby(["cluster", "group"], as_index=False).size().rename(columns={"size": "n"}), "ae_if_group_clusters.csv")
    dt = joined.groupby("date").agg(records=("excel_row", "size"), ae_flags=("ae_flag", "sum"),
                                     if_flags=("if_flag", "sum"), common=("group", lambda s: (s == 1).sum())).reset_index()
    for col in ("ae_flags", "if_flags", "common"):
        dt[col + "_share"] = dt[col] / dt.records
    save(dt, "ae_if_daily.csv")
    # 99% score budgets are diagnostics only; original AE and IF flags stay intact.
    train = raw.train.to_numpy(bool)
    aq = np.quantile(score[train], .99)
    iq = np.quantile(if_rows.if_42_score.to_numpy()[train], .99)
    aqf, iqf = score >= aq, if_rows.if_42_score.to_numpy() >= iq
    save(pd.DataFrame([{"ae_q99": aq, "if_q99": iq, "ae_n": int(aqf.sum()), "if_n": int(iqf.sum()),
                        "common_n": int((aqf & iqf).sum())}]), "ae_if_q99_sensitivity.csv")
    selected = selection["selected_K"]
    if selected is not None:
        focus = joined.loc[train & joined.ae_flag].cluster.value_counts()
        if not focus.empty:
            candidate = int(focus.index[0])
            selection["ae_focus_cluster"] = candidate
            selection["ae_focus_share_train"] = float(focus.iloc[0] / np.sum(a[train]))
            selection["ae_focus_enrichment_train"] = float(
                joined.loc[train & (joined.cluster == candidate), "ae_flag"].mean() / a[train].mean())
    overlap = int(np.sum(a & b))
    return {"status": "complete", "group_counts": np.bincount(groups, minlength=5)[1:].tolist(),
            "overlap": overlap, "ae_n": int(a.sum()), "if_n": int(b.sum()),
            "common_over_ae": float(overlap / a.sum()) if a.any() else None,
            "common_over_if": float(overlap / b.sum()) if b.any() else None,
            "jaccard": float(overlap / np.sum(a | b)) if np.any(a | b) else None,
            "score_spearman": float(spearmanr(score, if_rows.if_42_score).statistic)}


def quality_comparison(raw: pd.DataFrame, quality: pd.DataFrame, clusters: pd.DataFrame,
                       if_daily: pd.DataFrame, ae: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for date, part in raw.groupby("date", sort=True):
        q = quality.loc[quality.date == date]
        types = {int(r["defect type"]): int(r.defect) for _, r in q.iterrows()}
        row = {"date": date, "raw_records": len(part), "recorded_types": len(types),
               "recorded_sum_partial": sum(types.values()) if types else np.nan,
               "complete_3type_sum": sum(types.values()) if len(types) == 3 else np.nan}
        for typ in (1, 2, 3):
            row[f"type{typ}_count"] = types.get(typ, np.nan)
            row[f"type{typ}_per1000records"] = 1000 * types[typ] / len(part) if typ in types else np.nan
        row["complete_sum_per1000records"] = 1000 * row["complete_3type_sum"] / len(part)
        c = clusters.loc[clusters.date == date]
        for k in sorted(clusters.cluster.unique()):
            row[f"cluster{k}_share"] = float((c.cluster == k).mean())
        i = if_daily.loc[if_daily.date == date].iloc[0]
        row.update({"if_flag_share": i.if_flag_share, "if_score_median": i.score_median,
                    "if_score_p95": i.score_p95})
        if ae is not None:
            a = ae.loc[ae.date_raw == date]
            row["ae_flag_share"] = float(a.ae_42_flag.mean())
        rows.append(row)
    daily = pd.DataFrame(rows)
    save(daily, "quality_daily.csv")
    relationships = []
    predictors = [c for c in daily if c.endswith("_share") or c in ("if_score_median", "if_score_p95")]
    targets = [c for c in daily if c.startswith("type") or c in ("complete_3type_sum", "complete_sum_per1000records")]
    for xcol in predictors:
        for ycol in targets:
            pair = daily[[xcol, ycol]].dropna()
            if len(pair) < 5 or pair[xcol].nunique() < 2 or pair[ycol].nunique() < 2:
                relationships.append({"predictor": xcol, "recorded_measure": ycol, "valid_dates": len(pair),
                                      "spearman": np.nan, "leave_one_out_min": np.nan, "leave_one_out_max": np.nan,
                                      "leave_one_out_sign_consistent": np.nan})
                continue
            rho = float(spearmanr(pair[xcol], pair[ycol]).statistic)
            loo = []
            for index in pair.index:
                other = pair.drop(index)
                if other[xcol].nunique() >= 2 and other[ycol].nunique() >= 2:
                    loo.append(float(spearmanr(other[xcol], other[ycol]).statistic))
            relationships.append({"predictor": xcol, "recorded_measure": ycol, "valid_dates": len(pair),
                                  "spearman": rho, "leave_one_out_min": min(loo) if loo else np.nan,
                                  "leave_one_out_max": max(loo) if loo else np.nan,
                                  "leave_one_out_sign_consistent": bool(loo and all(np.sign(v) == np.sign(rho) for v in loo))})
    save(pd.DataFrame(relationships), "quality_exploratory_spearman.csv")
    return daily


def plots(k: pd.DataFrame, clusters: pd.DataFrame, if_rows: pd.DataFrame,
          daily: pd.DataFrame, ae: pd.DataFrame | None) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(k.K, k.inertia, "o-"); axes[0].set(xlabel="K (clusters)", ylabel="Inertia (scaled squared distance)", title="K-Means elbow, training rows")
    axes[1].plot(k.K[1:], k.silhouette_sample5000[1:], "o-"); axes[1].set(xlabel="K (clusters)", ylabel="Silhouette score", title="Silhouette, fixed 5,000-row training sample")
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUTPUT / "k_selection.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col, unit in zip(axes.flat, SHORT, ("bar", "kA", "V", "ms")):
        values = [clusters.loc[clusters.cluster == c, col] for c in sorted(clusters.cluster.unique())]
        ax.boxplot(values, tick_labels=[str(c) for c in sorted(clusters.cluster.unique())], showfliers=False)
        ax.set(xlabel="Cluster", ylabel=f"{col} ({unit})", title=f"{col} by cluster")
    fig.tight_layout(); fig.savefig(OUTPUT / "cluster_sensors.png", dpi=150); plt.close(fig)

    shares = pd.crosstab(clusters.date, clusters.cluster, normalize="index")
    fig, ax = plt.subplots(figsize=(9, 4)); im = ax.imshow(shares, aspect="auto", vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(len(shares.columns)), xticklabels=shares.columns,
           yticks=range(len(shares.index)), yticklabels=shares.index, xlabel="Cluster", ylabel="Date",
           title="Daily cluster share (fraction of Raw rows)")
    fig.colorbar(im, ax=ax, label="Share")
    fig.tight_layout(); fig.savefig(OUTPUT / "daily_cluster_share.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    merged = clusters.merge(if_rows[["excel_row", "if_42_score", "if_42_flag95"]], on="excel_row", validate="one_to_one")
    axes[0].boxplot([merged.loc[merged.cluster == c, "if_42_score"] for c in shares.columns], tick_labels=shares.columns, showfliers=False)
    axes[0].set(xlabel="Cluster", ylabel="IF score (-score_samples)", title="IF score by cluster")
    rates = merged.groupby("date").if_42_flag95.mean()
    axes[1].bar(rates.index, rates.to_numpy()); axes[1].tick_params(axis="x", rotation=45)
    axes[1].set(xlabel="Date", ylabel="IF candidate share of Raw rows", title="Daily IF candidate share")
    fig.tight_layout(); fig.savefig(OUTPUT / "if_scores.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    target = "complete_sum_per1000records"
    for ax, col in zip(axes.flat, [f"cluster{v}_share" for v in shares.columns][:2] + ["if_flag_share", "if_score_median"]):
        m = daily[[col, target, "date"]].dropna()
        ax.scatter(m[col], m[target]);
        for _, row in m.iterrows(): ax.annotate(row.date[5:], (row[col], row[target]), fontsize=8)
        ax.set(xlabel=col, ylabel="Recorded counts / 1,000 Raw rows", title=f"Exploratory: {col} vs recorded counts")
    fig.tight_layout(); fig.savefig(OUTPUT / "quality_exploratory.png", dpi=150); plt.close(fig)

    if ae is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(ae.ae_42_score, if_rows.if_42_score, s=5, alpha=.3)
        ax.set(xlabel="AE row MSE", ylabel="IF score (-score_samples)", title="AE and IF continuous scores")
        fig.tight_layout(); fig.savefig(OUTPUT / "ae_if_scores.png", dpi=150); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    p.add_argument("--ae-row-scores", type=Path, default=None)
    args = p.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw, quality, source = load(args.zip)
    x = raw[COLS].to_numpy(float)
    clusters, _, selection = kmeans(raw, x)
    if_rows, if_daily, if_info = isolation_forest(raw, x)
    cluster_if_comparison(raw, clusters, if_rows)
    ae, ae_status = get_ae(raw, args.ae_row_scores)
    compare = ae_comparison(raw, clusters, if_rows, ae, selection)
    daily = quality_comparison(raw, quality, clusters, if_daily, ae)
    plots(pd.read_csv(OUTPUT / "k_diagnostics.csv"), clusters, if_rows, daily, ae)
    manifest = {"source": source, "python": sys.version, "pandas": pd.__version__,
                "numpy": np.__version__, "sklearn": __import__("sklearn").__version__,
                "run_command": "./.venv/Scripts/python.exe -B experiments/NBJ/nbj-004/analyze.py",
                "K": selection, "IF": if_info, "AE": compare, "ae_status": ae_status,
                "raw_rows": len(raw), "quality_rows": len(quality)}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"K": selection, "IF": if_info, "AE": compare}, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
