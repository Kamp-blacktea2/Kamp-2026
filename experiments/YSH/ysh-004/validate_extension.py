"""Bounded scaling and temporal sensitivity checks for YSH-004."""

import os

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "4"

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import LocalOutlierFactor
from threadpoolctl import threadpool_limits

from analyze import (
    HERE,
    NAMES,
    X3,
    condition_indices,
    digest,
    gmm_grid,
    load_config,
    load_inputs,
    top_set,
)

OUT = HERE / "outputs/validation"
COLLECT = {}
FITS = {}


def record(table, **row):
    COLLECT.setdefault(table, []).append(row)


def persist():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in COLLECT.items():
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False, float_format="%.17g")
    (OUT / "fit_parameters.json").write_text(
        json.dumps(FITS, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def scale_fit(x, kind):
    center = x.mean(axis=0) if kind == "standard" else np.median(x, axis=0)
    spread = (
        x.std(axis=0) if kind == "standard" else np.subtract(*np.percentile(x, [75, 25], axis=0))
    )
    fallback = spread == 0
    spread[fallback] = x.std(axis=0)[fallback]
    active = spread > 0
    return center, spread, active, fallback


def transform(x, scale):
    center, spread, active, _ = scale
    return (x[:, active] - center[active]) / spread[active]


def compare(a, b):
    aa, _ = top_set(a, 0.05)
    bb, _ = top_set(b, 0.05)
    return dict(
        spearman=float(spearmanr(a, b).statistic),
        jaccard=float((aa & bb).sum() / (aa | bb).sum()),
        selected_a=int(aa.sum()),
        selected_b=int(bb.sum()),
    )


def pca_scores(z):
    model = PCA(n_components=z.shape[1], svd_solver="full").fit(z)
    score = model.transform(z)
    k = min(z.shape[1], int(np.searchsorted(np.cumsum(model.explained_variance_ratio_), 0.9) + 1))
    q = {}
    parts = {}
    for kk in range(1, z.shape[1] + 1):
        delta = z - score[:, :kk] @ model.components_[:kk] - model.mean_
        parts[kk] = delta**2
        q[kk] = parts[kk].sum(axis=1)
    eigen = model.explained_variance_[:k]
    valid = eigen > 1e-12
    t2 = np.sum(score[:, :k][:, valid] ** 2 / eigen[valid], axis=1)
    return model, k, q, parts, t2


def fit_gmm(x, ids, condition, scope, kind, rep, cols):
    scale = scale_fit(x, kind)
    z = transform(x, scale)
    grid, chosen, selected = gmm_grid(z)
    if selected is None:
        raise RuntimeError(f"No converged GMM: {condition}/{scope}/{kind}/{rep}")
    fixed = selected if chosen == 6 else gmm_grid(z, fixed_k=6)[2]
    if fixed is None:
        raise RuntimeError("Fixed K6 fit did not converge")
    for row in grid:
        record("gmm_grid", condition=condition, scope=scope, scaler=kind, representation=rep, **row)
    models = {"fixed6": fixed, "selected": selected}
    for selection, model in models.items():
        key = "|".join([condition, scope, kind, rep, selection])
        FITS[key] = dict(
            condition=condition,
            scope=scope,
            scaler=kind,
            representation=rep,
            selection=selection,
            cols=cols,
            fit_excel_rows=(ids + 2).tolist(),
            center=scale[0].tolist(),
            spread=scale[1].tolist(),
            active=scale[2].tolist(),
            fallback=scale[3].tolist(),
            chosen_k=chosen,
            weights=model.weights_.tolist(),
            means=model.means_.tolist(),
            covariances=model.covariances_.tolist(),
        )
    return scale, models, chosen


def posterior_frame(model, x, scale, ids):
    z = transform(x, scale)
    probability = model.predict_proba(z)
    assert np.allclose(probability.sum(axis=1), 1)
    return pd.DataFrame(
        dict(
            excel_row=ids + 2,
            state=probability.argmax(axis=1),
            max_p=probability.max(axis=1),
            log_density=model.score_samples(z),
            probability_sum=probability.sum(axis=1),
        )
    ).set_index("excel_row")


def classify_window(frame):
    confident = frame.max_p.to_numpy() >= 0.8
    counts = np.bincount(frame.state.to_numpy(int)[confident])
    label = int(counts.argmax()) if len(counts) and counts.max() >= 0.8 * len(frame) else -1
    return label, bool(confident.all())


def integrate(frame, pairs, condition, scope, kind, selection):
    for row in pairs.itertuples():
        a = frame.loc[np.arange(row.a_excel, row.a_excel + row.length)]
        b = frame.loc[np.arange(row.b_excel, row.b_excel + row.length)]
        state_a, certain_a = classify_window(a)
        state_b, certain_b = classify_window(b)
        if min(state_a, state_b) < 0:
            category = "mixed_or_uncertain"
            refined = "mixed_confident" if certain_a and certain_b else "mixed_with_low_confidence"
        else:
            category = refined = "same" if state_a == state_b else "different"
        record(
            "voltage_pair_validation",
            condition=condition,
            scope=scope,
            scaler=kind,
            selection=selection,
            length=row.length,
            offset=row.offset,
            segment_id=row.segment,
            a_excel=row.a_excel,
            b_excel=row.b_excel,
            state_a=state_a,
            state_b=state_b,
            category=category,
            refined_category=refined,
            all_rows_confident=certain_a and certain_b,
            passed=bool(row.passed),
        )


def run():
    started = time.perf_counter()
    config = load_config(HERE / "config.json")
    _, values, segments, _, raw_hash = load_inputs(config)
    previous = pd.read_csv(HERE / "outputs/tables/row_diagnostics.csv")
    pairs = pd.read_csv(config["previous_pairs"])
    pairs = pairs[
        (pairs.variable == "voltage")
        & pairs.length.isin([15, 16, 17])
        & (pairs.phase == "confirmation")
    ]
    caches = {}
    for condition in ["all", "controlled100", "unique4"]:
        ids = condition_indices(segments, condition, values)
        x = values[ids]
        for kind in ["standard", "robust"]:
            print(f"FULL {condition} {kind}", flush=True)
            scale = scale_fit(x, kind)
            z = transform(x, scale)
            pca, k, q, parts, t2 = pca_scores(z)
            iso = IsolationForest(
                n_estimators=300, max_samples=min(256, len(x)), random_state=42, n_jobs=4
            ).fit(z)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                lof = LocalOutlierFactor(n_neighbors=20, n_jobs=4).fit(z)
            scores = {
                "IF": -iso.score_samples(z),
                "LOF": -lof.negative_outlier_factor_,
                "PCA_T2": t2,
            }
            scores.update({f"PCA_Q{kk}": score for kk, score in q.items() if kk < z.shape[1]})
            record(
                "pca_summary",
                condition=condition,
                scaler=kind,
                k90=k,
                explained=json.dumps(pca.explained_variance_ratio_.tolist()),
                force_variance=float(np.var(z[:, 0])),
                warnings=";".join(str(w.message) for w in caught),
            )
            FITS[f"{condition}|full|{kind}|PCA"] = dict(
                center=scale[0].tolist(),
                spread=scale[1].tolist(),
                active=scale[2].tolist(),
                fit_excel_rows=(ids + 2).tolist(),
                components=pca.components_.tolist(),
                pca_mean=pca.mean_.tolist(),
                eigenvalues=pca.explained_variance_.tolist(),
                k90=k,
            )
            row_frame = pd.DataFrame(
                dict(condition=condition, scaler=kind, excel_row=ids + 2, **scores)
            )
            for kk in range(1, 4):
                for j, name in enumerate(NAMES):
                    row_frame[f"Q{kk}_{name}"] = parts[kk][:, j]
            COLLECT.setdefault("row_scores", []).extend(row_frame.to_dict("records"))
            caches[(condition, kind, "scores")] = scores
            if kind == "standard":
                old = previous[previous.condition == condition].set_index("excel_row").loc[ids + 2]
                for name, column in [("IF", "if_score"), ("LOF", "lof_score"), ("PCA_T2", "t2")]:
                    record(
                        "baseline_reproduction",
                        condition=condition,
                        model=name,
                        max_absolute_error=float(
                            np.max(np.abs(scores[name] - old[column].to_numpy()))
                        ),
                        **compare(scores[name], old[column].to_numpy()),
                    )
            for rep, cols in [("X3", X3), ("X4", [0, 1, 2, 3])]:
                sc, models, chosen = fit_gmm(x[:, cols], ids, condition, "full", kind, rep, cols)
                for selection, gm in models.items():
                    frame = posterior_frame(gm, x[:, cols], sc, ids)
                    caches[(condition, kind, rep, selection)] = frame
                    record(
                        "gmm_summary",
                        condition=condition,
                        scope="full",
                        scaler=kind,
                        representation=rep,
                        selection=selection,
                        n=len(ids),
                        k=gm.n_components,
                        uncertain_pct=float(100 * (frame.max_p < 0.8).mean()),
                    )
                    if rep == "X3" and condition != "unique4":
                        integrate(
                            frame,
                            pairs[pairs.condition == condition],
                            condition,
                            "full",
                            kind,
                            selection,
                        )
                if kind == "standard":
                    old = (
                        previous[previous.condition == condition]
                        .set_index("excel_row")
                        .loc[ids + 2]
                    )
                    record(
                        "baseline_reproduction",
                        condition=condition,
                        model=f"GMM_{rep}",
                        ari=adjusted_rand_score(
                            old[f"{rep}_state"], caches[(condition, kind, rep, "fixed6")].state
                        ),
                    )
            persist()
        for model, score in caches[(condition, "standard", "scores")].items():
            record(
                "scale_comparison",
                condition=condition,
                model=model,
                representation="X4",
                selection="fixed",
                n=len(ids),
                **compare(score, caches[(condition, "robust", "scores")][model]),
            )
        for rep in ["X3", "X4"]:
            for selection in ["fixed6", "selected"]:
                a = caches[(condition, "standard", rep, selection)]
                b = caches[(condition, "robust", rep, selection)]
                record(
                    "scale_comparison",
                    condition=condition,
                    model="GMM",
                    representation=rep,
                    selection=selection,
                    n=len(ids),
                    ari=adjusted_rand_score(a.state, b.state),
                )

    for condition in ["all", "controlled100"]:
        front, rear = [], []
        for row in segments[segments.condition == condition].itertuples():
            ids = np.arange(row.excel_start - 2, row.excel_end - 1)
            midpoint = len(ids) // 2
            front.extend(ids[:midpoint])
            rear.extend(ids[midpoint:])
        front, rear = np.asarray(front), np.asarray(rear)
        assert len(np.intersect1d(front, rear)) == 0
        for kind in ["standard", "robust"]:
            print(f"TEMPORAL {condition} {kind}", flush=True)
            scale, models, chosen = fit_gmm(
                values[front][:, X3], front, condition, "front", kind, "X3", X3
            )
            for selection, gm in models.items():
                frame = posterior_frame(gm, values[rear][:, X3], scale, rear)
                full = caches[(condition, kind, "X3", selection)].loc[rear + 2]
                train_density = gm.score_samples(transform(values[front][:, X3], scale))
                cutoff = np.quantile(train_density, 0.01)
                record(
                    "temporal_summary",
                    condition=condition,
                    scaler=kind,
                    selection=selection,
                    train_n=len(front),
                    holdout_n=len(rear),
                    k=gm.n_components,
                    ari_vs_full=adjusted_rand_score(full.state, frame.state),
                    uncertain_pct=float(100 * (frame.max_p < 0.8).mean()),
                    low_density_pct=float(100 * (frame.log_density < cutoff).mean()),
                    train_density_p01=float(cutoff),
                )
                score_rows = frame.reset_index().assign(
                    condition=condition, scaler=kind, selection=selection
                )
                COLLECT.setdefault("temporal_row_states", []).extend(score_rows.to_dict("records"))
                integrate(
                    frame, pairs[pairs.condition == condition], condition, "front", kind, selection
                )
            persist()

    # Complete the originally omitted PCA leave-one-fifth-out sensitivity.
    for fold in range(5):
        removed = []
        for row in segments[(segments.condition == "all") & (segments.n >= 200)].itertuples():
            ids = np.arange(row.excel_start - 2, row.excel_end - 1)
            removed.extend(np.array_split(ids, 5)[fold])
        ids = np.setdiff1d(np.arange(len(values)), removed)
        for kind in ["standard", "robust"]:
            sc = scale_fit(values[ids], kind)
            _, k, q, _, t2 = pca_scores(transform(values[ids], sc))
            original = caches[("all", kind, "scores")]
            for kk in [1, 2, 3]:
                record(
                    "pca_subset_stability",
                    scaler=kind,
                    fold=fold + 1,
                    n=len(ids),
                    model=f"PCA_Q{kk}",
                    **compare(original[f"PCA_Q{kk}"][ids], q[kk]),
                )
            # T2 uses the full-fit retained dimension, not a newly chosen dimension.
            params = FITS[f"all|full|{kind}|PCA"]
            refit = PCA(n_components=4, svd_solver="full").fit(transform(values[ids], sc))
            kk = params["k90"]
            score = refit.transform(transform(values[ids], sc))[:, :kk]
            fixed_t2 = np.sum(score**2 / refit.explained_variance_[:kk], axis=1)
            record(
                "pca_subset_stability",
                scaler=kind,
                fold=fold + 1,
                n=len(ids),
                model="PCA_T2",
                **compare(original["PCA_T2"][ids], fixed_t2),
            )

    persist()
    detail = pd.DataFrame(COLLECT["voltage_pair_validation"])
    for column, filename in [
        ("category", "voltage_summary"),
        ("refined_category", "voltage_refined"),
    ]:
        grouped = (
            detail.groupby(["condition", "scope", "scaler", "selection", "length", column])
            .agg(
                pairs=("passed", "size"),
                passed=("passed", "sum"),
                segments=("segment_id", "nunique"),
            )
            .reset_index()
        )
        grouped["pass_pct"] = 100 * grouped.passed / grouped.pairs
        grouped.to_csv(OUT / f"{filename}.csv", index=False)
    by_state = (
        detail[(detail.category == "same")]
        .groupby(["condition", "scope", "scaler", "selection", "length", "state_a"])
        .agg(pairs=("passed", "size"), passed=("passed", "sum"), segments=("segment_id", "nunique"))
        .reset_index()
    )
    by_state["pass_pct"] = 100 * by_state.passed / by_state.pairs
    by_state["supported"] = (by_state.pairs >= 30) & (by_state.segments >= 2)
    by_state.to_csv(OUT / "voltage_specific_state.csv", index=False)
    assert digest(config["raw"]) == raw_hash
    manifest = dict(
        completed=True,
        seconds=time.perf_counter() - started,
        python=sys.version,
        raw_sha256=raw_hash,
        seed=42,
        threads=4,
        scope="Scaling and within-segment front/rear sensitivity, not independent quality validation",
        files={
            str(p): digest(p)
            for p in [
                Path(__file__),
                HERE / "analyze.py",
                HERE / "실험계획.md",
                HERE / "config.json",
            ]
        },
        packages={
            name: __import__(name).__version__ for name in ["numpy", "pandas", "scipy", "sklearn"]
        },
    )
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f'COMPLETED {manifest["seconds"]:.1f}s', flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
