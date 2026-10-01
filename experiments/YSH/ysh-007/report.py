from __future__ import annotations

import json
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common import HERE, OUT, read_config

FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DATES = read_config()["dates"]
LABELS = [d[5:] for d in DATES]


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=170, bbox_inches="tight")
    plt.close()


def raw4():
    raw = pd.read_csv(OUT / "daily_raw4.csv").set_index("date").loc[DATES]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, f in zip(axes.ravel(), ["F", "I", "V", "t"]):
        ax.plot(LABELS, raw[f"{f}_median"], marker="o", color="#176b87")
        ax.fill_between(LABELS, raw[f"{f}_p10"].to_numpy(),
                        raw[f"{f}_p90"].to_numpy(), alpha=.18, color="#176b87")
        ax.set_title(f"{f}: median and p10-p90")
        ax.grid(alpha=.2)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Raw4 distributions by date")
    save("raw4_by_date.png")


def selection():
    frame = pd.read_csv(OUT / "global_model_selection.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for (scaler, cov), d in frame.groupby(["scaler", "covariance_type"]):
        d = d.sort_values("K")
        label = f"{scaler}/{cov}"
        axes[0].plot(d.K, d.mean_val_loglik, marker="o", label=label)
        axes[1].plot(d.K, d.mean_train_bic, marker="o", label=label)
    axes[0].set(xlabel="K", ylabel="Mean validation log density (Raw units)")
    axes[1].set(xlabel="K", ylabel="Mean train BIC (Raw units)")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    save("global_model_selection.png")


def components():
    d = pd.read_csv(OUT / "daily_component_share.csv")
    d = d[d.phase.eq("lodo")].set_index("date").loc[DATES]
    columns = [f"C{i}_posterior_share" for i in range(6)]
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(d[columns].to_numpy(), cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set(xticks=np.arange(6), xticklabels=[f"C{i}" for i in range(6)],
           yticks=np.arange(9), yticklabels=LABELS,
           title="LODO mean posterior component share")
    fig.colorbar(im, ax=ax, label="Share")
    save("lodo_component_share.png")


def scores():
    d = pd.read_csv(OUT / "daily_score_summary.csv")
    d = d[d.phase.eq("lodo")].set_index("date").loc[DATES]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, field, label in zip(axes, ["global_nll", "mahalanobis_d2"],
                                ["Global negative log density", "Assigned component Mahalanobis d2"]):
        ax.plot(LABELS, d[f"{field}_median"], marker="o", label="median")
        ax.plot(LABELS, d[f"{field}_p95"], marker="o", label="p95")
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend()
    axes[1].tick_params(axis="x", rotation=45)
    save("lodo_score_distribution.png")


def residuals():
    d = pd.read_csv(OUT / "residual_pattern_models.csv")
    d = d[d.phase.eq("reference_in_sample") & d.policy.eq("global")].sort_values("pattern")
    cols = ["residual_F", "residual_I", "residual_V", "residual_t"]
    if not set(cols).issubset(d.columns):
        print("Residual profile plot skipped: columns", list(d.columns))
        return
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    for ax, c in zip(axes, cols):
        ax.bar(d.pattern.astype(str), d[c], color="#176b87")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(xlabel="Pattern", title=c)
        ax.grid(axis="y", alpha=.2)
    save("residual_profile.png")


def quality():
    d = pd.read_csv(OUT / "regime_confounding.csv").set_index("date").loc[DATES]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, defect in zip(axes, ["type1", "type2", "type3"]):
        valid = d[defect].notna()
        colors = d.loc[valid, "regime"].map({"A": "#176b87", "B": "#d57b31"})
        ax.scatter(d.loc[valid, "global_nll_p95"], d.loc[valid, defect], c=colors, s=45)
        for date, r in d.loc[valid].iterrows():
            ax.annotate(date[5:], (r["global_nll_p95"], r[defect]), fontsize=7)
        ax.set(title=f"{defect} recorded count", xlabel="LODO NLL p95",
               ylabel="Result count")
        ax.grid(alpha=.2)
    save("result_vs_lodo_score.png")


def row_clusters():
    cfg = read_config()
    raw = pd.read_excel(HERE.parents[2] / "data" / "Welding_Data_Set_01.xlsx",
                        sheet_name="Raw data")
    scored = pd.read_csv(OUT / "in_sample_scores.csv").sort_values("excel_row")
    raw["excel_row"] = np.arange(len(raw)) + 2
    assert len(raw) == len(scored) == 11939
    assert np.array_equal(raw.excel_row, scored.excel_row)
    x = raw[cfg["features"]].to_numpy(float)
    payload = joblib.load(OUT / "model_full_descriptive.joblib")
    z = payload["scaler"].transform(x)
    pca = PCA(n_components=4).fit(z)
    pc = pca.transform(z)
    labels = scored.assigned_component.to_numpy(int)
    counts = np.bincount(labels, minlength=payload["model"].n_components)
    summary = []
    for k in range(payload["model"].n_components):
        mask = labels == k
        summary.append({
            "component": k, "rows": int(mask.sum()),
            "row_share": float(mask.mean()),
            "unique_raw4": int(pd.DataFrame(x[mask]).drop_duplicates().shape[0]),
            "mean_posterior_max": float(scored.loc[mask, "posterior_max"].mean()),
            **{f"mean_{f}": float(x[mask, j].mean())
               for j, f in enumerate(["F", "I", "V", "t"])},
        })
    pd.DataFrame(summary).to_csv(OUT / "row_cluster_summary.csv", index=False,
                                 float_format="%.12g")
    positions = pd.DataFrame({
        "excel_row": raw.excel_row, "component": labels,
        **{f"PC{i+1}": pc[:, i] for i in range(4)},
    })
    positions.to_csv(OUT / "row_cluster_pca.csv", index=False,
                     float_format="%.12g")
    color = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    order = np.argsort(-counts)
    views = [
        (axes[0, 0], pc[:, 0], pc[:, 1], "PC1", "PC2"),
        (axes[0, 1], pc[:, 2], pc[:, 3], "PC3", "PC4"),
        (axes[1, 0], x[:, 0], x[:, 2], "F (bar)", "V (V)"),
        (axes[1, 1], x[:, 1], x[:, 3], "I (kA)", "t (ms)"),
    ]
    for ax, xx, yy, xlabel, ylabel in views:
        for k in order:
            mask = labels == k
            ax.scatter(xx[mask], yy[mask], s=5, alpha=.22, color=color[k],
                       linewidths=0, rasterized=True,
                       label=f"C{k}: {counts[k]:,} rows")
        if xlabel.startswith("PC"):
            ax.set_xlabel(f"{xlabel} ({pca.explained_variance_ratio_[int(xlabel[2])-1]:.1%})")
            ax.set_ylabel(f"{ylabel} ({pca.explained_variance_ratio_[int(ylabel[2])-1]:.1%})")
        else:
            ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.16)
    axes[0, 0].legend(loc="best", fontsize=8, markerscale=3, framealpha=.9)
    fig.suptitle("All 11,939 Raw4 rows: full-fit GMM hard assignment", fontsize=14)
    save("row_gmm_clusters.png")
    repeated = pd.DataFrame({
        "F": x[:, 0], "I": x[:, 1], "V": x[:, 2], "t": x[:, 3],
        "component": labels, "PC1": pc[:, 0], "PC2": pc[:, 1],
    }).groupby(["F", "I", "V", "t", "component"], as_index=False).agg(
        rows=("PC1", "size"), PC1=("PC1", "first"), PC2=("PC2", "first"))
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in order:
        part = repeated[repeated.component.eq(k)]
        ax.scatter(part.PC1, part.PC2, s=5 * part.rows,
                   color=color[k], alpha=.5, edgecolors="none",
                   label=f"C{k}: {counts[k]:,} rows")
    ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
           title="1,574 distinct Raw4 values; bubble size shows repeat count")
    ax.grid(alpha=.16)
    ax.legend(fontsize=8, markerscale=.7)
    save("row_gmm_repeat_bubbles.png")

if __name__ == "__main__":
    raw4()
    selection()
    components()
    scores()
    residuals()
    quality()
    row_clusters()
    print("Figures:", *sorted(p.name for p in FIG.glob("*.png")))








