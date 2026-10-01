"""Render an ordered 2020-03-25 idx 700..1000 review from saved YSH-007 outputs."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from common import OUT, load_inputs, read_config


def main() -> None:
    config = read_config()
    raw, _, _, _, _, _ = load_inputs(config)
    scored = pd.read_csv(OUT / "physical_row_followup.csv")
    full = pd.read_csv(OUT / "in_sample_scores.csv")[
        ["excel_row", "assigned_component"]
    ].rename(columns={"assigned_component": "component_full"})
    window = raw.loc[
        raw["date"].eq("2020-03-25") & raw["idx"].between(700, 1000),
        ["excel_row", "idx", "date"]
    ].merge(scored.drop(columns=["date"]), on="excel_row", validate="one_to_one")
    window = window.merge(full, on="excel_row", validate="one_to_one").sort_values("idx")
    if len(window) != 301 or not window.idx.is_unique or not np.array_equal(
        window.idx.to_numpy(), np.arange(700, 1001)
    ):
        raise ValueError("Unexpected 2020-03-25 idx range")
    cutoffs = pd.read_csv(OUT / "physical_signature_cutoffs.csv")
    cutoffs = cutoffs[cutoffs.held_out_date.eq("2020-03-25")].set_index(
        ["column", "quantile"])["train_cutoff"]
    def q(name: str, percentile: float) -> float:
        return float(cutoffs.loc[(name, percentile)])

    fig, axes = plt.subplots(
        8, 1, figsize=(17, 15), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, .42, .9, .7],
                     "hspace": .13}
    )
    x = window.idx.to_numpy()
    review = window.row_review_candidate.to_numpy(bool)
    colors = {"F": "#b05749", "I": "#377eb8", "V": "#589b77",
              "t": "#9270b0", "E_proxy_j": "#a37a2e"}
    panels = [
        ("F", "F (bar)", [(.75, "q75 F")]),
        ("I", "I (kA)", [(.25, "q25 I"), (.75, "q75 I")]),
        ("V", "V (V)", []),
        ("t", "t (ms)", [(.75, "q75 t")]),
        ("E_proxy_j", "E proxy (V*A*s)", [(.10, "q10 E"),
                                         (.25, "q25 E"), (.90, "q90 E")]),
    ]
    for ax, (column, label, limits) in zip(axes[:5], panels):
        ax.plot(x, window[column], color=colors[column], lw=1.15)
        ax.scatter(x[review], window.loc[review, column], s=12, c="black",
                   zorder=5, label="final review row" if column == "F" else None)
        for percentile, caption in limits:
            cutoff = q(column, percentile)
            ax.axhline(cutoff, color="#666666", ls="--", lw=.8, alpha=.6)
            ax.annotate(caption, xy=(.997, cutoff), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=8, color="#555555")
        ax.set_ylabel(label, fontsize=10)
        ax.grid(alpha=.16)
    axes[0].legend(loc="upper right", fontsize=8)

    cmap = ListedColormap(["#674ea7", "#3d85c6", "#6aa84f",
                           "#e69138", "#999999", "#c27ba0"])
    norm = BoundaryNorm(np.arange(-.5, 6.5, 1), cmap.N)
    matrix = np.vstack([window.component_full.to_numpy(),
                        window.assigned_component.to_numpy()])
    axes[5].imshow(matrix, aspect="auto", interpolation="nearest",
                   extent=(699.5, 1000.5, -.5, 1.5), origin="lower",
                   cmap=cmap, norm=norm)
    axes[5].set_yticks([0, 1], ["Full-fit C", "LODO C"])
    axes[5].set_ylabel("GMM", fontsize=10)
    axes[5].grid(axis="x", alpha=.15)
    fig.legend(
        [Patch(facecolor=cmap(j), label=f"C{j}") for j in range(6)],
        [f"C{j}" for j in range(6)],
        loc="upper center", bbox_to_anchor=(.5, .965),
        ncol=6, fontsize=9, title="GMM component colors"
    )

    axes[6].plot(x, window.global_nll_train_percentile,
                 color="#db8d38", lw=1, label="Global NLL train percentile")
    axes[6].plot(x, window.mahalanobis_d2_train_percentile,
                 color="#356ea6", lw=1, label="Within-component d2 train percentile")
    axes[6].axhline(.90, color="#555555", ls="--", lw=.9, alpha=.7)
    axes[6].set_ylim(0, 1.04)
    axes[6].set_ylabel("LODO score", fontsize=10)
    axes[6].legend(loc="lower left", ncol=2, fontsize=8)
    axes[6].grid(alpha=.16)

    flag_info = [
        ("Final review", "row_review_candidate", "#222222"),
        ("Indentation", "indentation_candidate", "#bc5145"),
        ("Insufficient", "insufficient_candidate", "#377eb8"),
        ("Crack-related", "crack_related_candidate", "#8f67ab"),
    ]
    for y, (name, field, color) in enumerate(flag_info[::-1]):
        mask = window[field].to_numpy(bool)
        axes[7].scatter(x[mask], np.full(mask.sum(), y), marker="|",
                        s=70, linewidths=1.7, color=color)
    axes[7].set_yticks(range(4), [name for name, _, _ in flag_info[::-1]], fontsize=8)
    axes[7].set_ylim(-.6, 3.6)
    axes[7].grid(axis="x", alpha=.16)
    axes[7].set_xlabel("Raw idx within 2020-03-25")

    for ax in axes:
        ax.axvspan(769.5, 774.5, color="#efdba8", alpha=.3, zorder=-1)
        ax.axvspan(815.5, 826.5, color="#efdba8", alpha=.3, zorder=-1)
        ax.set_xlim(699.5, 1000.5)
    axes[-1].set_xticks(np.arange(700, 1001, 25))
    fig.suptitle(
        "2020-03-25  |  Raw idx 700-1000  |  22 final review rows / 301 rows",
        fontsize=15, y=.992
    )
    fig.text(.5, .008,
             "Shaded: final-review runs idx 770-774 and 816-826. "
             "Signature rows are physical screening conditions, not observed defects.",
             ha="center", fontsize=9, color="#555555")
    OUT.joinpath("figures").mkdir(exist_ok=True)
    target = OUT / "figures" / "physical_20200325_idx700_1000.png"
    fig.savefig(target, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(target)
    print("counts", {field: int(window[field].sum()) for _, field, _ in flag_info})


if __name__ == "__main__":
    main()