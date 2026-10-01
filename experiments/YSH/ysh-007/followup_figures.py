"""Static review figures for the post-hoc YSH-007 physical audit."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import OUT

def save(fig, name):
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / name, dpi=180, bbox_inches="tight")
    plt.close(fig)

def main():
    daily = pd.read_csv(OUT / "physical_daily_result.csv")
    profile = pd.read_csv(OUT / "physical_component_profiles.csv")
    lag = pd.read_csv(OUT / "physical_sequence_sensitivity.csv")
    neighbors = pd.read_csv(OUT / "physical_neighbor_summary.csv")
    dates = pd.to_datetime(daily.date).dt.strftime("%m-%d")
    x = np.arange(len(daily))
    fig, (ax, ay) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    specs = [("indentation", "#bb4d45"), ("insufficient", "#397ab4"),
             ("crack_related", "#8a63a5")]
    for k, (name, color) in enumerate(specs):
        ax.bar(x + (k-1)*.25, daily[name+"_rows"]/daily.rows, width=.24,
               label=name+" candidate", color=color)
    ax.set_ylabel("Candidate share of Raw rows")
    ax.set_ylim(bottom=0)
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=.2)
    for j, col in enumerate(("type1", "type2", "type3")):
        ay.plot(x, daily[col], marker="o", label=col, color=specs[j][1])
    ay.set_ylabel("Recorded Result count")
    ay.set_xticks(x, [f"{d}\n{r}" for d, r in zip(dates, daily.regime)])
    ay.grid(axis="y", alpha=.2)
    ay.legend(ncol=3, fontsize=9)
    fig.suptitle("Physical screening signatures and date-level Result")
    fig.tight_layout()
    save(fig, "physical_daily_signatures_result.png")

    grouped = lag.groupby("lag")[["same","pairs","unique_endpoint_same","unique_endpoint_pairs"]].sum()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(grouped.index, grouped.same/grouped.pairs, marker="o",
            label="all original pairs", color="#376a9a")
    ax.plot(grouped.index, grouped.unique_endpoint_same/grouped.unique_endpoint_pairs,
            marker=".", label="unique Raw4 endpoints", color="#b77b39")
    ax.axvline(3, color="#a33", linestyle="--", linewidth=1)
    ax.set(xlabel="Row lag within safe segment", ylabel="Same assigned component rate",
           title="Component recurrence by row lag (not a physical cycle ID)")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, "physical_sequence_lags.png")

    avg = neighbors.groupby("width")[["previous_review_share","current_review_share",
                                      "next_review_share"]].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    places = np.arange(len(avg))
    for k, (col, color) in enumerate(zip(avg.columns, ("#9bb8cd","#b85549","#9ec7a4"))):
        ax.bar(places+(k-1)*.24, avg[col], .23, label=col.replace("_review_share",""),
               color=color)
    ax.set_xticks(places, [str(i) for i in avg.index])
    ax.set(xlabel="Window width (rows)", ylabel="Mean review-candidate share",
           title="Before/current/after windows around selected rows")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    save(fig, "physical_neighbor_context.png")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    sizes = np.sqrt(profile.rows) * 6
    ax.scatter(profile.F_median, profile.E_proxy_j_median, s=sizes,
               c=profile.insufficient_share, cmap="viridis", alpha=.8,
               edgecolor="black", linewidth=.7)
    for rec in profile.itertuples():
        ax.annotate("C"+str(rec.component_full), (rec.F_median, rec.E_proxy_j_median),
                    xytext=(5, 5), textcoords="offset points")
    ax.set(xlabel="Median F (bar)", ylabel="Median E proxy (V*A*s)",
           title="Full-fit component profiles; size = support")
    ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig, "physical_component_proxy.png")
    print("4 follow-up figures saved")

if __name__ == "__main__":
    main()