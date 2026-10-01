from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

RAW_PATH = ROOT / "data" / "Welding_Data_Set_01.xlsx"
Y7 = ROOT / "experiments" / "YSH" / "ysh-007" / "outputs"
Y8 = ROOT / "experiments" / "YSH" / "ysh-008" / "outputs"

plt.rcParams.update(
    {
        "font.family": "Malgun Gothic",
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

FEATURES = [
    ("weld force(bar)", "F", "bar", "#1f77b4"),
    ("weld current(kA)", "I", "kA", "#ff7f0e"),
    ("weld Voltage(v)", "V", "V", "#2ca02c"),
    ("weld time(ms)", "t", "ms", "#9467bd"),
]
B_DATES = ["2020-03-25", "2020-03-27", "2020-03-31", "2020-04-03"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def contiguous_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    padded = np.r_[False, mask, False].astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def save_figure(fig: plt.Figure, filename: str) -> Path:
    path = FIG / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


raw = pd.read_excel(RAW_PATH, sheet_name="Raw data")
raw.insert(0, "excel_row", np.arange(2, len(raw) + 2))
raw["date"] = pd.to_datetime(raw["working time"]).dt.strftime("%Y-%m-%d")

scores = pd.read_csv(Y7 / "in_sample_scores.csv")
physical = pd.read_csv(Y7 / "physical_row_followup.csv")
daily = pd.read_csv(Y7 / "physical_daily_result.csv")
association = pd.read_csv(Y7 / "physical_result_association.csv")
mapping = pd.read_csv(Y8 / "repeat298_mapping.csv")
relative = pd.read_csv(Y8 / "ae_gmm_relative_profile.csv")
parent_blocks = pd.read_csv(Y8 / "repeat298_parent_blocks.csv")
decomposition = pd.read_csv(Y8 / "repeat298_gmm_decomposition.csv")

assert len(raw) == 11939
assert len(scores) == len(raw) == len(physical)
assert scores["excel_row"].is_unique and physical["excel_row"].is_unique
assert len(mapping) == 4 * 298
assert mapping["relative_position"].nunique() == 298


# Figure 1: raw segment and arithmetic idx phases.  No cycle label is used.
sample = raw.loc[raw["date"].eq("2020-03-24")].head(96).copy()
x = sample["excel_row"].to_numpy()
fig, axes = plt.subplots(
    7,
    1,
    figsize=(16, 13),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.48, 0.48, 0.48], "hspace": 0.12},
)
for ax, (col, label, unit, color) in zip(axes[:4], FEATURES):
    ax.plot(x, sample[col], color=color, linewidth=1.25)
    ax.set_ylabel(f"{label} ({unit})")
    for modulus, alpha, width in [(4, 0.05, 0.5), (8, 0.09, 0.65), (16, 0.18, 0.9)]:
        boundaries = sample.loc[sample["idx"].mod(modulus).eq(0), "excel_row"] + 0.5
        for boundary in boundaries:
            ax.axvline(boundary, color="black", alpha=alpha, linewidth=width)
for ax, modulus, color in zip(axes[4:], [4, 8, 16], ["#4c78a8", "#f58518", "#54a24b"]):
    ax.step(x, sample["idx"].mod(modulus), where="mid", color=color, linewidth=1)
    ax.scatter(x, sample["idx"].mod(modulus), s=7, color=color)
    ax.set_ylabel(f"idx\nmod {modulus}", rotation=0, ha="right", va="center")
    ax.set_yticks([0, modulus - 1])
axes[-1].set_xlabel("Excel row (representative original segment: 2020-03-24, first 96 rows)")
fig.suptitle(
    "Figure 1. Raw F/I/V/t and fixed idx phases — arithmetic alignment audit, not a process-cycle label",
    fontsize=14,
    y=0.995,
)
fig1 = save_figure(fig, "figure1_raw_idx_phase_audit.png")


# Parent-union membership is reconstructed only from the frozen YSH-008 block table.
parent_mask_by_excel: dict[int, bool] = {}
for row in parent_blocks.itertuples(index=False):
    for start, end in [(int(row.a_start), int(row.a_end)), (int(row.b_start), int(row.b_end))]:
        for excel_row in range(start, end + 1):
            parent_mask_by_excel[excel_row] = True
core_rows = set(mapping["excel_row"].astype(int))


# Figure 2: all B rows, parent-repeat union, repeat298, fixed in-sample C0/C3, and F>3 bar.
b = raw.loc[raw["date"].isin(B_DATES)].copy()
b["date_order"] = pd.Categorical(b["date"], categories=B_DATES, ordered=True)
b = b.sort_values(["date_order", "excel_row"], kind="stable").reset_index(drop=True)
b["b_position"] = np.arange(1, len(b) + 1)
b["parent_repeat_union"] = b["excel_row"].map(parent_mask_by_excel).eq(True)
b["repeat298_core"] = b["excel_row"].isin(core_rows)
b = b.merge(scores[["excel_row", "assigned_component"]], on="excel_row", how="left", validate="one_to_one")
b["high_F_gt3"] = b["weld force(bar)"].gt(3.0)
assert len(b) == 5600
assert int(b["parent_repeat_union"].sum()) == 5584
assert int(b["repeat298_core"].sum()) == 1192

fig, axes = plt.subplots(
    7,
    1,
    figsize=(18, 14),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.26, 0.26, 0.26], "hspace": 0.10},
)
bx = b["b_position"].to_numpy()
parent_spans = contiguous_spans(b["parent_repeat_union"].to_numpy())
core_spans = contiguous_spans(b["repeat298_core"].to_numpy())
for ax, (col, label, unit, color) in zip(axes[:4], FEATURES):
    for start, end in parent_spans:
        ax.axvspan(bx[start] - 0.5, bx[end] + 0.5, color="#d9d9d9", alpha=0.38, zorder=0)
    for start, end in core_spans:
        ax.axvspan(bx[start] - 0.5, bx[end] + 0.5, color="#f5a623", alpha=0.25, zorder=0)
    ax.plot(bx, b[col], color=color, linewidth=0.65, zorder=1)
    ax.set_ylabel(f"{label} ({unit})")

strip_specs = [
    (b["assigned_component"].eq(0), "fixed GMM C0", "#4c78a8"),
    (b["assigned_component"].eq(3), "fixed GMM C3", "#b279a2"),
    (b["high_F_gt3"], "F > 3 bar", "#e45756"),
]
for ax, (mask, label, color) in zip(axes[4:], strip_specs):
    ax.scatter(bx[mask], np.zeros(int(mask.sum())), marker="|", s=18, color=color, linewidths=0.8)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center")

date_counts = b.groupby("date", observed=True, sort=False).size().reindex(B_DATES)
offset = 0
for date, count in date_counts.items():
    center = offset + count / 2
    axes[0].text(center, 1.03, f"{date}\nn={int(count)}", transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
    if offset:
        for ax in axes:
            ax.axvline(offset + 0.5, color="black", linewidth=0.8, alpha=0.45)
    offset += int(count)
axes[-1].set_xlabel("B-date rows concatenated within date by original Excel row")
axes[0].legend(
    handles=[
        Patch(facecolor="#d9d9d9", alpha=0.6, label="parent exact-repeat union"),
        Patch(facecolor="#f5a623", alpha=0.5, label="repeat298 core"),
    ],
    loc="upper right",
    frameon=False,
)
fig.suptitle(
    "Figure 2. B rows with frozen parent exact-repeat mapping (5,584/5,600) and repeat298 core (1,192)",
    fontsize=14,
    y=0.998,
)
fig2 = save_figure(fig, "figure2_b_parent_repeat_timeline.png")


# Figure 3: relative-position profile.  Raw4 and fixed-AE errors are identical by occurrence.
raw_cols = [x[0] for x in FEATURES]
raw_range = mapping.groupby("relative_position")[raw_cols].agg(lambda s: float(s.max() - s.min()))
assert float(raw_range.to_numpy().max()) == 0.0

def deterministic_mode(values: pd.Series) -> str:
    counts = values.value_counts()
    top = counts[counts.eq(counts.max())].index.astype(str).tolist()
    order = {"F": 0, "I": 1, "V": 2, "t": 3}
    return sorted(top, key=lambda value: order.get(value, 99))[0]


mapping_agg = mapping.groupby("relative_position", sort=True).agg(
    **{col: (col, "first") for col in raw_cols},
    nll_percentile_median=("global_nll_train_percentile", "median"),
    d2_percentile_median=("mahalanobis_d2_train_percentile", "median"),
    max_abs_conditional_z_median_check=("max_abs_conditional_z", "median"),
    gmm_dominant_variable=("gmm_dominant_variable", deterministic_mode),
).reset_index()
profile = relative.merge(mapping_agg, on="relative_position", how="left", validate="one_to_one")
profile.to_csv(OUT / "figure3_repeat298_profile.csv", index=False)

fig, axes = plt.subplots(
    8,
    1,
    figsize=(18, 18),
    sharex=True,
    gridspec_kw={"height_ratios": [0.8, 0.8, 0.8, 0.8, 1.25, 1, 1, 0.75], "hspace": 0.11},
)
rx = profile["relative_position"].to_numpy()
for ax in axes:
    ax.axvspan(0.5, 12.5, color="#f5a623", alpha=0.16)
    ax.axvspan(138.5, 143.5, color="#7e57c2", alpha=0.15)
for ax, (col, label, unit, color) in zip(axes[:4], FEATURES):
    ax.plot(rx, profile[col], color=color, linewidth=1)
    ax.set_ylabel(f"{label} ({unit})")

error_specs = [
    ("ae_total_error", "total", "black", 1.5),
    ("ae_error_F", "F", "#1f77b4", 0.9),
    ("ae_error_I", "I", "#ff7f0e", 0.9),
    ("ae_error_V", "V", "#2ca02c", 0.9),
    ("ae_error_t", "t", "#9467bd", 0.9),
]
for col, label, color, width in error_specs:
    axes[4].plot(rx, profile[col], label=label, color=color, linewidth=width)
axes[4].set_ylabel("AE squared error")
axes[4].legend(ncol=5, frameon=False, loc="upper right")

axes[5].plot(rx, profile["nll_percentile_median"], color="#e45756", label="NLL percentile")
axes[5].plot(rx, profile["d2_percentile_median"], color="#4c78a8", label="d² percentile")
axes[5].axhline(0.9, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
axes[5].scatter(rx[profile["gmm_nll_q90_support"].ge(3)], profile.loc[profile["gmm_nll_q90_support"].ge(3), "nll_percentile_median"], s=9, color="#e45756")
axes[5].scatter(rx[profile["gmm_d2_q90_support"].ge(3)], profile.loc[profile["gmm_d2_q90_support"].ge(3), "d2_percentile_median"], s=9, color="#4c78a8")
axes[5].set_ylim(-0.02, 1.02)
axes[5].set_ylabel("LODO train percentile")
axes[5].legend(frameon=False, loc="lower right")

dominant_colors = {"F": "#1f77b4", "I": "#ff7f0e", "V": "#2ca02c", "t": "#9467bd"}
axes[6].plot(rx, profile["max_abs_conditional_z_median"], color="#555555", linewidth=0.9)
for variable, color in dominant_colors.items():
    mask = profile["gmm_dominant_variable"].eq(variable)
    axes[6].scatter(rx[mask], profile.loc[mask, "max_abs_conditional_z_median"], s=13, color=color, label=variable)
axes[6].axhline(2.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
axes[6].set_ylabel("median max |conditional-z|")
axes[6].legend(title="modal dominant", ncol=4, frameon=False, loc="upper right")

axes[7].step(rx, profile["physical_global_candidate_support"] / 4.0, where="mid", color="#e45756")
axes[7].fill_between(rx, 0, profile["physical_global_candidate_support"] / 4.0, step="mid", color="#e45756", alpha=0.25)
axes[7].axhline(0.75, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
axes[7].set_ylim(-0.03, 1.03)
axes[7].set_ylabel("physical\nsupport / 4")
axes[7].set_xlabel("repeat298 relative_position")
axes[0].legend(
    handles=[
        Patch(facecolor="#f5a623", alpha=0.35, label="AE episode 1–12"),
        Patch(facecolor="#7e57c2", alpha=0.35, label="GMM local d² primary episode 139–143"),
    ],
    frameon=False,
    loc="upper right",
)
fig.suptitle(
    "Figure 3. repeat298 aligned profile — four identical Raw4 occurrences, fixed AE and LODO GMM diagnostics",
    fontsize=14,
    y=0.997,
)
fig3 = save_figure(fig, "figure3_repeat298_aligned_profile.png")


# Figure 4: group distributions reconstructed from frozen YSH-007 scores and YSH-008 mappings.
frame = physical[["excel_row", "regime", "global_nll", "mahalanobis_d2"]].copy()
frame["repeat298_core"] = frame["excel_row"].isin(core_rows)
frame["parent_repeat_union"] = frame["excel_row"].map(parent_mask_by_excel).eq(True)
frame["audit_group"] = np.select(
    [
        frame["regime"].eq("A"),
        frame["repeat298_core"],
        frame["regime"].eq("B") & frame["parent_repeat_union"] & ~frame["repeat298_core"],
        frame["regime"].eq("B") & ~frame["parent_repeat_union"],
    ],
    ["A", "B-repeat298", "B-parent-other", "outside-parent"],
    default="unassigned",
)
assert not frame["audit_group"].eq("unassigned").any()
group_order = ["A", "B-repeat298", "B-parent-other", "outside-parent"]
expected_counts = {"A": 6339, "B-repeat298": 1192, "B-parent-other": 4392, "outside-parent": 16}
actual_counts = frame["audit_group"].value_counts().to_dict()
assert actual_counts == expected_counts

group_summary = frame.groupby("audit_group", sort=False).agg(
    rows=("excel_row", "size"),
    global_nll_mean=("global_nll", "mean"),
    global_nll_median=("global_nll", "median"),
    global_nll_p90=("global_nll", lambda s: s.quantile(0.9)),
    d2_mean=("mahalanobis_d2", "mean"),
    d2_median=("mahalanobis_d2", "median"),
    d2_p90=("mahalanobis_d2", lambda s: s.quantile(0.9)),
).reindex(group_order)
group_summary.to_csv(OUT / "figure4_group_summary.csv")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
colors = ["#9e9e9e", "#f5a623", "#4c78a8", "#e45756"]
for ax, metric, title in [
    (axes[0], "global_nll", "Global NLL (LODO)"),
    (axes[1], "mahalanobis_d2", "Within-component d² (LODO)"),
]:
    data = [frame.loc[frame["audit_group"].eq(group), metric].to_numpy() for group in group_order]
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.65)
    parts["cmedians"].set_color("black")
    ax.boxplot(data, widths=0.12, showfliers=False, patch_artist=True, boxprops={"facecolor": "white", "alpha": 0.75})
    ax.set_xticks(range(1, 5), [f"{group}\nn={expected_counts[group]:,}" for group in group_order])
    ax.set_title(title)
    ax.set_ylabel(metric)
axes[1].text(
    0.98,
    0.98,
    "outside-parent n=16:\nunstable comparison base",
    transform=axes[1].transAxes,
    ha="right",
    va="top",
    color="#b22222",
    bbox={"facecolor": "white", "edgecolor": "#b22222", "alpha": 0.85},
)
fig.suptitle("Figure 4. Frozen GMM score distributions by repeat context (rows are not independent defects)", fontsize=14)
fig4 = save_figure(fig, "figure4_gmm_distributions_by_repeat_context.png")


# Figure 5: date-level signatures and date-level Result only.
daily["date"] = pd.to_datetime(daily["date"])
daily["indentation_rows_share"] = daily["indentation_rows"] / daily["rows"]
daily["insufficient_rows_share"] = daily["insufficient_rows"] / daily["rows"]
daily["crack_related_rows_share"] = daily["crack_related_rows"] / daily["rows"]
daily.to_csv(OUT / "figure5_daily_signatures_result.csv", index=False)

panel_specs = [
    (
        "indentation_rows_share",
        "type1",
        "파임 관련 signature",
        "I≥LODO q75, t≥q75, E_proxy≥q90 (F는 함께 검토)",
        "#4c78a8",
    ),
    (
        "insufficient_rows_share",
        "type2",
        "용접부족 관련 signature",
        "(I≤q25 & E_proxy≤q10) OR (F≥q75 & E_proxy≤q25)",
        "#f58518",
    ),
    (
        "crack_related_rows_share",
        "type3",
        "크랙 관련 signature",
        "E_proxy≥q90 & F≥q75 & (I≥q75 OR t≥q75)",
        "#54a24b",
    ),
]
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
positions = np.arange(len(daily))
for ax, (metric, result_col, title, definition, color) in zip(axes, panel_specs):
    ax.plot(positions, daily[metric] * 100, marker="o", color=color, linewidth=1.8, label="signature row share")
    ax.set_ylabel("signature share (%)", color=color)
    ax.tick_params(axis="y", labelcolor=color)
    twin = ax.twinx()
    values = daily[result_col]
    twin.bar(positions, values, width=0.45, color="#777777", alpha=0.38, label="daily Result count")
    twin.set_ylabel("Result count", color="#555555")
    twin.set_ylim(0, max(4, float(values.max(skipna=True)) + 1))
    for pos, value in zip(positions, values):
        if pd.isna(value):
            twin.text(pos, 0.12, "NA", ha="center", va="bottom", fontsize=8, color="#b22222")
    rho_row = association.loc[
        association["scope"].eq("all")
        & association["result_type"].eq(result_col)
        & association["metric"].eq(metric)
    ].iloc[0]
    ax.set_title(f"{title}: {definition}\n날짜 집계 Spearman ρ={rho_row.rho:.3f}, n={int(rho_row.n_dates)}")
    ax.grid(axis="x", alpha=0.12)
labels = [f"{date:%m-%d}\n{regime}" for date, regime in zip(daily["date"], daily["regime"])]
axes[-1].set_xticks(positions, labels)
axes[-1].set_xlabel("date and exploratory A/B label (bars are daily aggregate Result, not row labels)")
fig.suptitle("Figure 5. Physical proxy/signature prevalence beside daily Result records", fontsize=14, y=0.997)
fig5 = save_figure(fig, "figure5_physical_signatures_vs_daily_result.png")


figure_paths = [fig1, fig2, fig3, fig4, fig5]
image_shapes = {}
for path in figure_paths:
    image = plt.imread(path)
    image_shapes[path.name] = [int(value) for value in image.shape]
    assert path.stat().st_size > 50_000

primary_counts = {
    "ae": int(relative["ae_candidate_support"].ge(3).sum()),
    "nll": int(relative["gmm_nll_q90_support"].ge(3).sum()),
    "d2": int(relative["gmm_d2_q90_support"].ge(3).sum()),
    "conditional_z": int(relative["conditional_z_2_5_support"].ge(3).sum()),
    "physical": int(relative["physical_global_candidate_support"].ge(3).sum()),
}
assert primary_counts == {"ae": 12, "nll": 253, "d2": 17, "conditional_z": 10, "physical": 16}

verification = {
    "status": "passed",
    "new_model_training": False,
    "raw_rows": len(raw),
    "raw_sha256": sha256(RAW_PATH),
    "guidebook_welding_source_file_found": False,
    "figure1_sample_rows": len(sample),
    "B_rows": len(b),
    "B_parent_repeat_union_rows": int(b["parent_repeat_union"].sum()),
    "B_repeat298_core_rows": int(b["repeat298_core"].sum()),
    "outside_parent_rows": int((~b["parent_repeat_union"]).sum()),
    "repeat298_rows": len(mapping),
    "repeat298_positions": mapping["relative_position"].nunique(),
    "repeat298_raw4_max_occurrence_range": float(raw_range.to_numpy().max()),
    "primary_positive_positions_support_ge3of4": primary_counts,
    "figure4_group_counts": actual_counts,
    "figure_shapes": image_shapes,
    "source_hashes": """
        str(path.relative_to(ROOT)).replace("\", "/"): sha256(path)
        for path in [
            Y7 / "in_sample_scores.csv",
            Y7 / "physical_row_followup.csv",
            Y7 / "physical_daily_result.csv",
            Y7 / "physical_result_association.csv",
            Y8 / "repeat298_mapping.csv",
            Y8 / "ae_gmm_relative_profile.csv",
            Y8 / "repeat298_parent_blocks.csv",
            Y8 / "repeat298_gmm_decomposition.csv",
        ]
    """,
}
verification["source_hashes"] = {
    path.relative_to(ROOT).as_posix(): sha256(path)
    for path in [
        Y7 / "in_sample_scores.csv",
        Y7 / "physical_row_followup.csv",
        Y7 / "physical_daily_result.csv",
        Y7 / "physical_result_association.csv",
        Y8 / "repeat298_mapping.csv",
        Y8 / "ae_gmm_relative_profile.csv",
        Y8 / "repeat298_parent_blocks.csv",
        Y8 / "repeat298_gmm_decomposition.csv",
    ]
}
(OUT / "verification.json").write_text(json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps(verification, indent=2, ensure_ascii=False))
