"""Create offline Plotly explorer plus reproducible Matplotlib figures (CPU)."""

from pathlib import Path
import argparse, json, hashlib, sys, itertools
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotly.offline import get_plotlyjs

ap = argparse.ArgumentParser()
ap.add_argument("--output", required=True)
a = ap.parse_args()
out = Path(a.output)
data = json.loads((out / "chart_data.json").read_text(encoding="utf-8"))
raw = pd.read_csv(out / "raw_audit.csv")
pairs = pd.read_csv(out / "nested_pair_metrics.csv")
cols = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
names = ["force", "current", "voltage", "time"]
labels = ["Force (bar)", "Current (kA)", "Voltage (V)", "Energization (ms)"]
colors = ["#247ba0", "#dd6e42", "#3e885b", "#775da6"]
figdir = out / "figures"
figdir.mkdir(exist_ok=True)
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110}
)


def save(fig, name):
    fig.savefig(figdir / (name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
for j, ax in enumerate(axs):
    ax.plot(raw.excel_row, raw[cols[j]], lw=0.6, color=colors[j])
    ax.set_ylabel(labels[j])
    ax.grid(alpha=0.2)
    for _, g in raw.groupby("date", sort=False):
        ax.axvline(g.excel_row.iloc[0], color="gray", alpha=0.25)
axs[-1].set_xlabel("Original Excel row (not elapsed time)")
fig.suptitle("All 11,939 records: raw values, unchanged")
fig.tight_layout()
save(fig, "01_overview")
reps = pd.read_csv(out / "exact_repeated_blocks.csv").sort_values("length", ascending=False)
r = reps.iloc[0]
fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
for j, ax in enumerate(axs):
    for start, end, date, style in [
        (r.a_start, r.a_end, r.a_date, "-"),
        (r.b_start, r.b_end, r.b_date, "--"),
    ]:
        v = raw.loc[raw.excel_row.between(start, end), cols[j]].to_numpy()
        ax.plot(np.arange(1, len(v) + 1), v, style, lw=1, label=f"{date}, rows {start}-{end}")
    ax.set_ylabel(labels[j])
    ax.grid(alpha=0.2)
axs[0].legend(fontsize=9)
axs[-1].set_xlabel("Record within matched block")
fig.suptitle(f"Exact equality in all four variables: {r.length} records across dates")
fig.tight_layout()
save(fig, "02_long_repeat")

# Whole-population comparison; never cherry-pick one block as proof.
fig, axs = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
for j, ax in enumerate(axs):
    for condition, c in [("A_all", "#247ba0"), ("A_dedup", "#dd6e42")]:
        v = pairs.loc[
            (pairs.condition == condition) & (pairs.variable == names[j]), "correlation"
        ].dropna()
        ax.hist(
            v,
            bins=np.linspace(-1, 1, 26),
            density=True,
            histtype="step",
            lw=1.6,
            label=condition,
            color=c,
        )
    ax.set_title(labels[j])
    ax.set_xlabel("20-record pair correlation")
    ax.axvline(0, color="gray", lw=0.6)
axs[0].set_ylabel("Density")
axs[0].legend()
fig.suptitle("All complete 100-record blocks: ten pairs of internal 20-record windows")
fig.tight_layout()
save(fig, "03_correlations")

# Illustrate earliest full baseline block, fixed choice independent of its correlations.
first = next(s for s in data["A_all"] if len(s["values"]) >= 100)
z = np.array(first["values"])
chunk = z[:100].reshape(5, 20, 4)
fig, axs = plt.subplots(4, 3, figsize=(16, 11))
for j in range(4):
    axs[j, 0].plot(np.arange(1, 101), z[:100, j], color=colors[j])
    axs[j, 0].set_ylabel(labels[j])
    for k in range(1, 5):
        axs[j, 0].axvline(k * 20 + 0.5, color="gray", lw=0.5)
    for k in range(5):
        axs[j, 1].plot(np.arange(1, 21), chunk[k, :, j], label=f"Block {k+1}", lw=1)
        axs[j, 2].plot(np.arange(1, 21), chunk[k, :, j] - chunk[k, :, j].mean(), lw=1)
    for ax in axs[j]:
        ax.grid(alpha=0.2)
for ax, title in zip(
    axs[0], ["100 raw records", "Five 20-record blocks, raw", "Five blocks, mean removed"]
):
    ax.set_title(title)
axs[0, 1].legend(fontsize=8, ncol=2)
fig.suptitle(
    f'First complete baseline block: {first["date"]}, Excel {first["rows"][0]}-{first["rows"][99]}'
)
fig.tight_layout()
save(fig, "04_nested_example")
for w in [20, 50, 100]:
    fig, axs = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    n = min(500, len(z))
    stats = first["rolling"][str(w)]
    for j, ax in enumerate(axs):
        med = np.array(stats["median"], dtype=float)[:n, j]
        q1 = np.array(stats["q1"], dtype=float)[:n, j]
        q3 = np.array(stats["q3"], dtype=float)[:n, j]
        xx = np.arange(1, n + 1)
        ax.plot(xx, z[:n, j], lw=0.8, color=colors[j], label="Raw")
        ax.plot(xx, med, lw=1, color="#222", label="Past median")
        ax.fill_between(xx, q1, q3, alpha=0.22, color=colors[j], label="Past Q1-Q3")
        ax.set_ylabel(labels[j])
    axs[0].legend(ncol=3)
    axs[-1].set_xlabel("Record in segment")
    fig.suptitle(f"Past-only rolling window: {w} records; first segment, first {n} records")
    fig.tight_layout()
    save(fig, f"05_rolling_{w}")

# Explicit expansion selection: all complete 100-record blocks are inspected in the

# explorer. Aggregate evidence did not establish a common 20-record pattern, so

# expand coverage to all available aligned 200/500 blocks, not the best-looking ones.
expanded = []
autocorr = []
for tag, segments in data.items():
    for s in segments:
        n = len(s["values"])
        z0 = np.array(s["values"])
        for w in [200, 500]:
            if n < 100:
                expanded.append(
                    {
                        "condition": tag,
                        "segment": s["id"],
                        "length": w,
                        "start": None,
                        "actual_length": n,
                        "status": "skip",
                        "reason": "segment shorter than 100; no complete parent block",
                    }
                )
                continue
            for start in range(0, n, w):
                nn = min(w, n - start)
                expanded.append(
                    {
                        "condition": tag,
                        "segment": s["id"],
                        "length": w,
                        "start": start + 1,
                        "actual_length": nn,
                        "status": "complete" if nn == w else "partial",
                        "reason": "coverage expansion after no general 20-record recurrence in full-block comparison; boundary not crossed",
                    }
                )
        if n >= 200:
            for lag in [1, 10, 20, 50, 100, 200, 500]:
                if n - lag < 100:
                    continue
                for j in range(4):
                    v, u = z0[:-lag, j], z0[lag:, j]
                    c = float(np.corrcoef(v, u)[0, 1]) if np.ptp(v) > 0 and np.ptp(u) > 0 else None
                    autocorr.append(
                        {
                            "condition": tag,
                            "segment": s["id"],
                            "variable": names[j],
                            "lag_records": lag,
                            "pairs": len(v),
                            "correlation": c,
                        }
                    )
pd.DataFrame(expanded).to_csv(out / "expansion_index.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(autocorr).to_csv(out / "autocorrelation.csv", index=False, encoding="utf-8-sig")
for w in [200, 500]:
    z = np.array(first["values"])[:w]
    fig, axs = plt.subplots(4, 2, figsize=(14, 10))
    for j in range(4):
        axs[j, 0].plot(np.arange(1, len(z) + 1), z[:, j], lw=0.8, color=colors[j])
        axs[j, 0].set_ylabel(labels[j])
        for k in range(20, len(z), 20):
            axs[j, 0].axvline(k + 0.5, color="gray", alpha=0.3, lw=0.5)
        for start in range(0, len(z) - 19, 20):
            v = z[start : start + 20, j]
            axs[j, 1].plot(np.arange(1, 21), v - v.mean(), alpha=0.6, lw=0.8)
    axs[0, 0].set_title(f"{w} raw records, marks every 20")
    axs[0, 1].set_title("Complete 20-record subblocks, mean removed")
    fig.suptitle(f'Expansion example: {first["date"]}, first {w} records')
    fig.tight_layout()
    save(fig, f"06_expanded_{w}")

# Validate the four 298-record blocks explicitly, separate from maximal matches.
starts = [1879, 4693, 7803, 10879]
blocks = [raw.loc[raw.excel_row.between(s, s + 297), cols].to_numpy() for s in starts]
assert all(np.array_equal(blocks[0], b) for b in blocks[1:])
verification = {
    "known_298_blocks_identical": True,
    "known_298_excel_starts": starts,
    "matplotlib": matplotlib.__version__,
    "python": sys.version,
    "expansion_policy": "all aligned 200/500 blocks in segments with >=100 records; short tails labelled partial",
    "figure_count": len(list(figdir.glob("*.png"))),
}
(out / "visualization_checks.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
(out / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")


# Compact numbers only for plotting. CSV retains full computed precision.
def compact(v):
    if isinstance(v, float):
        return round(v, 7)
    if isinstance(v, list):
        return [compact(x) for x in v]
    if isinstance(v, dict):
        return {k: compact(x) for k, x in v.items()}
    return v


template = Path(__file__).with_name("explorer_template.html").read_text(encoding="utf-8")
(out / "graphs.html").write_text(
    template.replace(
        "__DATA__", json.dumps(compact(data), ensure_ascii=False, separators=(",", ":"))
    ),
    encoding="utf-8",
)
print(json.dumps(verification, indent=2))
print("Offline graph explorer created:", out / "graphs.html")
