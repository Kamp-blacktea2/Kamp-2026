"""Independent checks and additional boundary/lag sensitivity summaries."""

from pathlib import Path
import json, itertools, hashlib
import numpy as np
import pandas as pd

p = Path(__file__).resolve().parent / "outputs"
raw = pd.read_csv(p / "raw_audit.csv")
d = json.loads((p / "chart_data.json").read_text(encoding="utf-8"))
cols = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
names = ["force", "current", "voltage", "time"]
checks = {}
count = 0
offsetmetrics = []
boundaries = 0
for tag, segments in d.items():
    for s in segments:
        z = np.array(s["values"])
        n = len(z)
        rr = np.array(s["rows"])
        idx = np.array(s["idx"])
        assert len(set(raw.set_index("excel_row").loc[rr, "date"])) == 1
        assert np.all(np.diff(idx) == 1) or n == 1
        if tag.endswith("dedup"):
            assert not raw.set_index("excel_row").loc[rr, "repeat_excluded"].any()
        boundaries += 1
        for w in [20, 50, 100]:
            stat = s["rolling"][str(w)]
            assert all(v[0] is None for v in stat["median"][: min(w, n)])
            for i in sorted(set([w, n // 2, n - 1])):
                if i < w or i >= n:
                    continue
                hist = z[i - w : i]
                actual = np.array(stat["median"][i])
                assert np.allclose(actual, np.median(hist, axis=0))
                assert np.allclose(stat["std"][i], np.std(hist, axis=0, ddof=1))
                assert np.allclose(stat["q1"][i], np.quantile(hist, 0.25, axis=0))
                assert np.allclose(stat["q3"][i], np.quantile(hist, 0.75, axis=0))
                count += 1
        for off in [0, 10]:
            for start in range(off, n - 99, 100):
                zz = z[start : start + 100].reshape(5, 20, 4)
                for l, r in itertools.combinations(range(5), 2):
                    for j, name in enumerate(names):
                        x, y = zz[l, :, j], zz[r, :, j]
                        c = (
                            float(np.corrcoef(x, y)[0, 1])
                            if np.ptp(x) > 0 and np.ptp(y) > 0
                            else None
                        )
                        offsetmetrics.append(
                            {
                                "condition": tag,
                                "offset": off,
                                "segment": s["id"],
                                "start": start + 1,
                                "variable": name,
                                "left": l,
                                "right": r,
                                "correlation": c,
                                "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
                                "centered_rmse": float(
                                    np.sqrt(np.mean(((x - x.mean()) - (y - y.mean())) ** 2))
                                ),
                            }
                        )
b = pd.read_csv(p / "block_index.csv")
for tag, segs in d.items():
    for w in [20, 50, 100]:
        assert b.loc[
            (b.condition == tag) & (b.length_requested == w), "actual_length"
        ].sum() == sum(len(s["rows"]) for s in segs)
reps = pd.read_csv(p / "exact_repeated_blocks.csv")
vals = raw[cols].to_numpy()
for r in reps.itertuples():
    assert (vals[r.a_start - 2 : r.a_end - 1] == vals[r.b_start - 2 : r.b_end - 1]).all()

# Evidence for unchanged row-keyed rolling statistics between A and B.
lookup = {
    tag: {r: s["rolling"]["100"]["median"][i] for s in segs for i, r in enumerate(s["rows"])}
    for tag, segs in d.items()
}
assert lookup["A_all"] == lookup["B_all"]
assert lookup["A_dedup"] == lookup["B_dedup"]
offset = pd.DataFrame(offsetmetrics)
offset.to_csv(p / "boundary_offset_pair_metrics.csv", index=False, encoding="utf-8-sig")
offset.groupby(["condition", "offset", "variable"]).agg(
    pairs=("rmse", "size"),
    median_corr=("correlation", "median"),
    median_rmse=("rmse", "median"),
    median_centered_rmse=("centered_rmse", "median"),
).to_csv(p / "boundary_offset_summary.csv", encoding="utf-8-sig")
change = pd.read_csv(p / "change_candidates.csv")
change.groupby(["condition", "variable"]).agg(events=("excel_row", "size")).to_csv(
    p / "change_summary.csv", encoding="utf-8-sig"
)
raw.assign(condition="all").groupby("condition")[cols].agg(["mean", "median", "std"]).to_csv(
    p / "all_distribution.csv"
)
raw.loc[~raw.repeat_excluded, cols].agg(["mean", "median", "std"]).to_csv(
    p / "dedup_distribution.csv"
)
for path, h in json.loads((p / "summary.json").read_text(encoding="utf-8"))["input_hashes"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == h
checks = {
    "independent_window_checks": count,
    "segment_boundary_checks": boundaries,
    "all_block_coverage": True,
    "all_exact_repeat_pairs_rechecked": len(reps),
    "A_B_row_keyed_w100_equal": True,
    "input_hashes_still_unchanged": True,
}
(p / "verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
print(json.dumps(checks, indent=2))
print(pd.read_csv(p / "boundary_offset_summary.csv").to_string(index=False))
print(pd.read_csv(p / "change_summary.csv").to_string(index=False))
