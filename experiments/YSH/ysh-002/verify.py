"""Independent arithmetic checks and tabular phase/replication diagnostics."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
TABLES = ROOT / "outputs" / "tables"
COLS = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
NAMES = ["force", "current", "voltage", "time"]


def main():
    raw = pd.read_excel(CONFIG["raw"], sheet_name="Raw data")
    x = raw[COLS].to_numpy(float)
    lag_table = pd.read_csv(TABLES / "lag_scan.csv")
    segments = pd.read_csv(TABLES / "segments.csv")
    checks = 0

    # Independent np.corrcoef and scalar equality/run counting at low, middle, high lags.
    for segment, group in lag_table.groupby("segment"):
        meta = segments.loc[segments.segment == segment].iloc[0]
        values = x[int(meta.excel_start) - 2 : int(meta.excel_end) - 1]
        assert len(values) == meta.n
        assert np.array_equal(group.lag.to_numpy(), np.arange(1, meta.max_lag + 1))
        for index in sorted(set([0, len(group) // 2, len(group) - 1])):
            row = group.iloc[index]
            lag = int(row.lag)
            left, right = values[:-lag], values[lag:]
            equal = [bool(all(a == b)) for a, b in zip(left, right)]
            assert sum(equal) == row.exact_count
            assert len(equal) == row.pairs >= 100
            longest = current = 0
            for match in equal:
                current = current + 1 if match else 0
                longest = max(longest, current)
            assert longest == row.longest_exact
            for j, name in enumerate(NAMES):
                if np.ptp(left[:, j]) and np.ptp(right[:, j]):
                    expected = np.corrcoef(left[:, j], right[:, j])[0, 1]
                    assert np.isclose(expected, row[f"raw_{name}"], atol=1e-10)
                else:
                    assert pd.isna(row[f"raw_{name}"])
            checks += 1

    # Verify every retained change's raw median shift independently.
    changes = pd.read_csv(TABLES / "change_candidates.csv")
    retained = changes[changes.retained]
    for row in retained.itertuples():
        meta = segments.loc[segments.segment == row.segment].iloc[0]
        values = x[int(meta.excel_start) - 2 : int(meta.excel_end) - 1]
        t, w = row.position, row.window
        expected = np.median(values[t : t + w], axis=0) - np.median(values[t - w : t], axis=0)
        assert np.allclose(expected, [getattr(row, f"delta_{name}") for name in NAMES])
        assert t >= w and t + w <= len(values)

    repeats = pd.read_csv(TABLES / "repeat_alignment.csv")
    assert len(repeats) == 245
    date_edges = []
    for threshold in [50, 100, 200]:
        neighbors = {
            str(date.date()): set() for date in pd.to_datetime(raw["working time"]).unique()
        }
        eligible = repeats[repeats.length >= threshold]
        for row in eligible.itertuples():
            left = x[row.a_start - 2 : row.a_end - 1]
            right = x[row.b_start - 2 : row.b_end - 1]
            assert np.array_equal(left, right)
            neighbors[row.a_date].add(row.b_date)
            neighbors[row.b_date].add(row.a_date)
        remaining = set(neighbors)
        groups = []
        while remaining:
            stack = [min(remaining)]
            visited = set()
            while stack:
                date = stack.pop()
                if date not in visited:
                    visited.add(date)
                    stack.extend(neighbors[date] - visited)
            groups.append(visited)
            remaining -= visited
        assert len(groups) == 1 and len(groups[0]) == 9
        cross = eligible[eligible.a_date != eligible.b_date]
        for (a, b), group in cross.groupby(["a_date", "b_date"]):
            date_edges.append(
                {
                    "threshold": threshold,
                    "date_a": a,
                    "date_b": b,
                    "repeat_pairs": len(group),
                    "max_length": int(group.length.max()),
                }
            )
    pd.DataFrame(date_edges).to_csv(TABLES / "date_connections.csv", index=False)

    # Use predeclared peak candidates for profiles; no pictures and no significance claim.
    peaks = pd.read_csv(TABLES / "peak_candidates.csv")
    profiles, folds, family_halves = [], [], []
    for (segment, lag), group in peaks.groupby(["segment", "lag"]):
        if lag <= 1:
            continue
        meta = segments.loc[segments.segment == segment].iloc[0]
        values = x[int(meta.excel_start) - 2 : int(meta.excel_end) - 1]
        cycles = len(values) // lag
        if cycles < 3:
            continue
        tensor = values[: cycles * lag].reshape(cycles, lag, 4)
        for name in NAMES:
            if (
                not group.metric.isin([f"raw_{name}", f"diff_{name}"]).any()
                and not group.metric.eq("exact_rate").any()
            ):
                continue
            j = NAMES.index(name)
            means = tensor[:, :, j].mean(axis=0)
            between = np.var(means)
            total = np.var(tensor[:, :, j])
            folds.append(
                {
                    "condition": meta.condition,
                    "segment": segment,
                    "lag": lag,
                    "variable": name,
                    "cycles": cycles,
                    "phase_mean_variance_fraction": between / total if total > 0 else np.nan,
                }
            )
            for phase in range(lag):
                profiles.append(
                    {
                        "condition": meta.condition,
                        "segment": segment,
                        "lag": lag,
                        "variable": name,
                        "phase": phase,
                        "cycles": cycles,
                        "mean": means[phase],
                        "std": np.std(tensor[:, phase, j], ddof=1),
                    }
                )
    pd.DataFrame(folds).to_csv(TABLES / "phase_summary.csv", index=False)
    pd.DataFrame(profiles).to_csv(TABLES / "phase_profiles.csv", index=False)

    # Report exact-pair families by the explicitly shared interval, avoiding whole-family identity.
    for row in repeats.nlargest(10, "length").itertuples():
        values = x[row.a_start - 2 : row.a_end - 1]
        for lag in [1, 16, 18, 20, 27, 32, 36, 48, 64, 100]:
            if len(values) - lag < 100:
                continue
            for j, name in enumerate(NAMES):
                a, b = values[:-lag, j], values[lag:, j]
                corr = np.corrcoef(a, b)[0, 1] if np.ptp(a) and np.ptp(b) else np.nan
                family_halves.append(
                    {
                        "representative_excel": row.a_start,
                        "other_excel": row.b_start,
                        "length": row.length,
                        "lag": lag,
                        "variable": name,
                        "correlation": corr,
                        "note": "one explicit pair, overlaps other representatives; not independent family",
                    }
                )
    pd.DataFrame(family_halves).to_csv(TABLES / "shared_interval_summary.csv", index=False)
    digest = hashlib.sha256(Path(CONFIG["raw"]).read_bytes()).hexdigest()
    assert digest == CONFIG["expected_sha256"]
    result = {
        "lag_spot_checks": checks,
        "retained_change_checks": len(retained),
        "exact_pairs_verified": len(repeats),
        "date_graph_bfs_thresholds": [50, 100, 200],
        "all_date_graphs_one_component": True,
        "all_expected_lags_present": True,
        "input_unchanged": True,
        "phase_summary_rows": len(folds),
    }
    (ROOT / "outputs" / "verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
