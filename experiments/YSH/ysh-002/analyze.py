"""YSH-002: deterministic sequence audit; no labels, models or time reconstruction."""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
NAMES = ["force", "current", "voltage", "time"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def run_bounds(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def correlations(left, right):
    a = left - left.mean(axis=0)
    b = right - right.mean(axis=0)
    denominator = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    # Test constancy using actual range, not numerical roundoff in the mean.
    valid = (np.ptp(left, axis=0) > 0) & (np.ptp(right, axis=0) > 0)
    return np.divide(
        (a * b).sum(axis=0), denominator, out=np.full(4, np.nan), where=valid & (denominator > 0)
    )


class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size)

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return int(item)

    def join(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def labels(self):
        return np.array([self.find(i) for i in range(len(self.parent))])


def make_segments(raw, excluded):
    selected = raw.loc[~excluded].copy()
    boundary = selected.source_segment.ne(
        selected.source_segment.shift()
    ) | selected.excel_row.diff().ne(1)
    selected["segment"] = boundary.cumsum()
    return [part.copy() for _, part in selected.groupby("segment", sort=False)]


def lag_record(values, lag):
    n = len(values)
    corr = correlations(values[:-lag], values[lag:])
    delta = np.diff(values, axis=0)
    diff_corr = (
        correlations(delta[:-lag], delta[lag:]) if len(delta) - lag >= 100 else np.full(4, np.nan)
    )
    same = np.all(values[:-lag] == values[lag:], axis=1)
    runs = run_bounds(same)
    longest = max(runs, key=lambda item: item[1] - item[0], default=(0, 0))
    item = {
        "lag": lag,
        "pairs": n - lag,
        "diff_pairs": max(0, n - 1 - lag),
        "exact_count": int(same.sum()),
        "exact_rate": float(same.mean()),
        "longest_exact": int(longest[1] - longest[0]),
        "exact_run_position": int(longest[0] + 1),
        "three_cycles": n >= 3 * lag,
    }
    item.update({f"raw_{name}": corr[j] for j, name in enumerate(NAMES)})
    item.update({f"diff_{name}": diff_corr[j] for j, name in enumerate(NAMES)})
    return item


def scan(segments, condition, config):
    records, metadata = [], []
    for number, part in enumerate(segments, 1):
        identifier = f"{condition}_s{number:02d}"
        n = len(part)
        values = part[COLS].to_numpy(float)
        frequencies = part.groupby(COLS, dropna=False).size().to_numpy()
        chance = float(np.sum(frequencies * (frequencies - 1)) / (n * (n - 1))) if n > 1 else np.nan
        info = {
            "condition": condition,
            "segment": identifier,
            "date": part.date.iloc[0],
            "n": n,
            "excel_start": int(part.excel_row.iloc[0]),
            "excel_end": int(part.excel_row.iloc[-1]),
            "max_lag": max(0, min(config["max_lag"], n - config["min_pairs"])),
            "unordered_reference": chance,
        }
        metadata.append(info)
        for lag in range(1, info["max_lag"] + 1):
            records.append({**info, **lag_record(values, lag)})
    return records, metadata


def find_peaks(lags):
    records = []
    metrics = ["exact_rate"] + [f"{kind}_{name}" for kind in ["raw", "diff"] for name in NAMES]
    for (condition, segment), part in lags.groupby(["condition", "segment"], sort=False):
        for metric in metrics:
            values = part[metric].to_numpy()
            candidates = []
            for i, value in enumerate(values):
                if not np.isfinite(value) or not bool(part.iloc[i].three_cycles):
                    continue
                neighbors = np.r_[values[max(0, i - 5) : i], values[i + 1 : i + 6]]
                neighbors = neighbors[np.isfinite(neighbors)]
                if len(neighbors) < 2 or value < neighbors.max():
                    continue
                contrast = value - np.median(neighbors)
                if contrast > 1e-12:
                    candidates.append(
                        {
                            "condition": condition,
                            "segment": segment,
                            "metric": metric,
                            "lag": int(part.iloc[i].lag),
                            "value": value,
                            "contrast": contrast,
                            "pairs": int(part.iloc[i].pairs),
                        }
                    )
            records.extend(sorted(candidates, key=lambda r: (-r["contrast"], r["lag"]))[:5])
    return pd.DataFrame(records)


def graph_audit(raw, repeats, thresholds):
    summaries, memberships, masks = [], [], {}
    for threshold in thresholds:
        uf = UnionFind(len(raw))
        excluded = np.zeros(len(raw), bool)
        covered = np.zeros(len(raw), bool)
        eligible = repeats[repeats.length >= threshold]
        for record in eligible.itertuples():
            a, b, n = record.a_start - 2, record.b_start - 2, record.length
            excluded[b : b + n] = True
            covered[a : a + n] = True
            covered[b : b + n] = True
            for k in range(n):
                uf.join(a + k, b + k)
        masks[threshold] = excluded
        for policy in ["row_correspondence", "window20", "whole_date"]:
            graph = UnionFind(len(raw))
            graph.parent = uf.labels().copy()
            if policy == "window20":
                # Overlapping 20-record inputs cover every adjacent pair of a >=20 segment.
                for _, part in raw.groupby("source_segment"):
                    if len(part) >= 20:
                        ids = part.index.to_numpy()
                        for left, right in zip(ids[:-1], ids[1:]):
                            graph.join(left, right)
            if policy == "whole_date":
                for _, part in raw.groupby("date"):
                    ids = part.index.to_numpy()
                    for left, right in zip(ids[:-1], ids[1:]):
                        graph.join(left, right)
            labels = graph.labels()
            unique, counts = np.unique(labels, return_counts=True)
            largest = int(counts.max())
            summaries.append(
                {
                    "threshold": threshold,
                    "policy": policy,
                    "components": len(counts),
                    "singletons": int((counts == 1).sum()),
                    "largest_rows": largest,
                    "largest_fraction": largest / len(raw),
                    "excluded_rows": int(excluded.sum()),
                    "covered_rows": int(covered.sum()),
                    "repeat_pairs": len(eligible),
                }
            )
            for label, count in zip(unique, counts):
                rows = raw.loc[labels == label]
                memberships.append(
                    {
                        "threshold": threshold,
                        "policy": policy,
                        "component": int(label),
                        "rows": int(count),
                        "dates": rows.date.nunique(),
                        "date_list": ";".join(sorted(rows.date.unique())),
                        "excel_first": int(rows.excel_row.min()),
                    }
                )
    return pd.DataFrame(summaries), pd.DataFrame(memberships), masks


def change_points(segments, condition, config):
    candidates, states, summaries = [], [], []
    for number, part in enumerate(segments, 1):
        name = f"{condition}_s{number:02d}"
        x = part[COLS].reset_index(drop=True)
        n = len(x)
        scale = (x.quantile(0.75) - x.quantile(0.25)).to_numpy() / 1.349
        for window in config["change_windows"]:
            before = x.rolling(window).median().shift(1)
            after = x.iloc[::-1].rolling(window).median().iloc[::-1]
            delta = (after - before).to_numpy()
            score = np.divide(
                np.abs(delta), scale, out=np.full_like(delta, np.nan), where=scale > 0
            )
            for threshold in config["change_thresholds"]:
                crossed = score >= threshold
                single = np.flatnonzero(crossed[:, :3].any(axis=1))
                joint = np.flatnonzero(crossed[:, :3].sum(axis=1) >= 2)
                # Cluster nearby candidates, then retain strongest; tie -> earliest.
                groups = (
                    np.split(joint, np.flatnonzero(np.diff(joint) > window) + 1)
                    if len(joint)
                    else []
                )
                kept = [
                    int(sorted(group, key=lambda t: (-np.nanmax(score[t]), t))[0])
                    for group in groups
                ]
                for position in single:
                    candidates.append(
                        {
                            "condition": condition,
                            "segment": name,
                            "window": window,
                            "threshold": threshold,
                            "position": int(position),
                            "excel_row_after": int(part.excel_row.iloc[position]),
                            "joint": bool(position in joint),
                            "retained": bool(position in kept),
                            **{f"delta_{v}": delta[position, j] for j, v in enumerate(NAMES)},
                        }
                    )
                cuts = [0] + kept + [n]
                for i, (left, right) in enumerate(zip(cuts[:-1], cuts[1:])):
                    states.append(
                        {
                            "condition": condition,
                            "segment": name,
                            "window": window,
                            "threshold": threshold,
                            "start": left,
                            "end": right,
                            "length": right - left,
                            "censored": i == 0 or i == len(cuts) - 2,
                        }
                    )
                summaries.append(
                    {
                        "condition": condition,
                        "segment": name,
                        "window": window,
                        "threshold": threshold,
                        "n": n,
                        "eligible_positions": max(0, n - 2 * window + 1),
                        "zero_iqr_variables": int((scale == 0).sum()),
                        "single_candidates": len(single),
                        "joint_candidates": len(joint),
                        "retained": len(kept),
                    }
                )
    return pd.DataFrame(candidates), pd.DataFrame(states), pd.DataFrame(summaries)


def variable_audit(raw, conditions):
    results = []
    for condition, segments in conditions.items():
        frame = pd.concat(segments)
        for column, name in zip(COLS, NAMES):
            values = frame[column]
            unique = np.sort(values.unique())
            gaps = np.diff(unique)
            representative_gaps = np.round(gaps[gaps > 1e-10], 9)
            gap_counts = pd.Series(representative_gaps).value_counts()
            longest = 0
            for part in segments:
                v = part[column].to_numpy()
                cuts = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1, len(v)]
                longest = max(longest, int(np.diff(cuts).max()))
            top = values.value_counts()
            results.append(
                {
                    "condition": condition,
                    "variable": name,
                    "n": len(values),
                    "unique": len(unique),
                    "minimum": values.min(),
                    "median": values.median(),
                    "maximum": values.max(),
                    "mean": values.mean(),
                    "std": values.std(),
                    "min_positive_gap": gaps.min() if len(gaps) else np.nan,
                    "typical_gap": gap_counts.index[0] if len(gap_counts) else np.nan,
                    "longest_run": longest,
                    "top_value": top.index[0],
                    "top_count": int(top.iloc[0]),
                    "top_share": top.iloc[0] / len(values),
                    "noninteger_count": int((np.abs(values - values.round()) > 1e-9).sum()),
                }
            )
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent
    out = root / "outputs"
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    start = time.time()

    assert sha(config["raw"]) == config["expected_sha256"], "Input version differs from plan"
    raw = pd.read_excel(config["raw"], sheet_name="Raw data")
    assert raw.shape == (11939, 10) and not raw.isna().any().any()
    raw["excel_row"] = np.arange(len(raw)) + 2
    raw["date"] = pd.to_datetime(raw["working time"]).dt.strftime("%Y-%m-%d")
    raw["date_position"] = raw.groupby("date", sort=False).cumcount() + 1
    conflict = raw.duplicated(["date", "idx"], keep=False)
    boundary = (
        raw.date.ne(raw.date.shift())
        | raw.idx.diff().ne(1)
        | conflict
        | conflict.shift(fill_value=False)
    )
    raw["source_segment"] = boundary.cumsum()
    ordered = raw.sort_values(["date", "idx", "excel_row"], kind="stable")
    groups_a = {tuple(g.excel_row) for _, g in raw.groupby("source_segment")}
    groups_b = {tuple(g.excel_row) for _, g in ordered.groupby("source_segment")}
    assert groups_a == groups_b

    repeats = pd.read_csv(config["repeat_table"])
    values = raw[COLS].to_numpy()
    for record in repeats.itertuples():
        left = values[record.a_start - 2 : record.a_end - 1]
        right = values[record.b_start - 2 : record.b_end - 1]
        assert np.array_equal(left, right) and len(left) == record.length
    assert repeats.length.max() == 1639
    known = [values[s - 2 : s - 2 + 298] for s in [1879, 4693, 7803, 10879]]
    assert all(np.array_equal(known[0], item) for item in known[1:])

    for side in ["a", "b"]:
        for end in ["start", "end"]:
            row_indices = repeats[f"{side}_{end}"] - 2
            for col in ["date_position", "idx", "source_segment"]:
                repeats[f"{side}_{end}_{col}"] = raw.loc[row_indices, col].to_numpy()
    repeats.to_csv(tables / "repeat_alignment.csv", index=False)

    graph, components, masks = graph_audit(raw, repeats, config["repeat_thresholds"])
    graph.to_csv(tables / "group_summary.csv", index=False)
    components.to_csv(tables / "component_sizes.csv", index=False)
    conditions = {"all": make_segments(raw, np.zeros(len(raw), bool))}
    conditions.update(
        {f"controlled{threshold}": make_segments(raw, mask) for threshold, mask in masks.items()}
    )

    lag_rows, metadata = [], []
    for condition, segments in conditions.items():
        rows, meta = scan(segments, condition, config)
        lag_rows.extend(rows)
        metadata.extend(meta)
        print(f"Scanned {condition}: {len(rows)} lag positions", flush=True)
    lags = pd.DataFrame(lag_rows)
    lags.to_csv(tables / "lag_scan.csv", index=False)
    pd.DataFrame(metadata).to_csv(tables / "segments.csv", index=False)
    peaks = find_peaks(lags)
    peaks.to_csv(tables / "peak_candidates.csv", index=False)

    # Same candidate lag at full/first/second halves and its multiples.
    segments_lookup = {
        f"{condition}_s{i:02d}": g
        for condition, segs in conditions.items()
        for i, g in enumerate(segs, 1)
    }
    validation = []
    for peak in peaks.itertuples():
        part = segments_lookup[peak.segment]
        x = part[COLS].to_numpy()
        partitions = {"full": x, "first_half": x[: len(x) // 2], "second_half": x[len(x) // 2 :]}
        for partition, data in partitions.items():
            for multiple in [1, 2, 3]:
                lag = peak.lag * multiple
                record = {
                    "condition": peak.condition,
                    "segment": peak.segment,
                    "metric": peak.metric,
                    "candidate_lag": peak.lag,
                    "multiple": multiple,
                    "partition": partition,
                    "n": len(data),
                    "eligible": len(data) - int(peak.metric.startswith("diff_")) - lag >= 100,
                }
                if record["eligible"]:
                    stats = lag_record(data, lag)
                    record.update(
                        {
                            "value": stats[peak.metric],
                            "pairs": (
                                stats["diff_pairs"]
                                if peak.metric.startswith("diff_")
                                else stats["pairs"]
                            ),
                            "three_cycles": len(data) >= 3 * lag,
                        }
                    )
                validation.append(record)
    pd.DataFrame(validation).to_csv(tables / "peak_rechecks.csv", index=False)

    # Same-date length controls: first/middle/last original window for each retained segment length.
    selected_lags = sorted(
        set([1, 20, 50, 100] + peaks.loc[peaks.condition == "all", "lag"].tolist())
    )
    matched = []
    for number, retained in enumerate(conditions["controlled100"], 1):
        n = len(retained)
        if n < 101:
            continue
        for original in conditions["all"]:
            if original.date.iloc[0] != retained.date.iloc[0] or len(original) < n:
                continue
            for begin in sorted(set([0, (len(original) - n) // 2, len(original) - n])):
                window = original.iloc[begin : begin + n]
                for lag in selected_lags:
                    if n - lag >= 100:
                        matched.append(
                            {
                                "retained_segment": f"controlled100_s{number:02d}",
                                "date": window.date.iloc[0],
                                "n": n,
                                "excel_start": int(window.excel_row.iloc[0]),
                                "same_rows": bool(
                                    np.array_equal(window.excel_row, retained.excel_row)
                                ),
                                **lag_record(window[COLS].to_numpy(), lag),
                            }
                        )
    pd.DataFrame(matched).to_csv(tables / "length_controls.csv", index=False)

    cp, states, change_summary = [], [], []
    for condition in ["all", "controlled100"]:
        a, b, c = change_points(conditions[condition], condition, config)
        cp.append(a)
        states.append(b)
        change_summary.append(c)
    pd.concat(cp).to_csv(tables / "change_candidates.csv", index=False)
    pd.concat(states).to_csv(tables / "state_durations.csv", index=False)
    pd.concat(change_summary).to_csv(tables / "change_summary.csv", index=False)
    variable_audit(raw, conditions).to_csv(tables / "variable_summary.csv", index=False)

    summary = {
        "input_sha256": sha(config["raw"]),
        "source_rows": len(raw),
        "source_segments": raw.source_segment.nunique(),
        "source_dates": raw.date.nunique(),
        "verified_repeat_pairs": len(repeats),
        "AB_same_internal_segments": True,
        "conditions": {
            name: {
                "rows": sum(len(g) for g in segs),
                "segments": len(segs),
                "dates": len(set(g.date.iloc[0] for g in segs)),
            }
            for name, segs in conditions.items()
        },
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "seconds": time.time() - start,
        "code_sha256": sha(__file__),
        "config_sha256": sha(config_path),
    }
    assert summary["input_sha256"] == config["expected_sha256"]
    save_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
