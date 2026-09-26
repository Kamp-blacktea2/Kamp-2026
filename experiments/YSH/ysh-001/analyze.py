"""YSH-001 deterministic, read-only workbook audit. No model training."""

from pathlib import Path
import argparse, hashlib, json, platform, sys, itertools, time
import numpy as np
import pandas as pd

COLS = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
NAMES = ["force", "current", "voltage", "time"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    p.write_text(
        json.dumps(x, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )


def records(df):
    return json.loads(df.to_json(orient="records", date_format="iso"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--scaled", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    hashes = {p: sha(p) for p in [a.raw, a.scaled]}
    sheets = pd.read_excel(a.raw, sheet_name=None)
    scaled = pd.read_excel(a.scaled)
    raw = sheets["Raw data"].copy()
    raw["excel_row"] = np.arange(len(raw)) + 2
    raw["date"] = pd.to_datetime(raw["working time"]).dt.strftime("%Y-%m-%d")
    raw["key_conflict"] = raw.duplicated(["date", "idx"], keep=False)

    # Original boundaries remain barriers even in sorted candidate B.
    boundaries = (
        raw.date.ne(raw.date.shift())
        | raw.idx.diff().ne(1)
        | raw.key_conflict
        | raw.key_conflict.shift(fill_value=False)
    )
    raw["source_segment"] = boundaries.cumsum()
    summaries = {
        k: {
            "shape": list(v.shape),
            "columns": list(v.columns),
            "missing": v.isna().sum().to_dict(),
            "unique": v.nunique(dropna=False).to_dict(),
            "duplicate_rows": int(v.duplicated().sum()),
        }
        for k, v in sheets.items()
    }
    summaries["scaled"] = {
        "shape": list(scaled.shape),
        "columns": list(scaled.columns),
        "missing": scaled.isna().sum().to_dict(),
        "unique": scaled.nunique(dropna=False).to_dict(),
        "duplicate_rows": int(scaled.duplicated().sum()),
    }
    for name, df in sheets.items():
        df.to_csv(out / (name.replace(" ", "_") + ".csv"), index=False, encoding="utf-8-sig")
    daily = raw.groupby("date").agg(
        rows=("idx", "size"),
        unique_idx=("idx", "nunique"),
        idx_min=("idx", "min"),
        idx_max=("idx", "max"),
    )
    daily["missing_idx_within_range"] = daily.idx_max - daily.idx_min + 1 - daily.unique_idx
    daily.to_csv(out / "daily_counts.csv", encoding="utf-8-sig")
    raw[COLS].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_csv(
        out / "distribution.csv", encoding="utf-8-sig"
    )
    raw.groupby("date")[COLS].agg(["min", "median", "mean", "std", "max"]).to_csv(
        out / "daily_distribution.csv", encoding="utf-8-sig"
    )
    raw[raw.key_conflict].to_csv(out / "key_conflicts.csv", index=False, encoding="utf-8-sig")
    issues = raw.loc[
        boundaries, ["excel_row", "date", "idx", "key_conflict", "source_segment"]
    ].copy()
    issues["previous_date"] = raw.date.shift()[boundaries]
    issues["idx_step"] = raw.idx.diff()[boundaries]
    issues.to_csv(out / "sequence_boundaries.csv", index=False, encoding="utf-8-sig")
    result = sheets["result"].copy()
    result["date"] = pd.to_datetime(result["working time"]).dt.strftime("%Y-%m-%d")
    pivot = result.pivot_table(
        index="date", columns="defect type", values="defect", aggfunc="sum"
    ).reindex(daily.index)
    joined = daily.join(pivot.add_prefix("defect_type_"))
    joined["recorded_defect_sum"] = pivot.sum(axis=1, min_count=1)
    joined.to_csv(out / "daily_result_join.csv", encoding="utf-8-sig")

    # Long exact blocks: test every offset, verify equality in all four raw columns.
    vals = raw[COLS].to_numpy()
    seg = raw.source_segment.to_numpy()
    repeats = []
    for lag in range(1, len(raw) - 19):
        eq = np.all(vals[:-lag] == vals[lag:], axis=1)

        # A run must remain within each source segment.
        start = np.flatnonzero(
            eq
            & np.r_[
                True,
                (~eq[:-1]) | (seg[1:-lag] != seg[: -lag - 1]) | (seg[lag + 1 :] != seg[lag:-1]),
            ]
        )
        end = (
            np.flatnonzero(
                eq
                & np.r_[
                    (~eq[1:]) | (seg[1:-lag] != seg[: -lag - 1]) | (seg[lag + 1 :] != seg[lag:-1]),
                    True,
                ]
            )
            + 1
        )
        for s, e in zip(start, end):
            if e - s >= 20:
                assert np.array_equal(vals[s:e], vals[s + lag : e + lag])
                repeats.append(
                    {
                        "a_start": int(s + 2),
                        "a_end": int(e + 1),
                        "b_start": int(s + lag + 2),
                        "b_end": int(e + lag + 1),
                        "length": int(e - s),
                        "a_date": raw.date.iloc[s],
                        "b_date": raw.date.iloc[s + lag],
                    }
                )
    rep = pd.DataFrame(repeats)
    rep.to_csv(out / "exact_repeated_blocks.csv", index=False, encoding="utf-8-sig")

    # Sensitivity only: retain earlier occurrence, remove later occurrence of >=100 records.
    exclude = np.zeros(len(raw), dtype=bool)
    for r in repeats:
        if r["length"] >= 100:
            exclude[r["b_start"] - 2 : r["b_end"] - 1] = True
    raw["repeat_excluded"] = exclude
    raw.to_csv(out / "raw_audit.csv", index=False, encoding="utf-8-sig")
    sc = scaled.iloc[:, 1:].to_numpy(float)

    # Prior inferred transforms: explicitly a hypothesis, not an authenticated scaler.
    inv = sc * np.array([118.26, 139.78, 95.78, 2170]) + np.array([1.74, 14.52, 2.46, 70])
    rounded = vals.copy()
    rounded[:, 2] = np.round(rounded[:, 2], 2)
    same = np.isclose(inv, rounded, atol=1e-7, rtol=0)
    scaling = {
        "hypothesis_offset": [1.74, 14.52, 2.46, 70],
        "hypothesis_scale": [118.26, 139.78, 95.78, 2170],
        "same_row_all4_match": int(same.all(axis=1).sum()),
        "same_row_column_matches": dict(zip(NAMES, map(int, same.sum(axis=0)))),
        "missing_cells": int(np.isnan(sc).sum()),
        "warning": "Same-row comparison only. Historical 11690 aligned pairs were not recomputed here; no proven 1:1 merge.",
    }
    pd.DataFrame(inv, columns=COLS).to_csv(
        out / "scaled_inverse_hypothesis.csv", index=False, encoding="utf-8-sig"
    )
    conditions = {}
    metrics = []
    blockrows = []
    pairs = []
    changes = []
    expansions = []
    rollingframes = []
    adjsets = {}
    for order in ["A", "B"]:
        ordered = (
            raw.copy()
            if order == "A"
            else raw.sort_values(["date", "idx", "excel_row"], kind="stable").copy()
        )
        for policy in ["all", "dedup"]:
            tag = order + "_" + policy
            d = ordered.copy()
            d["order_position"] = np.arange(len(d))
            if policy == "dedup":
                d = d[~d.repeat_excluded].copy()
            boundary = (
                d.date.ne(d.date.shift())
                | d.idx.diff().ne(1)
                | d.key_conflict
                | d.key_conflict.shift(fill_value=False)
                | d.source_segment.ne(d.source_segment.shift())
                | d.order_position.diff().ne(1)
            )
            d["segment"] = boundary.cumsum()
            cond = []
            valid = {w: 0 for w in [20, 50, 100]}
            nchanges = 0
            adjsets[tag] = set(zip(d.excel_row.iloc[:-1], d.excel_row.iloc[1:]))
            for sid, g in d.groupby("segment", sort=False):
                g = g.reset_index(drop=True)
                x = g[COLS]
                n = len(g)
                z = x.to_numpy()
                sname = f"{tag}_s{sid:03d}"
                item = {
                    "id": sname,
                    "date": g.date.iloc[0],
                    "rows": g.excel_row.tolist(),
                    "idx": g.idx.tolist(),
                    "values": z.tolist(),
                    "repeat": g.repeat_excluded.tolist(),
                    "rolling": {},
                }
                for w in [20, 50, 100]:
                    hist = x.shift(1).rolling(w, min_periods=w)
                    med = hist.median()
                    q1 = hist.quantile(0.25)
                    q3 = hist.quantile(0.75)
                    std = hist.std(ddof=1)
                    iqr = q3 - q1
                    dev = x - med
                    valid[w] += int(med.iloc[:, 0].notna().sum())
                    item["rolling"][str(w)] = {
                        k: json.loads(v.to_json(orient="values"))
                        for k, v in [
                            ("median", med),
                            ("q1", q1),
                            ("q3", q3),
                            ("std", std),
                            ("deviation", dev),
                        ]
                    }
                    frame = pd.DataFrame(
                        {"condition": tag, "segment": sname, "excel_row": g.excel_row, "window": w}
                    )
                    for j, name in enumerate(NAMES):
                        for k, v in [
                            ("median", med),
                            ("iqr", iqr),
                            ("std", std),
                            ("deviation", dev),
                        ]:
                            frame[name + "_" + k] = v.iloc[:, j]
                        frame[name + "_iqr_zero"] = iqr.iloc[:, j].eq(0)
                    rollingframes.append(frame)
                    if n > w:
                        assert np.allclose(med.iloc[w], np.median(z[:w], axis=0))
                delta = x.diff().abs()
                for j, name in enumerate(NAMES):
                    v = delta.iloc[:, j]
                    threshold = v.quantile(0.99)
                    flags = (
                        (v >= threshold) & (v > 0)
                        if v.max() > 0
                        else pd.Series(False, index=v.index)
                    )
                    for i in np.flatnonzero(flags):
                        changes.append(
                            {
                                "condition": tag,
                                "segment": sname,
                                "variable": name,
                                "excel_row": int(g.excel_row.iloc[i]),
                                "delta": float(v.iloc[i]),
                                "threshold": float(threshold),
                            }
                        )
                    nchanges += int(flags.sum())
                for w in [20, 50, 100]:
                    for begin in range(0, n, w):
                        end = min(begin + w, n)
                        blockrows.append(
                            {
                                "condition": tag,
                                "segment": sname,
                                "date": g.date.iloc[0],
                                "length_requested": w,
                                "start_position": begin + 1,
                                "end_position": end,
                                "excel_start": int(g.excel_row.iloc[begin]),
                                "excel_end": int(g.excel_row.iloc[end - 1]),
                                "actual_length": end - begin,
                                "complete": end - begin == w,
                                "repeat_rows": int(g.repeat_excluded.iloc[begin:end].sum()),
                            }
                        )
                for begin in range(0, n - 99, 100):
                    chunk = z[begin : begin + 100].reshape(5, 20, 4)
                    for l, r in itertools.combinations(range(5), 2):
                        for j, name in enumerate(NAMES):
                            v, u = chunk[l, :, j], chunk[r, :, j]
                            corr = (
                                float(np.corrcoef(v, u)[0, 1])
                                if np.ptp(v) > 0 and np.ptp(u) > 0
                                else None
                            )
                            pairs.append(
                                {
                                    "condition": tag,
                                    "segment": sname,
                                    "block_start": begin + 1,
                                    "variable": name,
                                    "left": l + 1,
                                    "right": r + 1,
                                    "rmse": float(np.sqrt(np.mean((v - u) ** 2))),
                                    "centered_rmse": float(
                                        np.sqrt(np.mean(((v - v.mean()) - (u - u.mean())) ** 2))
                                    ),
                                    "correlation": corr,
                                    "exact": bool(np.array_equal(v, u)),
                                }
                            )
                conditions.setdefault(tag, []).append(item)
            metrics.append(
                {
                    "condition": tag,
                    "rows": len(d),
                    "segments": len(cond) if cond else len(conditions[tag]),
                    "valid_w20": valid[20],
                    "valid_w50": valid[50],
                    "valid_w100": valid[100],
                    "change_variable_events": nchanges,
                }
            )
    pd.concat(rollingframes, ignore_index=True).to_csv(
        out / "rolling_statistics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(metrics).to_csv(out / "condition_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(blockrows).to_csv(out / "block_index.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pairs).to_csv(out / "nested_pair_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(changes).to_csv(out / "change_candidates.csv", index=False, encoding="utf-8-sig")
    adjacency = {}
    for policy in ["all", "dedup"]:
        aa, bb = adjsets["A_" + policy], adjsets["B_" + policy]
        adjacency[policy] = {
            "A_edges": len(aa),
            "B_edges": len(bb),
            "A_edges_not_in_B": len(aa - bb),
            "B_edges_not_in_A": len(bb - aa),
            "A_changed_fraction": len(aa - bb) / len(aa),
        }
    summary = {
        "input_hashes": hashes,
        "input_sizes": {p: Path(p).stat().st_size for p in hashes},
        "sheets": summaries,
        "raw_rows": len(raw),
        "constant_columns": [c for c in sheets["Raw data"] if sheets["Raw data"][c].nunique() == 1],
        "raw_measurement_unique": len(raw[COLS].drop_duplicates()),
        "raw_measurement_duplicates": int(raw[COLS].duplicated().sum()),
        "duplicate_without_idx": int(sheets["Raw data"].drop(columns="idx").duplicated().sum()),
        "key_conflict_rows": int(raw.key_conflict.sum()),
        "excluded_repeat_rows": int(exclude.sum()),
        "repeat_pairs_ge20": len(repeats),
        "repeat_pairs_ge100": sum(r["length"] >= 100 for r in repeats),
        "longest_exact_repeat": int(rep.length.max()) if len(rep) else 0,
        "scaling": scaling,
        "conditions": metrics,
        "adjacency": adjacency,
        "result_dates": int(result.date.nunique()),
        "result_defect_sum": int(result.defect.sum()),
        "result_missing_dates": list(daily.index.difference(result.date.unique())),
        "execution": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "command": sys.argv,
            "seconds": round(time.time() - t, 2),
            "code_sha256": sha(__file__),
        },
    }
    for p, h in hashes.items():
        assert sha(p) == h, "Input changed during audit"
    assert sum(daily.rows) == len(raw)
    summary["checks"] = {
        "input_hashes_unchanged": True,
        "daily_total_reconciled": True,
        "repeat_blocks_exactly_verified": True,
        "first_rolling_window_verified_each_segment": True,
    }
    dump(out / "summary.json", summary)
    dump(out / "chart_data.json", conditions)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ["sheets", "execution"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
