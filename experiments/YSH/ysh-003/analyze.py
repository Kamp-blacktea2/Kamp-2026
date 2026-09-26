"""Approximate repeated motifs, with explicit discovery and confirmation partitions."""

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


def finite_median(values):
    values = np.asarray(values)
    return float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else np.nan


def pair_metrics(a, b, config):
    """Arrays: pairs x records x variables. Never warp or rescale blocks."""
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    ea = np.sum(ac * ac, axis=1)
    eb = np.sum(bc * bc, axis=1)
    valid_a = np.ptp(a, axis=1) > 0
    valid_b = np.ptp(b, axis=1) > 0
    ea = np.where(valid_a, ea, 0)
    eb = np.where(valid_b, eb, 0)
    valid = valid_a & valid_b
    corr = np.divide(
        np.sum(ac * bc, axis=1), np.sqrt(ea * eb), out=np.full_like(ea, np.nan), where=valid
    )
    energy = ea + eb
    shape_error = np.sqrt(
        np.divide(
            np.sum((ac - bc) ** 2, axis=1), energy, out=np.full_like(ea, np.nan), where=energy > 0
        )
    )
    amplitude = np.sqrt(np.divide(eb, ea, out=np.full_like(ea, np.nan), where=ea > 0))
    passed = (
        valid & (corr >= config["correlation_min"]) & (shape_error <= config["shape_error_max"])
    )
    passed &= (amplitude >= config["amplitude_min"]) & (amplitude <= config["amplitude_max"])
    exact_variable = np.all(a == b, axis=1)
    exact_all = np.all(exact_variable, axis=1)
    return {
        "corr": corr,
        "shape_error": shape_error,
        "amplitude": amplitude,
        "raw_rmse": np.sqrt(np.mean((a - b) ** 2, axis=1)),
        "centered_rmse": np.sqrt(np.mean((ac - bc) ** 2, axis=1)),
        "direction_agreement": np.mean(
            np.sign(np.diff(a, axis=1)) == np.sign(np.diff(b, axis=1)), axis=1
        ),
        "valid": valid,
        "passed": passed,
        "exact_variable": exact_variable,
        "exact_all": exact_all,
    }


def blocks(values, begin, end, length, offset):
    first = begin + offset
    count = max(0, (end - first) // length)
    if count < 2:
        return None, None
    data = values[first : first + count * length].reshape(count, length, 4)
    positions = first + np.arange(count) * length
    return data, positions


def summarize(meta, metrics):
    near = ~metrics["exact_all"]
    n_near = int(near.sum())
    common = {
        **meta,
        "pairs": len(near),
        "exact_all_pairs": int((~near).sum()),
        "nonexact_pairs": n_near,
        "two_variable_pass_rate": (
            float(np.mean(metrics["passed"][near].sum(axis=1) >= 2)) if n_near else np.nan
        ),
        "four_variable_pass_rate": (
            float(np.mean(metrics["passed"][near].all(axis=1))) if n_near else np.nan
        ),
    }
    rows = []
    for j, name in enumerate(NAMES):
        changed = near & ~metrics["exact_variable"][:, j]
        row = {
            **common,
            "variable": name,
            "valid_nonexact_pairs": int((near & metrics["valid"][:, j]).sum()),
            "passed_nonexact_pairs": int((near & metrics["passed"][:, j]).sum()),
            "pass_rate": float(metrics["passed"][near, j].mean()) if n_near else np.nan,
            "identical_variable_pairs": int((near & metrics["exact_variable"][:, j]).sum()),
            "changed_variable_pairs": int(changed.sum()),
            "changed_variable_passed": int(metrics["passed"][changed, j].sum()),
        }
        for metric in [
            "corr",
            "shape_error",
            "amplitude",
            "raw_rmse",
            "centered_rmse",
            "direction_agreement",
        ]:
            row[metric] = finite_median(metrics[metric][near, j])
        rows.append(row)
    return rows


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
    assert sha(config["raw"]) == config["expected_sha256"]
    raw = pd.read_excel(config["raw"], sheet_name="Raw data")
    assert raw.shape == (11939, 10)
    values = raw[COLS].to_numpy(float)
    source_segments = pd.read_csv(config["segments"])
    source_segments = source_segments[
        source_segments.condition.isin(["all", "controlled100"])
    ].copy()
    source_segments.to_csv(tables / "segments.csv", index=False)
    lookup = {}
    for r in source_segments.itertuples():
        part = raw.iloc[r.excel_start - 2 : r.excel_end - 1]
        assert len(part) == r.n
        assert part["working time"].nunique() == 1
        assert len(part) == 1 or np.all(np.diff(part.idx.to_numpy()) == 1)
        lookup[r.segment] = (r, part[COLS].to_numpy(float))

    summary_rows = []
    partitions = []
    for r, data in lookup.values():
        midpoint = len(data) // 2
        for phase, begin, end in [
            ("discovery", 0, midpoint),
            ("confirmation", midpoint, len(data)),
        ]:
            partitions.append(
                {
                    "condition": r.condition,
                    "segment": r.segment,
                    "phase": phase,
                    "records": end - begin,
                    "excel_start": r.excel_start + begin,
                    "excel_end": r.excel_start + end - 1,
                }
            )
            for length in range(config["length_min"], config["length_max"] + 1):
                for offset in sorted(set([0, length // 2])):
                    data_blocks, positions = blocks(data, begin, end, length, offset)
                    if data_blocks is None:
                        continue
                    metrics = pair_metrics(data_blocks[:-1], data_blocks[1:], config)
                    meta = {
                        "condition": r.condition,
                        "segment": r.segment,
                        "phase": phase,
                        "length": length,
                        "offset": offset,
                        "partition_records": end - begin,
                        "full_blocks": len(data_blocks),
                        "omitted_records": end - begin - len(data_blocks) * length,
                    }
                    summary_rows.extend(summarize(meta, metrics))
        print(f"Scanned {r.segment}", flush=True)

    scan = pd.DataFrame(summary_rows)
    scan.to_csv(tables / "motif_scan.csv", index=False)
    pd.DataFrame(partitions).to_csv(tables / "partitions.csv", index=False)
    eligible = scan[
        (scan.condition == "all")
        & (scan.phase == "discovery")
        & (scan.length >= 8)
        & (scan.nonexact_pairs >= 5)
        & (scan.valid_nonexact_pairs >= 5)
    ]
    ranking = (
        eligible.groupby(["variable", "length"])
        .agg(
            mean_pass_rate=("pass_rate", "mean"),
            median_corr=("corr", "median"),
            groups=("segment", "size"),
            segments=("segment", "nunique"),
            compared_pairs=("nonexact_pairs", "sum"),
            median_raw_rmse=("raw_rmse", "median"),
            median_centered_rmse=("centered_rmse", "median"),
            median_shape_error=("shape_error", "median"),
        )
        .reset_index()
    )
    selected = (
        ranking.sort_values(
            ["variable", "mean_pass_rate", "median_corr", "length"],
            ascending=[True, False, False, True],
        )
        .groupby("variable", sort=False)
        .head(3)
        .copy()
    )
    selected["rank"] = selected.groupby("variable").cumcount() + 1
    ranking.to_csv(tables / "candidate_ranking.csv", index=False)
    selected.to_csv(tables / "selected_candidates.csv", index=False)

    # Re-evaluate only selected lengths for traceable pair examples and joint-variable counts.
    selected_lengths = sorted(selected.length.unique())
    pairs = []
    template_results = []
    for r, data in lookup.values():
        midpoint = len(data) // 2
        for length in selected_lengths:
            for offset in sorted(set([0, length // 2])):
                for phase, begin, end in [
                    ("discovery", 0, midpoint),
                    ("confirmation", midpoint, len(data)),
                ]:
                    data_blocks, positions = blocks(data, begin, end, length, offset)
                    if data_blocks is None:
                        continue
                    metrics = pair_metrics(data_blocks[:-1], data_blocks[1:], config)
                    for i in range(len(data_blocks) - 1):
                        common = {
                            "condition": r.condition,
                            "segment": r.segment,
                            "phase": phase,
                            "length": length,
                            "offset": offset,
                            "a_excel": int(r.excel_start + positions[i]),
                            "b_excel": int(r.excel_start + positions[i + 1]),
                            "exact_all": bool(metrics["exact_all"][i]),
                            "variables_passed": int(metrics["passed"][i].sum()),
                        }
                        for j, name in enumerate(NAMES):
                            row = {
                                **common,
                                "variable": name,
                                "passed": bool(metrics["passed"][i, j]),
                                "exact_variable": bool(metrics["exact_variable"][i, j]),
                            }
                            row.update(
                                {
                                    key: metrics[key][i, j]
                                    for key in [
                                        "corr",
                                        "shape_error",
                                        "amplitude",
                                        "raw_rmse",
                                        "centered_rmse",
                                        "direction_agreement",
                                    ]
                                }
                            )
                            pairs.append(row)

                # Confirmation blocks keep the absolute phase anchored in the discovery half.
                training, training_positions = blocks(data, 0, midpoint, length, offset)
                confirmation_offset = (offset - midpoint) % length
                testing, testing_positions = blocks(
                    data, midpoint, len(data), length, confirmation_offset
                )
                if training is None or testing is None:
                    continue
                template = training.mean(axis=0)
                repeated_template = np.broadcast_to(template, testing.shape)
                metrics = pair_metrics(repeated_template, testing, config)
                for j, name in enumerate(NAMES):
                    boundary_indices = testing_positions[1:]
                    all_steps = np.arange(testing_positions[0] + 1, testing_positions[-1] + length)
                    internal = all_steps[~np.isin(all_steps, boundary_indices)]
                    boundary_change = np.abs(
                        data[boundary_indices, j] - data[boundary_indices - 1, j]
                    )
                    internal_change = np.abs(data[internal, j] - data[internal - 1, j])
                    template_results.append(
                        {
                            "condition": r.condition,
                            "segment": r.segment,
                            "length": length,
                            "offset": offset,
                            "variable": name,
                            "training_blocks": len(training),
                            "confirmation_blocks": len(testing),
                            "template_corr": finite_median(metrics["corr"][:, j]),
                            "template_raw_rmse": finite_median(metrics["raw_rmse"][:, j]),
                            "template_centered_rmse": finite_median(metrics["centered_rmse"][:, j]),
                            "template_pass_rate": float(metrics["passed"][:, j].mean()),
                            "boundary_steps": len(boundary_change),
                            "internal_steps": len(internal_change),
                            "boundary_mean_change": float(boundary_change.mean()),
                            "internal_mean_change": float(internal_change.mean()),
                            "boundary_median_change": finite_median(boundary_change),
                            "internal_median_change": finite_median(internal_change),
                        }
                    )
    pd.DataFrame(pairs).to_csv(tables / "selected_pair_details.csv", index=False)
    pd.DataFrame(template_results).to_csv(tables / "template_boundary_checks.csv", index=False)
    result = {
        "raw_sha256": sha(config["raw"]),
        "plan_sha256_at_execution": sha(root / "실험계획.md"),
        "code_sha256": sha(__file__),
        "config_sha256": sha(config_path),
        "python": sys.version,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "seconds": time.time() - start,
        "summary_rows": len(scan),
        "selected_lengths": selected_lengths,
        "selected_candidates": selected.to_dict(orient="records"),
        "detail_rows": len(pairs),
        "template_rows": len(template_results),
    }
    assert result["raw_sha256"] == config["expected_sha256"]
    (out / "summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
