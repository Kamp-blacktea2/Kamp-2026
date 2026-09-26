"""Independent checks of selected motif pairs and immutable input."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "outputs" / "tables"
COLS = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
NAMES = ["force", "current", "voltage", "time"]


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    raw = pd.read_excel(config["raw"], sheet_name="Raw data")
    details = pd.read_csv(TABLES / "selected_pair_details.csv")
    scan = pd.read_csv(TABLES / "motif_scan.csv")
    segments = pd.read_csv(TABLES / "segments.csv").set_index("segment")
    checks = 0
    for keys, group in details.groupby(["condition", "phase", "length", "offset", "variable"]):
        sample = group.iloc[sorted(set([0, len(group) // 2, len(group) - 1]))]
        for row in sample.itertuples():
            column = COLS[NAMES.index(row.variable)]
            a = raw[column].iloc[row.a_excel - 2 : row.a_excel - 2 + row.length].to_numpy()
            b = raw[column].iloc[row.b_excel - 2 : row.b_excel - 2 + row.length].to_numpy()
            assert row.b_excel == row.a_excel + row.length
            meta = segments.loc[row.segment]
            middle = meta.excel_start + meta.n // 2
            if row.phase == "discovery":
                assert row.b_excel + row.length <= middle
            else:
                assert row.a_excel >= middle and row.b_excel + row.length <= meta.excel_end + 1
            ac, bc = a - a.mean(), b - b.mean()
            assert np.isclose(row.raw_rmse, np.sqrt(np.mean((a - b) ** 2)), atol=1e-10)
            assert np.isclose(row.centered_rmse, np.sqrt(np.mean((ac - bc) ** 2)), atol=1e-10)
            if np.ptp(a) > 0 and np.ptp(b) > 0:
                corr = np.corrcoef(a, b)[0, 1]
                error = np.sqrt(np.sum((ac - bc) ** 2) / (np.sum(ac * ac) + np.sum(bc * bc)))
                amplitude = b.std() / a.std()
                assert np.isclose(row.corr, corr, atol=1e-10)
                assert np.isclose(row.shape_error, error, atol=1e-10)
                assert np.isclose(row.amplitude, amplitude, atol=1e-10)
                expected = corr >= 0.7 and error <= 0.75 and 0.5 <= amplitude <= 2
                # Values exactly at the boundary are compared with direct implementation precision.
                if (
                    min(
                        abs(corr - 0.7), abs(error - 0.75), abs(amplitude - 0.5), abs(amplitude - 2)
                    )
                    > 1e-10
                ):
                    assert bool(row.passed) == expected
            else:
                assert pd.isna(row.corr) and not row.passed
            checks += 1

    for keys, group in details.groupby(
        ["condition", "segment", "phase", "length", "offset", "variable"]
    ):
        condition, segment, phase, length, offset, variable = keys
        selected = scan[
            (scan.condition == condition)
            & (scan.segment == segment)
            & (scan.phase == phase)
            & (scan.length == length)
            & (scan.offset == offset)
            & (scan.variable == variable)
        ]
        assert len(selected) == 1
        row = selected.iloc[0]
        near = group[~group.exact_all]
        assert row.pairs == len(group)
        assert row.nonexact_pairs == len(near)
        assert row.passed_nonexact_pairs == near.passed.sum()

    ranks = pd.read_csv(TABLES / "candidate_ranking.csv")
    selected = pd.read_csv(TABLES / "selected_candidates.csv")
    expected = (
        ranks.sort_values(
            ["variable", "mean_pass_rate", "median_corr", "length"],
            ascending=[True, False, False, True],
        )
        .groupby("variable", sort=False)
        .head(3)
    )
    assert list(zip(expected.variable, expected.length)) == list(
        zip(selected.variable, selected.length)
    )
    assert (scan.passed_nonexact_pairs <= scan.nonexact_pairs).all()
    assert (scan.valid_nonexact_pairs <= scan.nonexact_pairs).all()
    assert set(scan.length) == set(range(2, 101))
    assert hashlib.sha256(Path(config["raw"]).read_bytes()).hexdigest() == config["expected_sha256"]
    result = {
        "independent_pair_checks": checks,
        "summary_detail_reconciliation": True,
        "candidate_ranking_verified": True,
        "all_lengths_2_to_100_present": True,
        "pair_denominators_valid": True,
        "partition_boundaries_checked": True,
        "input_unchanged": True,
    }
    (ROOT / "outputs" / "verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
