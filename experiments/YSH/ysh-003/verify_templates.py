from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

root = Path("F:/Kamp/experiments/YSH/ysh-003")
out = root / "outputs"
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
raw = pd.read_excel(config["raw"])
columns = dict(
    force="weld force(bar)",
    current="weld current(kA)",
    voltage="weld Voltage(v)",
    time="weld time(ms)",
)
segments = pd.read_csv(out / "tables/segments.csv").set_index(["condition", "segment"])
results = pd.read_csv(out / "tables/template_boundary_checks.csv")


def same(a, b):
    assert np.isclose(a, b, rtol=1e-8, atol=1e-10, equal_nan=True), (a, b)


for row in results.itertuples():
    seg = segments.loc[(row.condition, row.segment)]
    values = raw.iloc[int(seg.excel_start) - 2 : int(seg.excel_end) - 1][
        columns[row.variable]
    ].to_numpy(float)
    n, offset, midpoint = row.length, row.offset, len(values) // 2
    train_starts = list(range(offset, midpoint - n + 1, n))
    test_starts = list(range(midpoint + (offset - midpoint) % n, len(values) - n + 1, n))
    assert len(train_starts) == row.training_blocks
    assert len(test_starts) == row.confirmation_blocks
    assert all(start % n == offset for start in test_starts)
    template = np.array([values[s : s + n] for s in train_starts]).mean(axis=0)
    centered = template - template.mean()
    correlations, raw_errors, centered_errors = [], [], []
    for s in test_starts:
        block = values[s : s + n]
        bc = block - block.mean()
        if np.ptp(template) > 0 and np.ptp(block) > 0:
            correlations.append(np.corrcoef(template, block)[0, 1])
        raw_errors.append(np.sqrt(np.mean((template - block) ** 2)))
        centered_errors.append(np.sqrt(np.mean((centered - bc) ** 2)))
    same(np.median(correlations) if correlations else np.nan, row.template_corr)
    same(np.median(raw_errors), row.template_raw_rmse)
    same(np.median(centered_errors), row.template_centered_rmse)
    boundaries = test_starts[1:]
    interiors = [s for s in range(test_starts[0] + 1, test_starts[-1] + n) if s not in boundaries]
    assert len(boundaries) == row.boundary_steps
    assert len(interiors) == row.internal_steps
    for positions, name in [(boundaries, "boundary"), (interiors, "internal")]:
        changes = [abs(values[s] - values[s - 1]) for s in positions]
        same(np.mean(changes), getattr(row, name + "_mean_change"))
        same(np.median(changes), getattr(row, name + "_median_change"))

summary = json.loads((out / "summary.json").read_text())
assert hashlib.sha256((root / "analyze.py").read_bytes()).hexdigest() == summary["code_sha256"]
assert hashlib.sha256(Path(config["raw"]).read_bytes()).hexdigest() == summary["raw_sha256"]
checks = dict(
    template_rows_checked=len(results),
    absolute_phase_checked=True,
    boundary_metrics_checked=True,
    input_and_analysis_hashes_valid=True,
)
(out / "final_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
print(checks)
