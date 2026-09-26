"""Check migration without rerunning experiments or changing their result files."""

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".venv"
assert Path(sys.prefix).resolve() == ENV.resolve()
checks = []


def record(name, detail):
    checks.append({"check": name, "passed": True, "detail": detail})
    print(f"PASS: {name}", flush=True)


packages = {}
for package, module_name in (
    ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
    ("scikit-learn", "sklearn"), ("openpyxl", "openpyxl"), ("ruptures", "ruptures"),
    ("torch", "torch"), ("matplotlib", "matplotlib"), ("plotly", "plotly"),
    ("joblib", "joblib"), ("threadpoolctl", "threadpoolctl"),
):
    module = importlib.import_module(module_name)
    location = Path(module.__file__).resolve()
    assert location.is_relative_to(ENV.resolve()), (package, location)
    packages[package] = {"version": importlib.metadata.version(package), "path": str(location)}
record("all analysis packages loaded from shared .venv", packages)

entries = []
for number in range(1, 6):
    folder = ROOT / "experiments/YSH" / f"ysh-{number:03}"
    scripts = ["analyze"] if number < 5 else ["common", "analyze_structure", "model_candidates", "compare_result", "diagnostics"]
    code = (
        "import sys,importlib; "
        "sys.path.insert(0,sys.argv[1]); "
        "[importlib.import_module(n) for n in sys.argv[2:]]"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code, str(folder), *scripts],
                            capture_output=True, text=True)
    assert result.returncode == 0, (folder, result.stdout, result.stderr)
    entries.append({"experiment": folder.name, "modules": scripts})
record("YSH-001 through 005 analysis modules import successfully", entries)

import numpy as np
import pandas as pd
import joblib
import torch
import ruptures as rpt
from threadpoolctl import threadpool_limits

folder = ROOT / "experiments/YSH/ysh-005"
sys.path.insert(0, str(folder))
import common
from analyze_structure import predict_phase

raw, segments = common.load()
x = raw[common.NAMES].to_numpy()
assert raw.shape[0] == 11939 and raw.train.sum() == 8470
record("Excel loading, raw/dictionary hashes, and segment continuity", {"rows": len(raw)})

with threadpool_limits(limits=4):
    summary = common.read("phase_metrics")
    expected = summary[(summary.condition == "all") & (summary.segment == "all_s01") &
                       (summary.origin == "idx") & (summary.variable == "voltage") & (summary.p == 4)].iloc[0]
    part = raw.iloc[:1200]
    mae, rmse = predict_phase(part.voltage.to_numpy(), (part.idx.to_numpy(int) - 1) % 4, int(expected.cut), 4)
    assert np.allclose([mae, rmse], [expected.m2_mae, expected.m2_rmse], atol=1e-10)
    record("Ridge phase prediction matches saved YSH-005 metrics", {"mae": mae, "rmse": rmse})

    key = "all_r0_S8_standard"
    saved = joblib.load(folder / "outputs/models" / f"{key}.joblib")
    groups = common.read("candidate_scores")
    groups = groups[groups.key == key]
    values = common.group_values(raw, groups)
    features = np.column_stack([values.mean(axis=1), np.ptp(values, axis=1)])
    z = saved["scaler"].transform(features)
    assert np.array_equal(saved["gmm"].predict(z), groups.state.to_numpy())
    assert np.allclose(-saved["if"].score_samples(z), groups.if_score.to_numpy(), atol=1e-10)
    record("saved GMM and IF inference match existing outputs", {"groups": len(groups)})

torch.set_num_threads(4)
model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.RReLU(), torch.nn.Linear(3, 2),
                            torch.nn.RReLU(), torch.nn.Linear(2, 3), torch.nn.RReLU(), torch.nn.Linear(3, 4))
model.load_state_dict(torch.load(folder / "outputs/models/ae_42.pt", map_location="cpu", weights_only=True))
model.eval()
arrays = np.load(folder / "outputs/models/ae_42_reconstruction.npz")
with torch.no_grad():
    output = model(torch.tensor(arrays["input"]))
assert np.allclose(output.numpy(), arrays["output"], atol=1e-7)
record("saved CPU autoencoder reconstruction matches", {"rows": len(output), "torch": torch.__version__})

# Real process values exercise the installed change-point extension.
points = rpt.Pelt(model="l2", min_size=16, jump=1).fit(x[:127]).predict(pen=10)
assert points[-1] == 127 and all(a < b for a, b in zip([0] + points[:-1], points))
record("ruptures operates on actual process records", {"breakpoints": points})

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from plotly.offline import get_plotlyjs
fig, ax = plt.subplots()
ax.plot(x[:20, 2])
fig.canvas.draw()
plt.close(fig)
assert len(get_plotlyjs()) > 1000
record("matplotlib render and Plotly bundled JavaScript available", "memory only; no experiment output written")

report = {"passed": True, "python": sys.version, "executable": sys.executable,
          "packages": packages, "checks": checks, "check_count": len(checks),
          "scope": "Environment compatibility checks only; no full experiment rerun or result regeneration.",
          "legacy_deps_present": {f"ysh-{n:03}": (ROOT / f"experiments/YSH/ysh-{n:03}/.deps").exists() for n in (4, 5)}}
(ROOT / "docs/environment_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"ENVIRONMENT VALIDATED: {len(checks)} groups of checks")
