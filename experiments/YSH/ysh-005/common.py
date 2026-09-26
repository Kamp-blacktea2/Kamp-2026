"""Shared, read-only input handling and explicit experiment configuration."""

import hashlib
import json
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

import numpy as np
import pandas as pd

NAMES = ["force", "current", "voltage", "time"]
COLS = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
TABLES = HERE / "outputs" / "tables"
CONFIG = {
    "raw_sha256": "d514d6aaa121630c04d7d51c97a56025e78e2c86868813f7722ba5db922c1f33",
    "dictionary_sha256": "c5e1635cb81a44510e146cf71bd78321272cc28208ce26835586340309e2ba43",
    "train_last_date": "2020-03-31",
    "periods": [2, 3, 4, 5, 8, 16],
    "phase_train_fraction": 0.6,
    "surrogates": 100,
    "block": 64,
    "seeds": [42, 7, 2026],
    "force_thresholds": [2.8, 3.0, 3.2],
    "threads": 4,
    "ae_epochs": 50,
    "ae_batch": 64,
    "ae_lr": 0.01,
    "ae_std_multiplier": 8,
    "if_trees": 300,
    "if_quantiles": [0.95, 0.99],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(rows, name):
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / f"{name}.csv", index=False, float_format="%.15g")


def read(name):
    return pd.read_csv(TABLES / f"{name}.csv")


def json_save(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def load():
    path = ROOT / "data" / "Welding_Data_Set_01.xlsx"
    assert digest(path) == CONFIG["raw_sha256"]
    assert digest(ROOT / "docs" / "data_dictionary.md") == CONFIG["dictionary_sha256"]
    raw = pd.read_excel(path, sheet_name="Raw data")
    assert raw.shape == (11939, 10) and not raw[COLS].isna().any().any()
    raw = raw.rename(columns=dict(zip(COLS, NAMES)))
    raw["date"] = pd.to_datetime(raw["working time"]).dt.strftime("%Y-%m-%d")
    raw["excel_row"] = np.arange(2, len(raw) + 2)
    raw["train"] = raw.date <= CONFIG["train_last_date"]
    assert raw.train.sum() == 8470
    assert np.all(raw.idx.to_numpy() == raw.idx.to_numpy().astype(int))
    segments = pd.read_csv(HERE.parent / "ysh-003" / "outputs" / "tables" / "segments.csv")
    for s in segments.itertuples():
        part = raw.iloc[s.excel_start - 2 : s.excel_end - 1]
        assert len(part) == s.n and (part.date == s.date).all()
        assert np.all(np.diff(part.idx) == 1)
    keep = np.zeros(len(raw), dtype=bool)
    for s in segments[segments.condition == "controlled100"].itertuples():
        keep[s.excel_start - 2 : s.excel_end - 1] = True
    assert keep.sum() == 2416
    raw["controlled100"] = keep
    return raw, segments


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def segment_part(raw, s):
    return raw.iloc[s.excel_start - 2 : s.excel_end - 1]


def group_values(raw, groups):
    start = groups.excel_start.to_numpy(int) - 2
    return raw[NAMES].to_numpy()[start[:, None] + np.arange(4)]
