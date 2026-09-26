"""Definitions and read-only process inputs for YSH-006A."""

import hashlib
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "4"

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TABLES = HERE / "outputs/tables"
RAW = ROOT / "data/Welding_Data_Set_01.xlsx"
COLS = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
BASE = ["F", "I", "V", "t"]
FEATURES = BASE + ["P", "E", "H", "R", "E_F", "H_F"]
DIFFERENCES = BASE + ["P", "E", "H", "R"]
UNITS = dict(zip(FEATURES, ["bar", "kA", "V", "ms", "W", "J", "A2*s", "ohm", "J/bar", "A2*s/bar"]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(value, name):
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(value).to_csv(TABLES / f"{name}.csv", index=False, float_format="%.17g")


def read(name):
    return pd.read_csv(TABLES / f"{name}.csv", float_precision="round_trip")


def json_save(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_config(path=None):
    return json.loads(Path(path or HERE / "config.json").read_text(encoding="utf-8"))


def load_process(config):
    assert digest(RAW) == config["raw_sha256"]
    assert digest(ROOT / "docs/data_dictionary.md") == config["dictionary_sha256"]
    raw = pd.read_excel(RAW, sheet_name="Raw data")
    assert raw.shape == (11939, 10) and not raw[COLS].isna().any().any()
    process = raw.rename(columns=dict(zip(COLS, BASE))).copy()
    process["date"] = pd.to_datetime(process["working time"]).dt.strftime("%Y-%m-%d")
    process["excel_row"] = np.arange(2, len(process) + 2)
    process["reference_period"] = process.date <= config["reference_last_date"]
    assert process.reference_period.sum() == 8470
    assert (process.F > 0).all() and (process.I > 0).all()
    process["P"] = process.V * (1000 * process.I)
    process["E"] = process.P * (process.t / 1000)
    process["H"] = (1000 * process.I) ** 2 * (process.t / 1000)
    process["R"] = process.V / (1000 * process.I)
    process["E_F"] = process.E / process.F
    process["H_F"] = process.H / process.F
    process["tuple_id"] = pd.factorize(pd.MultiIndex.from_frame(process[BASE]), sort=False)[0]
    segments = pd.read_csv(HERE.parent / "ysh-003/outputs/tables/segments.csv")
    keep = np.zeros(len(raw), dtype=bool)
    transitions = []
    for segment in segments.itertuples():
        indices = np.arange(segment.excel_start - 2, segment.excel_end - 1)
        part = process.iloc[indices]
        assert len(part) == segment.n and (part.date == segment.date).all()
        assert np.all(np.diff(part.idx.to_numpy()) == 1)
        if segment.condition == "controlled100":
            keep[indices] = True
        else:
            process.loc[indices, "segment"] = segment.segment
            for index in indices[1:]:
                entry = dict(date=process.date.iloc[index], segment=segment.segment,
                             previous_excel=index + 1, current_excel=index + 2,
                             previous_id=int(process.tuple_id.iloc[index - 1]),
                             current_id=int(process.tuple_id.iloc[index]),
                             reference_period=bool(process.reference_period.iloc[index]))
                for feature in DIFFERENCES:
                    entry[feature] = process[feature].iloc[index] - process[feature].iloc[index - 1]
                transitions.append(entry)
    assert keep.sum() == 2416 and len(transitions) == 11939 - 12
    process["controlled100"] = keep
    transition = pd.DataFrame(transitions)
    transition["controlled100"] = keep[transition.previous_excel.to_numpy(int) - 2] & keep[transition.current_excel.to_numpy(int) - 2]
    assert transition.controlled100.sum() == 2416 - 9
    return process, transition, segments


def select_condition(process, transition, condition):
    if condition == "all":
        return process, transition
    if condition == "unique4":
        return (process.drop_duplicates(["date", "tuple_id"]),
                transition.drop_duplicates(["date", "previous_id", "current_id"]))
    if condition == "controlled100":
        return process[process.controlled100], transition[transition.controlled100]
    raise ValueError(condition)


def summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {name: np.nan for name in ("mean", "median", "std", "mad", "iqr", "p10", "p25", "p75", "p90", "min", "max")}
    quantiles = np.quantile(values, [.1, .25, .5, .75, .9], method="linear")
    return dict(mean=values.mean(), median=quantiles[2], std=values.std(ddof=0),
                mad=np.median(np.abs(values - quantiles[2])), iqr=quantiles[3] - quantiles[1],
                p10=quantiles[0], p25=quantiles[1], p75=quantiles[3], p90=quantiles[4],
                min=values.min(), max=values.max())
