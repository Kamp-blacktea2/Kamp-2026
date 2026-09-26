"""Raw-only motif search. All offsets refer to original, uncompressed rows."""

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "4"

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TABLES = HERE / "outputs/tables"
COLS = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
NAMES = ["F", "I", "V", "t"]
VIEWS = {"F": [0], "I": [1], "V": [2], "t": [3], "FIV": [0, 1, 2], "FIVt": [0, 1, 2, 3]}
REPS = ["absolute", "shape", "delta"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(rows, name):
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / f"{name}.csv", index=False, float_format="%.17g")


def read(name):
    return pd.read_csv(TABLES / f"{name}.csv", float_precision="round_trip")


def write_json(value, name):
    path = HERE / "outputs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    path = ROOT / "data/Welding_Data_Set_01.xlsx"
    assert digest(path) == config["raw_sha256"]
    raw = pd.read_excel(path, sheet_name="Raw data")
    assert raw.shape == (11939, 10) and not raw.isna().any().any()
    raw["date"] = pd.to_datetime(raw["working time"]).dt.strftime("%Y-%m-%d")
    x = raw[COLS].to_numpy(float)
    reference = x[raw.date <= "2020-03-31"]
    median = np.median(reference, axis=0)
    iqr = np.quantile(reference, .75, axis=0) - np.quantile(reference, .25, axis=0)
    std = np.std(reference, axis=0)
    scale = np.where(iqr > 0, iqr, std)
    q = []
    for column in reference.T:
        steps = np.diff(np.unique(column))
        positive = steps[steps > 1e-9]
        q.append(positive.min() if len(positive) else np.inf)
    q = np.array(q)
    assert np.all(scale > 0)
    segments = pd.read_csv(HERE.parent / "ysh-003/outputs/tables/segments.csv")
    original = segments[segments.condition == "all"].reset_index(drop=True)
    keep = np.zeros(len(x), bool)
    for row in segments[segments.condition == "controlled100"].itertuples():
        keep[row.excel_start - 2:row.excel_end - 1] = True
    assert keep.sum() == 2416
    return config, raw, x, original, keep, median, scale, q


def windows(x, segments, keep, width):
    starts, group, position, front, controlled = [], [], [], [], []
    for s, row in segments.iterrows():
        for pos in range(max(0, int(row.n) - width + 1)):
            start = int(row.excel_start) - 2 + pos
            starts.append(start)
            group.append(s)
            position.append(pos)
            front.append(pos + width <= int(.6 * row.n))
            controlled.append(keep[start:start + width].all())
    starts = np.array(starts, int)
    array = x[starts[:, None] + np.arange(width)]
    meta = pd.DataFrame(dict(start=starts, segment=group, pos=position,
                             front=front, controlled=controlled))
    meta["date"] = segments.iloc[meta.segment].date.to_numpy()
    meta["back"] = meta.pos.to_numpy() >= (.6 * segments.iloc[meta.segment].n.to_numpy()).astype(int)
    return array, meta


def anchors(meta, segments, width, burn=100, phase=0):
    selected = []
    for segment in range(len(segments)):
        valid = meta[(meta.segment == segment) & meta.front & (meta.pos >= burn + phase)]
        valid = valid[(valid.pos - burn - phase) % 4 == 0]
        index = valid.index.to_numpy()
        if len(index) > 32:
            index = index[np.linspace(0, len(index) - 1, 32).astype(int)]
        selected.extend(index)
    return np.array(selected, int)


def transform(array, view, rep, median, scale):
    values = array[:, :, view]
    if rep == "absolute":
        values = values - median[view]
    elif rep == "shape":
        values = values - values.mean(axis=1, keepdims=True)
    elif rep == "delta":
        values = np.diff(values, axis=1)
    return (values / scale[view]).reshape(len(array), -1)


def active_pair(array, a, view, q):
    activity = np.ptp(array[:, :, view], axis=1) > q[view]
    common = np.sum(activity & activity[a], axis=1)
    return common >= (1 if len(view) == 1 else 2)


def greedy(indices, distance, meta, width, count=None):
    """Distance priority, Excel tie-break, and nonoverlap within each original segment."""
    if not len(indices):
        return np.array([], int)
    order = indices[np.lexsort((meta.start.to_numpy()[indices], distance[indices]))]
    blocked = np.zeros(len(meta), bool)
    cache_key = f"nonoverlap_bounds_{width}"
    if cache_key not in meta.attrs:
        starts = meta.start.to_numpy()
        groups = meta.segment.to_numpy()
        group_start = np.maximum.accumulate(np.r_[0, np.where(np.diff(groups)!=0,np.arange(1,len(meta)),0)])
        edges = np.r_[np.flatnonzero(np.diff(groups)!=0)+1,len(meta)]
        group_end = edges[np.searchsorted(edges,np.arange(len(meta)),side="right")]
        left = np.maximum(np.searchsorted(starts,starts-width+1),group_start)
        right = np.minimum(np.searchsorted(starts,starts+width),group_end)
        meta.attrs[cache_key] = (left,right)
    left,right = meta.attrs[cache_key]
    chosen = []
    for index in order:
        if blocked[index]:
            continue
        chosen.append(index)
        blocked[left[index]:right[index]] = True
        if count is not None and len(chosen) == count:
            break
    return np.array(chosen, int)


def compare(array, meta, a, view, rep, distances, q, threshold=.25, exclusion=1, controlled=False, initial=False):
    width = array.shape[1]
    same = meta.segment.to_numpy() == meta.segment.iloc[a]
    eligible = ~(same & (np.abs(meta.pos.to_numpy() - meta.pos.iloc[a]) < exclusion * width))
    if controlled:
        eligible &= meta.controlled.to_numpy() & bool(meta.controlled.iloc[a])
    if initial:
        eligible &= meta.pos.to_numpy() >= 100
    active = active_pair(array, a, view, q)
    exact_view = np.all(array[:, :, view] == array[a:a + 1, :, view], axis=(1, 2))
    exact4 = np.all(array == array[a], axis=(1, 2))
    near4 = np.all(np.abs(array - array[a]) <= q, axis=(1, 2)) & ~exact4
    matches = eligible & (distances <= threshold)
    if rep != "absolute":
        matches &= active
    nonexact = matches & ~exact_view & active
    selected = greedy(np.flatnonzero(nonexact), distances, meta, width)
    return eligible, matches, selected, exact_view, exact4, near4, active


def context(x, segments, keep, a_start, b_start, view, rep, median, scale, q, length=8, before=0, controlled=False):
    starts = [int(a_start)-before, int(b_start)-before]
    for start in starts:
        valid = ((segments.excel_start.to_numpy()-2 <= start) &
                 (segments.excel_end.to_numpy()-1 >= start+length))
        if not valid.any():
            return np.nan, False, "segment_boundary"
        if controlled and not keep[start:start+length].all():
            return np.nan, False, "controlled_gap"
    pair = np.stack([x[s:s+length] for s in starts])
    active = active_pair(pair, 0, view, q)[1]
    z = transform(pair, view, rep, median, scale)
    distance = np.sqrt(np.mean((z[0]-z[1])**2))
    return float(distance), bool(active), "ok"


def gap_stats(positions):
    gaps = np.diff(np.sort(positions))
    if len(gaps) == 0:
        return dict(gaps=0, mode=np.nan, max_share=np.nan, multiple4=np.nan,
                    median=np.nan, iqr=np.nan, gap16_share=np.nan), gaps
    values, counts = np.unique(gaps, return_counts=True)
    order = np.lexsort((values, -counts))
    return dict(gaps=len(gaps), mode=int(values[order[0]]), max_share=counts.max()/len(gaps),
                multiple4=np.mean(gaps % 4 == 0), median=np.median(gaps),
                iqr=np.quantile(gaps,.75)-np.quantile(gaps,.25),
                gap16_share=np.mean(gaps == 16),
                top3=";".join(f"{values[i]}:{counts[i]}" for i in order[:3])), gaps
