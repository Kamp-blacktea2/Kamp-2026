from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "outputs"
FEATURES = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
SHORT = ["F", "I", "V", "t"]


def read_config() -> dict:
    return json.loads((HERE / "config.json").read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_inputs(config: dict):
    path = ROOT / "data" / "Welding_Data_Set_01.xlsx"
    digest = sha256(path)
    if digest != config["raw_sha256"]:
        raise ValueError(f"Raw SHA-256 mismatch: {digest}")
    raw = pd.read_excel(path, sheet_name="Raw data")
    result = pd.read_excel(path, sheet_name="result")
    if len(raw) != 11939 or set(config["features"]) - set(raw.columns):
        raise ValueError("Raw structure differs from the plan")
    raw = raw.copy()
    raw["excel_row"] = np.arange(len(raw)) + 2
    raw["date"] = pd.to_datetime(raw["working time"]).dt.strftime("%Y-%m-%d")
    if raw[config["features"]].isna().any().any():
        raise ValueError("Unexpected Raw feature NA")
    if sorted(raw["date"].unique()) != config["dates"]:
        raise ValueError("Unexpected Raw dates")
    regime = {d: r for r, days in config["regime"].items() for d in days}
    raw["regime"] = raw["date"].map(regime)
    if raw["regime"].isna().any():
        raise ValueError("Regime assignment missing")
    result = result.copy()
    result["date"] = pd.to_datetime(result["working time"]).dt.strftime("%Y-%m-%d")
    quality = result.pivot_table(index="date", columns="defect type", values="defect",
                                 aggfunc="first").reindex(config["dates"])
    quality = quality.rename(columns={1: "type1", 2: "type2", 3: "type3"})
    for c in ("type1", "type2", "type3"):
        if c not in quality:
            quality[c] = np.nan
    quality = quality[["type1", "type2", "type3"]].reset_index().rename(columns={"index": "date"})
    quality["raw_rows"] = quality["date"].map(raw.groupby("date").size())
    quality["regime"] = quality["date"].map(regime)
    seg = pd.read_csv(HERE.parent / "ysh-003" / "outputs" / "tables" / "segments.csv")
    all_seg = seg.loc[seg["condition"].eq("all")].copy()
    controlled = seg.loc[seg["condition"].eq("controlled100")].copy()
    if all_seg["n"].sum() != len(raw) or controlled["n"].sum() != 2416:
        raise ValueError("Segment counts differ")
    ctl_mask = np.zeros(len(raw), dtype=bool)
    for row in controlled.itertuples(index=False):
        ctl_mask[(raw["excel_row"] >= row.excel_start) &
                 (raw["excel_row"] <= row.excel_end)] = True
    if ctl_mask.sum() != 2416:
        raise ValueError("controlled100 mask mismatch")
    unique_mask = ~raw.duplicated(["date"] + config["features"], keep="first")
    return raw, quality, all_seg, ctl_mask, unique_mask.to_numpy(), digest


def scaler_for(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler(quantile_range=(25, 75))
    raise ValueError(name)


def fit_model(x: np.ndarray, scaler_name: str, k: int, covariance: str,
              seed: int, gmm_config: dict):
    scaler = scaler_for(scaler_name).fit(x)
    z = scaler.transform(x)
    if not np.isfinite(z).all() or np.any(scaler.scale_ <= 0):
        raise ValueError("Nonfinite scaled input")
    model = GaussianMixture(n_components=k, covariance_type=covariance,
                            random_state=seed, **gmm_config).fit(z)
    return scaler, model


def raw_loglik(model, scaler, x: np.ndarray) -> np.ndarray:
    z = scaler.transform(x)
    return model.score_samples(z) - np.log(scaler.scale_).sum()


def raw_bic_aic(model, scaler, x: np.ndarray) -> tuple[float, float]:
    z = scaler.transform(x)
    correction = 2.0 * len(x) * np.log(scaler.scale_).sum()
    return model.bic(z) + correction, model.aic(z) + correction


def component_means_raw(model, scaler) -> np.ndarray:
    return scaler.inverse_transform(model.means_)


def align_labels(reference: np.ndarray, candidate: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if len(reference) != len(candidate):
        raise ValueError("Cannot align different K")
    cost = np.linalg.norm((candidate[:, None, :] - reference[None, :, :]) /
                          np.maximum(scale[None, None, :], 1e-9), axis=2)
    row, col = linear_sum_assignment(cost)
    mapping = np.empty(len(candidate), dtype=int)
    mapping[row] = col
    return mapping


def d2_and_residual(model, scaler, x: np.ndarray):
    z = scaler.transform(x)
    posterior = model.predict_proba(z)
    assigned = posterior.argmax(axis=1)
    delta = z - model.means_[assigned]
    d2 = np.empty(len(x), dtype=float)
    white = np.empty_like(delta)
    for k in range(model.n_components):
        idx = np.flatnonzero(assigned == k)
        if not len(idx):
            continue
        d = delta[idx]
        if model.covariance_type == "diag":
            eigenvalues = np.maximum(model.covariances_[k], 1e-12)
            white[idx] = d / np.sqrt(eigenvalues)
            d2[idx] = np.sum(d * d / eigenvalues, axis=1)
        else:
            cov = model.covariances_[k]
            eigenvalues, vectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            white[idx] = (d @ vectors) / np.sqrt(eigenvalues)
            d2[idx] = np.sum(white[idx] ** 2, axis=1)
    raw_residual = x - component_means_raw(model, scaler)[assigned]
    return posterior, assigned, d2, white, raw_residual


def score_frame(model, scaler, x: np.ndarray, original: pd.DataFrame,
                mapping: np.ndarray | None = None) -> pd.DataFrame:
    posterior, assigned, d2, _, _ = d2_and_residual(model, scaler, x)
    mapped = mapping[assigned] if mapping is not None else assigned
    mapped_post = np.zeros_like(posterior)
    if mapping is None:
        mapped_post = posterior
    else:
        mapped_post[:, mapping] = posterior
    safe = np.clip(posterior, 1e-300, 1)
    entropy = -(safe * np.log(safe)).sum(axis=1)
    if model.n_components > 1:
        entropy /= np.log(model.n_components)
    out = original[["excel_row", "date", "regime"]].reset_index(drop=True).copy()
    out["assigned_component"] = mapped
    out["posterior_max"] = posterior.max(axis=1)
    out["entropy"] = entropy
    out["global_nll"] = -raw_loglik(model, scaler, x)
    out["mahalanobis_d2"] = d2
    for k in range(model.n_components):
        out[f"posterior_C{k}"] = mapped_post[:, k]
    return out


def train_percentile(train_score: np.ndarray, test_score: np.ndarray):
    ordered = np.sort(train_score)
    return np.searchsorted(ordered, test_score, side="right") / len(ordered)


def upper_pool(d2: np.ndarray, assigned: np.ndarray, policy: str):
    chosen = np.zeros(len(d2), dtype=bool)
    thresholds = {}
    if policy == "global":
        t = float(np.quantile(d2, 0.9))
        chosen = d2 >= t
        thresholds["all"] = t
    elif policy == "component":
        for k in np.unique(assigned):
            part = assigned == k
            if part.sum() < 20:
                thresholds[str(k)] = None
                continue
            t = float(np.quantile(d2[part], 0.9))
            chosen[part] = d2[part] >= t
            thresholds[str(k)] = t
    else:
        raise ValueError(policy)
    return chosen, thresholds


def apply_pool(d2: np.ndarray, assigned: np.ndarray, policy: str, thresholds: dict):
    if policy == "global":
        return d2 >= thresholds["all"]
    chosen = np.zeros(len(d2), dtype=bool)
    for k, value in thresholds.items():
        if value is None:
            continue
        part = assigned == int(k)
        chosen[part] = d2[part] >= value
    return chosen


def rows_to_daily(frame: pd.DataFrame, fields: list[str]):
    table = frame.groupby("date").size().rename("count").to_frame()
    for field in fields:
        agg = frame.groupby("date")[field].agg(
            mean="mean", median="median", std="std", max="max",
            p75=lambda x: x.quantile(0.75), p90=lambda x: x.quantile(0.90),
            p95=lambda x: x.quantile(0.95), p99=lambda x: x.quantile(0.99),
            iqr=lambda x: x.quantile(0.75) - x.quantile(0.25))
        agg.columns = [field + "_" + c for c in agg.columns]
        table = table.join(agg)
    return table.reset_index()


def write_csv(df: pd.DataFrame, name: str):
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / name, index=False, float_format="%.12g")

