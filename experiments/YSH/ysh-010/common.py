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
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "outputs"
FIG = OUT / "figures"
MODELS = OUT / "models"
SOURCE_PATH = ROOT / "data" / "Data_RSW.csv"
KAMP_PATH = ROOT / "data" / "Welding_Data_Set_01.xlsx"
Y7 = HERE.parent / "ysh-007" / "outputs"
Y8 = HERE.parent / "ysh-008" / "outputs"

KAMP_FEATURES = [
    "weld force(bar)",
    "weld current(kA)",
    "weld Voltage(v)",
    "weld time(ms)",
]
SOURCE_COLUMNS = [
    "Sample ID",
    "Pressure (PSI)",
    "Welding Time (ms)",
    "Angle (Deg)",
    "Force (N)",
    "Current (A)",
    "Thickness A (mm)",
    "Thickness B (mm)",
    "Material",
    "PullTest (N)",
    "NuggetDiameter (mm)",
    "Category",
    "Comments",
]


def read_config() -> dict:
    return json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_output_dirs() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)


def write_csv(frame: pd.DataFrame, name: str) -> None:
    ensure_output_dirs()
    frame.to_csv(OUT / name, index=False, float_format="%.12g", encoding="utf-8-sig")


def write_json(payload: dict, name: str) -> None:
    ensure_output_dirs()
    (OUT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def load_source_raw(config: dict) -> pd.DataFrame:
    digest = sha256(SOURCE_PATH)
    if digest != config["source_sha256"]:
        raise ValueError(f"Source SHA-256 mismatch: {digest}")
    frame = pd.read_csv(SOURCE_PATH, encoding="utf-8-sig")
    if list(frame.columns) != SOURCE_COLUMNS:
        raise ValueError(f"Unexpected source columns: {list(frame.columns)}")
    if len(frame) != config["source_rows"] or frame["Sample ID"].nunique() != config["source_samples"]:
        raise ValueError("Unexpected source row/sample count")
    return frame


def source_consistency(raw: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "Pressure (PSI)",
        "Welding Time (ms)",
        "Angle (Deg)",
        "Thickness A (mm)",
        "Thickness B (mm)",
        "Material",
        "PullTest (N)",
        "NuggetDiameter (mm)",
        "Category",
    ]
    grouped = raw.groupby("Sample ID", sort=True)
    audit = grouped.size().rename("raw_rows").to_frame()
    for field in fields:
        audit[f"{field}_nunique"] = grouped[field].nunique(dropna=False)
        audit[f"{field}_first"] = grouped[field].first()
        if pd.api.types.is_numeric_dtype(raw[field]):
            audit[f"{field}_min"] = grouped[field].min()
            audit[f"{field}_max"] = grouped[field].max()
            audit[f"{field}_median"] = grouped[field].median()
    audit["communication_error_rows"] = grouped.apply(
        lambda part: int(
            part["Comments"].fillna("").astype(str).str.contains("Communication error", case=False).sum()
        ),
        include_groups=False,
    )
    audit["has_any_metadata_inconsistency"] = audit.filter(like="_nunique").gt(1).any(axis=1)
    return audit.reset_index()


def aggregate_source(raw: pd.DataFrame) -> pd.DataFrame:
    work = raw.copy()
    for column in ["Force (N)", "Current (A)"]:
        work[f"{column}_communication_error"] = work[column].eq(-99)
        work.loc[work[column].eq(-99), column] = np.nan

    grouped = work.groupby("Sample ID", sort=True)
    sample = grouped.size().rename("raw_rows").to_frame()
    for column, prefix in [("Current (A)", "current"), ("Force (N)", "force")]:
        sample[f"{prefix}_valid_rows"] = grouped[column].count()
        sample[f"{prefix}_invalid_rows"] = grouped[f"{column}_communication_error"].sum()
        for statistic in ["min", "mean", "max", "median", "std"]:
            sample[f"{prefix}_{statistic}"] = getattr(grouped[column], statistic)()

    first_fields = {
        "Pressure (PSI)": "pressure_psi",
        "Welding Time (ms)": "welding_time_ms",
        "Angle (Deg)": "angle_deg",
        "Material": "material",
        "Category": "category",
        "NuggetDiameter (mm)": "nugget_mm",
    }
    for source, target in first_fields.items():
        sample[target] = grouped[source].first()
    sample["thickness_a_mm"] = grouped["Thickness A (mm)"].median()
    sample["thickness_b_mm"] = grouped["Thickness B (mm)"].median()
    sample["pulltest_n"] = grouped["PullTest (N)"].median()
    sample["pressure_bar"] = sample["pressure_psi"] * 0.0689475729
    for statistic in ["min", "mean", "max", "median", "std"]:
        sample[f"current_{statistic}_ka"] = sample[f"current_{statistic}"] / 1000.0
    sample["abnormal"] = sample["category"].isin(["Bad", "Explode"]).astype(int)
    sample["condition_group"] = (
        sample["pressure_psi"].astype(str)
        + "|"
        + sample["welding_time_ms"].astype(str)
        + "|"
        + sample["angle_deg"].astype(str)
    )
    return sample.reset_index().rename(columns={"Sample ID": "sample_id"})


def load_kamp(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    digest = sha256(KAMP_PATH)
    if digest != config["kamp_sha256"]:
        raise ValueError(f"KAMP SHA-256 mismatch: {digest}")
    raw = pd.read_excel(KAMP_PATH, sheet_name="Raw data")
    result = pd.read_excel(KAMP_PATH, sheet_name="result")
    if len(raw) != config["kamp_rows"] or set(KAMP_FEATURES) - set(raw.columns):
        raise ValueError("Unexpected KAMP Raw structure")
    raw = raw.copy()
    raw["excel_row"] = np.arange(2, len(raw) + 2)
    raw["date"] = pd.to_datetime(raw["working time"]).dt.strftime("%Y-%m-%d")
    raw["F"] = raw["weld force(bar)"].astype(float)
    raw["I"] = raw["weld current(kA)"].astype(float)
    raw["V"] = raw["weld Voltage(v)"].astype(float)
    raw["t"] = raw["weld time(ms)"].astype(float)
    if raw[["F", "I", "V", "t"]].isna().any().any():
        raise ValueError("Unexpected KAMP Raw4 NA")
    result = result.copy()
    result["date"] = pd.to_datetime(result["working time"]).dt.strftime("%Y-%m-%d")
    quality = result.pivot_table(
        index="date", columns="defect type", values="defect", aggfunc="first"
    ).rename(columns={1: "type1", 2: "type2", 3: "type3"})
    for column in ["type1", "type2", "type3"]:
        if column not in quality:
            quality[column] = np.nan
    quality = quality[["type1", "type2", "type3"]].reset_index()
    return raw, quality


def robust_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values, axis=0)
    q25 = np.nanquantile(values, 0.25, axis=0)
    q75 = np.nanquantile(values, 0.75, axis=0)
    iqr = q75 - q25
    return median, iqr


def robust_transform(values: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    if np.any(~np.isfinite(iqr)) or np.any(iqr <= 0):
        raise ValueError(f"Non-positive IQR: {iqr}")
    return (np.asarray(values, dtype=float) - median) / iqr


def quantile_transform(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values, dtype=float)
    for column in range(reference.shape[1]):
        ref = np.sort(reference[:, column])
        left = np.searchsorted(ref, values[:, column], side="left")
        right = np.searchsorted(ref, values[:, column], side="right")
        out[:, column] = (left + 0.5 * (right - left) + 0.5) / (len(ref) + 1.0)
    return out


def source_transform_fit(
    x_train: np.ndarray, y_train: np.ndarray, representation: str
) -> dict:
    good = x_train[np.asarray(y_train) == 0]
    if not len(good):
        raise ValueError("No Good samples in training fold")
    if representation == "RZ":
        median, iqr = robust_fit(good)
        if np.any(iqr <= 0):
            raise ValueError(f"Good IQR is zero: {iqr}")
        return {"representation": representation, "median": median, "iqr": iqr}
    if representation == "Q":
        return {"representation": representation, "reference": good.copy()}
    raise ValueError(representation)


def source_transform_apply(values: np.ndarray, fitted: dict) -> np.ndarray:
    if fitted["representation"] == "RZ":
        return robust_transform(values, fitted["median"], fitted["iqr"])
    return quantile_transform(fitted["reference"], values)


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.0) -> dict:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "brier": brier_score_loss(y, expit(score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def optimal_bacc_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, dict]:
    unique = np.unique(np.asarray(score, dtype=float))
    if len(unique) == 1:
        candidates = unique
    else:
        candidates = np.r_[
            np.nextafter(unique[0], -np.inf),
            (unique[:-1] + unique[1:]) / 2,
            np.nextafter(unique[-1], np.inf),
        ]
    rows = []
    for threshold in candidates:
        metrics = classification_metrics(y, score, float(threshold))
        rows.append((metrics["balanced_accuracy"], metrics["specificity"], -abs(threshold), threshold, metrics))
    best = max(rows, key=lambda item: (item[0], item[1], item[2]))
    return float(best[3]), best[4]


def regression_metrics(y: np.ndarray, prediction: np.ndarray, dummy: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    dummy = np.asarray(dummy, dtype=float)
    rho = spearmanr(y, prediction).statistic
    mae = mean_absolute_error(y, prediction)
    dummy_mae = mean_absolute_error(y, dummy)
    return {
        "mae": mae,
        "rmse": mean_squared_error(y, prediction) ** 0.5,
        "r2": r2_score(y, prediction),
        "spearman": rho,
        "dummy_mae": dummy_mae,
        "dummy_mae_improvement": (dummy_mae - mae) / dummy_mae if dummy_mae else np.nan,
    }


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return rankdata(values, method="average") / (len(values) + 1.0)


def episode_ids(frame: pd.DataFrame, flag_column: str) -> pd.Series:
    flag = frame[flag_column].fillna(False).astype(bool).to_numpy()
    dates = frame["date"].astype(str).to_numpy()
    rows = frame["excel_row"].to_numpy()
    segments = frame["safe_segment"].astype(str).to_numpy() if "safe_segment" in frame else np.repeat("", len(frame))
    ids = np.zeros(len(frame), dtype=int)
    current = 0
    previous = -2
    for index in np.flatnonzero(flag):
        if (
            index == 0
            or rows[index] != previous + 1
            or dates[index] != dates[index - 1]
            or segments[index] != segments[index - 1]
            or not flag[index - 1]
        ):
            current += 1
        ids[index] = current
        previous = rows[index]
    return pd.Series(ids, index=frame.index, dtype=int)
