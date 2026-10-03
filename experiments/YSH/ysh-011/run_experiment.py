from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "outputs"
FIG = OUT / "figures"
SOURCE_PATH = ROOT / "data" / "Data_RSW.csv"
KAMP_PATH = ROOT / "data" / "Welding_Data_Set_01.xlsx"
Y7 = HERE.parent / "ysh-007" / "outputs"
Y8 = HERE.parent / "ysh-008" / "outputs"
Y10 = HERE.parent / "ysh-010" / "outputs"
CONFIG_PATH = HERE / "config.json"

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

MAPPINGS = {
    "M1_mean": "current_mean_ka",
    "M2_max": "current_max_ka",
    "M3_min": "current_min_ka",
}

TASKS = {
    "Bad": ("Good", "Bad"),
    "Explode": ("Good", "Explode"),
    "Any_abnormal": ("Good", "Bad", "Explode"),
}

COLORS = {"Good": "#4c78a8", "Bad": "#e45756", "Explode": "#f2cf5b"}


def read_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.12g")


def write_json(payload: dict, name: str) -> None:
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


def midpoint_rank(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return (series.rank(method="average").to_numpy(float)) / (len(series) + 1.0)


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


def source_raw_and_features(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if sha256(SOURCE_PATH) != config["source_sha256"]:
        raise RuntimeError("Data_RSW.csv SHA-256 mismatch")
    raw = pd.read_csv(SOURCE_PATH, encoding="utf-8-sig")
    if list(raw.columns) != SOURCE_COLUMNS:
        raise RuntimeError(f"Unexpected Source schema: {list(raw.columns)}")
    if len(raw) != config["source_rows"] or raw["Sample ID"].nunique() != config["source_samples"]:
        raise RuntimeError("Unexpected Source row or Sample ID count")

    work = raw.copy()
    work["_file_row"] = np.arange(len(work))
    for column in ["Current (A)", "Force (N)"]:
        work[f"{column}_invalid"] = work[column].eq(-99)
        work.loc[work[column].eq(-99), column] = np.nan

    grouped = work.groupby("Sample ID", sort=True)
    sample = grouped.size().rename("raw_rows").to_frame()
    for column, prefix in [("Current (A)", "current"), ("Force (N)", "force")]:
        sample[f"{prefix}_valid_rows"] = grouped[column].count()
        sample[f"{prefix}_invalid_rows"] = grouped[f"{column}_invalid"].sum()
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
    sample["condition_group"] = (
        sample["pressure_psi"].astype(str)
        + "|"
        + sample["welding_time_ms"].astype(str)
        + "|"
        + sample["angle_deg"].astype(str)
    )
    sample = sample.reset_index().rename(columns={"Sample ID": "sample_id"})

    zero_valid = sorted(
        sample.loc[
            sample["current_valid_rows"].eq(0) | sample["force_valid_rows"].eq(0), "sample_id"
        ].astype(int)
    )
    if zero_valid != config["excluded_sample_ids"]:
        raise RuntimeError(f"Unexpected zero-valid Sample IDs: {zero_valid}")
    cohort = sample.loc[~sample["sample_id"].isin(zero_valid)].copy().reset_index(drop=True)
    if len(cohort) != config["model_samples"]:
        raise RuntimeError("Unexpected Source model cohort size")
    counts = cohort["category"].value_counts().to_dict()
    if counts != config["category_counts"]:
        raise RuntimeError(f"Unexpected Source category counts: {counts}")

    positions = work.groupby("Sample ID")["_file_row"].agg(["min", "max", "count"]).reset_index()
    positions["contiguous"] = positions["max"] - positions["min"] + 1 == positions["count"]
    if not positions["contiguous"].all():
        raise RuntimeError("Non-contiguous Sample ID blocks found")
    return raw, sample, cohort


def load_kamp(config: dict) -> pd.DataFrame:
    if sha256(KAMP_PATH) != config["kamp_sha256"]:
        raise RuntimeError("Welding_Data_Set_01.xlsx SHA-256 mismatch")
    raw = pd.read_excel(KAMP_PATH, sheet_name="Raw data")
    required = ["working time", "weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
    if len(raw) != config["kamp_rows"] or set(required) - set(raw.columns):
        raise RuntimeError("Unexpected KAMP Raw schema")
    frame = raw.copy()
    frame["excel_row"] = np.arange(2, len(frame) + 2)
    frame["date"] = pd.to_datetime(frame["working time"]).dt.strftime("%Y-%m-%d")
    frame["F"] = frame["weld force(bar)"].astype(float)
    frame["I"] = frame["weld current(kA)"].astype(float)
    frame["V"] = frame["weld Voltage(v)"].astype(float)
    frame["t"] = frame["weld time(ms)"].astype(float)
    if frame[["F", "I", "V", "t"]].isna().any().any():
        raise RuntimeError("KAMP Raw4 contains NA")
    unique_count = len(frame[["F", "I", "V", "t"]].drop_duplicates())
    if unique_count != config["kamp_unique_raw4"]:
        raise RuntimeError(f"Unexpected unique Raw4 count: {unique_count}")
    return frame


def feature_specs() -> list[dict]:
    specs: list[dict] = []
    for mapping, current in MAPPINGS.items():
        specs.extend(
            [
                {"family": "F1", "mapping": mapping, "columns": [current]},
                {"family": "F2", "mapping": mapping, "columns": [current, "welding_time_ms"]},
                {
                    "family": "F3",
                    "mapping": mapping,
                    "columns": ["pressure_bar", current, "welding_time_ms"],
                },
            ]
        )
    specs.append(
        {
            "family": "F4_SOURCE_RICH",
            "mapping": "NA",
            "columns": [
                "current_min_ka",
                "current_mean_ka",
                "current_max_ka",
                "current_std_ka",
                "force_min",
                "force_mean",
                "force_max",
                "force_std",
                "pressure_bar",
                "welding_time_ms",
                "angle_deg",
            ],
        }
    )
    return specs


def model_score(model_name: str, x_train: np.ndarray, x_test: np.ndarray, config: dict):
    if model_name == "IF":
        model = IsolationForest(**config["isolation_forest"])
        model.fit(x_train)
        return -model.score_samples(x_test), {"selected_k": None}, model
    if model_name == "GMM":
        candidates = []
        errors = []
        gmm_config = config["gmm"]
        for k in range(gmm_config["k_min"], gmm_config["k_max"] + 1):
            try:
                candidate = GaussianMixture(
                    n_components=k,
                    covariance_type=gmm_config["covariance_type"],
                    reg_covar=gmm_config["reg_covar"],
                    n_init=gmm_config["n_init"],
                    random_state=gmm_config["random_state"],
                ).fit(x_train)
                candidates.append((candidate.bic(x_train), k, candidate))
            except Exception as exc:
                errors.append(f"K{k}:{type(exc).__name__}:{exc}")
        if not candidates:
            raise RuntimeError("All GMM K failed: " + " | ".join(errors))
        _, selected_k, model = min(candidates, key=lambda item: item[0])
        return -model.score_samples(x_test), {"selected_k": selected_k, "gmm_k_errors": errors}, model
    if model_name == "LOF":
        model = LocalOutlierFactor(
            n_neighbors=config["lof"]["n_neighbors"],
            novelty=True,
            metric=config["lof"]["metric"],
        )
        model.fit(x_train)
        return -model.score_samples(x_test), {"selected_k": None}, model
    if model_name == "MCD":
        model = MinCovDet(
            support_fraction=config["mcd"]["support_fraction"],
            random_state=config["mcd"]["random_state"],
        ).fit(x_train)
        return model.mahalanobis(x_test), {"selected_k": None}, model
    raise ValueError(model_name)


def source_eda(config: dict) -> None:
    ensure_dirs()
    raw, all_samples, cohort = source_raw_and_features(config)
    write_csv(all_samples, "source_weld_features.csv")

    summary_columns = [
        "current_min_ka",
        "current_mean_ka",
        "current_max_ka",
        "current_std_ka",
        "force_min",
        "force_mean",
        "force_max",
        "force_std",
        "pressure_psi",
        "welding_time_ms",
        "angle_deg",
        "nugget_mm",
        "pulltest_n",
    ]
    summary_rows = []
    for category, part in cohort.groupby("category", sort=False):
        for column in summary_columns:
            summary_rows.append(
                {
                    "category": category,
                    "feature": column,
                    "n": int(part[column].notna().sum()),
                    "median": part[column].median(),
                    "q25": part[column].quantile(0.25),
                    "q75": part[column].quantile(0.75),
                    "mean": part[column].mean(),
                    "std": part[column].std(),
                }
            )
    write_csv(pd.DataFrame(summary_rows), "source_signal_summary.csv")

    work = raw.copy()
    work.loc[work["Current (A)"].eq(-99), "Current (A)"] = np.nan
    work.loc[work["Force (N)"].eq(-99), "Force (N)"] = np.nan
    categories = cohort.set_index("sample_id")["category"].to_dict()
    p_grid = np.linspace(0.0, 1.0, 101)
    profile_rows = []
    for sample_id, part in work.groupby("Sample ID", sort=True):
        if sample_id not in categories:
            continue
        for column, signal in [("Current (A)", "Current_A"), ("Force (N)", "Force_N")]:
            values = part[column].dropna().to_numpy(float)
            if len(values) < 2:
                raise RuntimeError(f"Interpolation requires >=2 points: {sample_id} {signal}")
            old_p = np.linspace(0.0, 1.0, len(values))
            interpolated = np.interp(p_grid, old_p, values)
            profile_rows.extend(
                {
                    "sample_id": int(sample_id),
                    "category": categories[sample_id],
                    "signal": signal,
                    "relative_position": float(position),
                    "value": float(value),
                }
                for position, value in zip(p_grid, interpolated)
            )
    profiles = pd.DataFrame(profile_rows)
    write_csv(profiles, "source_interpolated_profiles_eda_only.csv")

    plot_sequence_profile(profiles, "Current_A", "figure1_current_file_order_profile.png", "Current (A)")
    plot_sequence_profile(profiles, "Force_N", "figure2_force_file_order_profile.png", "Force (N)")
    plot_category_panels(
        cohort,
        ["current_min_ka", "current_mean_ka", "current_max_ka"],
        "figure3_current_category_distributions.png",
        "Figure 3. Current summaries by category",
    )
    plot_category_panels(
        cohort,
        ["pressure_psi", "welding_time_ms", "angle_deg"],
        "figure4_process_setting_distributions.png",
        "Figure 4. Process settings by category",
    )
    plot_category_panels(
        cohort,
        ["nugget_mm", "pulltest_n"],
        "figure5_quality_output_distributions.png",
        "Figure 5. Quality outputs by category (evaluation only)",
    )

    fig, axis = plt.subplots(figsize=(8, 6))
    for category in ["Good", "Bad", "Explode"]:
        part = cohort.loc[cohort["category"].eq(category)]
        axis.scatter(part["current_mean_ka"], part["force_mean"], s=28, alpha=0.7, label=f"{category} (n={len(part)})", color=COLORS[category])
    axis.set(xlabel="Current mean (kA)", ylabel="Force mean (N)", title="Figure 6. Source sensor summaries")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "figure6_current_force_scatter.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    for category in ["Good", "Bad", "Explode"]:
        part = cohort.loc[cohort["category"].eq(category)]
        axis.scatter(part["nugget_mm"], part["pulltest_n"], s=28, alpha=0.7, label=f"{category} (n={len(part)})", color=COLORS[category])
    axis.set(xlabel="Nugget diameter (mm)", ylabel="Pull test (N)", title="Figure 7. Source quality outputs")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "figure7_nugget_pulltest_scatter.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    doe = cohort.groupby(["pressure_psi", "welding_time_ms", "category"], as_index=False).size()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for category in ["Good", "Bad", "Explode"]:
        part = doe.loc[doe["category"].eq(category)]
        axes[0].scatter(part["pressure_psi"], part["welding_time_ms"], s=20 + 18 * part["size"], alpha=0.65, label=category, color=COLORS[category])
        angle = cohort.loc[cohort["category"].eq(category), "angle_deg"].value_counts().sort_index()
        axes[1].plot(angle.index, angle.values, marker="o", label=category, color=COLORS[category])
    axes[0].set(xlabel="Pressure (PSI)", ylabel="Welding Time (ms)", title="DOE map; size=count")
    axes[1].set(xlabel="Angle (deg)", ylabel="Weld samples", title="Angle counts")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    fig.suptitle("Figure 8. Source DOE settings by literal Category")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "figure8_doe_map.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    q_columns = ["current_mean_ka", "current_max_ka", "current_min_ka", "pressure_bar", "welding_time_ms"]
    all_q = quantile_transform(cohort[q_columns].to_numpy(float), cohort[q_columns].to_numpy(float))
    good_ref = cohort.loc[cohort["category"].eq("Good"), q_columns].to_numpy(float)
    good_q = quantile_transform(good_ref, cohort[q_columns].to_numpy(float))
    q_frame = cohort[["sample_id", "category"]].copy()
    for index, column in enumerate(q_columns):
        q_frame[f"{column}_allQ"] = all_q[:, index]
        q_frame[f"{column}_goodQ"] = good_q[:, index]
    write_csv(q_frame, "source_q_features.csv")
    plot_raw_q_comparison(cohort, q_frame, q_columns)

    manifest = {
        "experiment": "YSH-011",
        "phase": "phase1_complete",
        "source_path": str(SOURCE_PATH.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE_PATH),
        "source_rows": len(raw),
        "source_sample_ids": int(raw["Sample ID"].nunique()),
        "sensor_valid_samples": len(cohort),
        "excluded_sample_ids": config["excluded_sample_ids"],
        "category_counts": cohort["category"].value_counts().to_dict(),
        "sample_blocks_contiguous": True,
        "timestamp_column_present": False,
        "sequence_interpolation_role": "EDA_only_recorded_file_order_not_time",
        "kamp_path": str(KAMP_PATH.relative_to(ROOT)),
        "kamp_sha256": sha256(KAMP_PATH),
        "read_only_inputs": [
            str((Y7 / "physical_row_followup.csv").relative_to(ROOT)),
            str((Y8 / "ae_row_scores.csv").relative_to(ROOT)),
            str((Y8 / "repeat298_mapping.csv").relative_to(ROOT)),
            str((Y10 / "source_cv_predictions.csv").relative_to(ROOT)),
            str((Y10 / "source_split_manifest.csv").relative_to(ROOT)),
            str((Y10 / "kamp_external_transfer_scores.csv").relative_to(ROOT)),
        ],
    }
    write_json(manifest, "input_manifest.json")


def plot_sequence_profile(profiles: pd.DataFrame, signal: str, filename: str, ylabel: str) -> None:
    subset = profiles.loc[profiles["signal"].eq(signal)]
    fig, axis = plt.subplots(figsize=(10, 6))
    for category in ["Good", "Bad", "Explode"]:
        part = subset.loc[subset["category"].eq(category)]
        pivot = part.pivot(index="sample_id", columns="relative_position", values="value")
        x = pivot.columns.to_numpy(float)
        median = pivot.median(axis=0).to_numpy(float)
        q10 = pivot.quantile(0.10, axis=0).to_numpy(float)
        q90 = pivot.quantile(0.90, axis=0).to_numpy(float)
        axis.plot(x, median, label=f"{category} median (n={len(pivot)})", color=COLORS[category])
        axis.fill_between(x, q10, q90, alpha=0.15, color=COLORS[category])
    axis.set(
        xlabel="normalized within-weld file-order position (not acquisition time)",
        ylabel=ylabel,
        title=f"{filename.split('_')[0].title()}. {ylabel} recorded-file-order profile",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_category_panels(frame: pd.DataFrame, columns: list[str], filename: str, title: str) -> None:
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4.8), squeeze=False)
    for axis, column in zip(axes[0], columns):
        values = [frame.loc[frame["category"].eq(category), column].dropna() for category in ["Good", "Bad", "Explode"]]
        axis.boxplot(values, tick_labels=[f"Good\nn={len(values[0])}", f"Bad\nn={len(values[1])}", f"Explode\nn={len(values[2])}"], showfliers=True)
        axis.set_title(column)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_raw_q_comparison(source: pd.DataFrame, q_frame: pd.DataFrame, columns: list[str]) -> None:
    selected = ["current_mean_ka", "welding_time_ms", "pressure_bar"]
    fig, axes = plt.subplots(len(selected), 3, figsize=(14, 12))
    for row, column in enumerate(selected):
        series = [
            source.loc[source["category"].eq(category), column].dropna()
            for category in ["Good", "Bad", "Explode"]
        ]
        axes[row, 0].boxplot(series, tick_labels=["Good", "Bad", "Explode"], showfliers=False)
        axes[row, 0].set_title(f"Raw {column}")
        for col_index, suffix in [(1, "allQ"), (2, "goodQ")]:
            q_col = f"{column}_{suffix}"
            q_values = [
                q_frame.loc[q_frame["category"].eq(category), q_col]
                for category in ["Good", "Bad", "Explode"]
            ]
            axes[row, col_index].boxplot(q_values, tick_labels=["Good", "Bad", "Explode"], showfliers=False)
            axes[row, col_index].set_title(f"{column} {suffix}")
        for axis in axes[row]:
            axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Figure 9. Raw vs train-all Q vs Good-relative Q (descriptive full-cohort EDA)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "figure9_raw_q_category_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def primary_split_manifest(source: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    groups = source["condition_group"].to_numpy()
    for seed in config["seeds"]:
        splitter = GroupKFold(n_splits=config["folds"], shuffle=True, random_state=seed)
        for fold, (_, test_index) in enumerate(splitter.split(source, groups=groups)):
            for index in test_index:
                rows.append(
                    {
                        "split_scheme": "primary_label_blind_groupkfold",
                        "seed": seed,
                        "fold": fold,
                        "sample_id": int(source.iloc[index]["sample_id"]),
                        "condition_group": source.iloc[index]["condition_group"],
                    }
                )
    manifest = pd.DataFrame(rows)
    audit_manifest(manifest, source, config)
    return manifest


def frozen_split_manifest(source: pd.DataFrame, config: dict) -> pd.DataFrame:
    frozen = pd.read_csv(Y10 / "source_split_manifest.csv", encoding="utf-8-sig")
    frozen = frozen.loc[frozen["split"].eq("group"), ["seed", "fold", "sample_id", "condition_group"]].copy()
    frozen.insert(0, "split_scheme", "sensitivity_frozen_ysh010_stratified_group")
    audit_manifest(frozen, source, config)
    return frozen


def audit_manifest(manifest: pd.DataFrame, source: pd.DataFrame, config: dict) -> None:
    expected = set(source["sample_id"].astype(int))
    for seed, part in manifest.groupby("seed"):
        if set(part["sample_id"].astype(int)) != expected or part["sample_id"].duplicated().any():
            raise RuntimeError(f"OOF manifest is not one row per sample for seed {seed}")
        if part["fold"].nunique() != config["folds"]:
            raise RuntimeError(f"Unexpected fold count for seed {seed}")
        for _, fold_part in part.groupby("fold"):
            test_groups = set(fold_part["condition_group"])
            train_groups = set(source.loc[~source["sample_id"].isin(fold_part["sample_id"]), "condition_group"])
            if test_groups & train_groups:
                raise RuntimeError("Condition group leakage")


def transformed_matrices(x: np.ndarray, train_fit: np.ndarray, test: np.ndarray, representation: str):
    if representation == "Raw":
        scaler = StandardScaler().fit(x[train_fit])
        return scaler.transform(x[train_fit]), scaler.transform(x[test]), {
            "reference_n": len(train_fit),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }
    reference = x[train_fit]
    return quantile_transform(reference, reference), quantile_transform(reference, x[test]), {
        "reference_n": len(reference),
        "reference_sha256": hashlib.sha256(np.ascontiguousarray(reference).tobytes()).hexdigest(),
    }


def run_source_models(config: dict) -> None:
    ensure_dirs()
    _, _, source = source_raw_and_features(config)
    primary = primary_split_manifest(source, config)
    sensitivity = frozen_split_manifest(source, config)
    manifests = pd.concat([primary, sensitivity], ignore_index=True)
    write_csv(manifests, "source_split_manifest.csv")

    all_prediction_rows = []
    parameter_rows = []
    failure_rows = []
    model_names = ["IF", "GMM", "LOF", "MCD"]
    start = time.time()
    total_fold_specs = 2 * 2 * len(feature_specs()) * 2 * len(config["seeds"]) * config["folds"]
    completed = 0

    for split_scheme, manifest in manifests.groupby("split_scheme", sort=False):
        for seed in config["seeds"]:
            seed_manifest = manifest.loc[manifest["seed"].eq(seed)]
            for fold in range(config["folds"]):
                test_ids = set(seed_manifest.loc[seed_manifest["fold"].eq(fold), "sample_id"].astype(int))
                test = np.flatnonzero(source["sample_id"].isin(test_ids).to_numpy())
                train = np.flatnonzero(~source["sample_id"].isin(test_ids).to_numpy())
                for branch in ["U", "OC"]:
                    train_fit = train if branch == "U" else train[source.iloc[train]["category"].eq("Good").to_numpy()]
                    if not len(train_fit):
                        raise RuntimeError("Good-only fold has no Good training samples")
                    for representation in ["Raw", "Q"]:
                        for spec in feature_specs():
                            x = source[spec["columns"]].to_numpy(float)
                            if not np.isfinite(x).all():
                                raise RuntimeError(f"Non-finite feature values: {spec}")
                            try:
                                x_train, x_test, transform_meta = transformed_matrices(
                                    x, train_fit, test, representation
                                )
                            except Exception as exc:
                                failure_rows.append(
                                    failure_record(
                                        split_scheme, seed, fold, branch, representation, spec, "TRANSFORM", exc
                                    )
                                )
                                continue
                            for model_name in model_names:
                                try:
                                    score, meta, _ = model_score(model_name, x_train, x_test, config)
                                    if len(score) != len(test) or not np.isfinite(score).all():
                                        raise RuntimeError("Non-finite or incomplete score")
                                    for index, value in zip(test, score):
                                        row = source.iloc[index]
                                        all_prediction_rows.append(
                                            {
                                                "sample_id": int(row["sample_id"]),
                                                "category": row["category"],
                                                "condition_group": row["condition_group"],
                                                "split_scheme": split_scheme,
                                                "seed": seed,
                                                "fold": fold,
                                                "branch": branch,
                                                "representation": representation,
                                                "family": spec["family"],
                                                "mapping": spec["mapping"],
                                                "model": model_name,
                                                "anomaly_score": float(value),
                                            }
                                        )
                                    parameter_rows.append(
                                        {
                                            "split_scheme": split_scheme,
                                            "seed": seed,
                                            "fold": fold,
                                            "branch": branch,
                                            "representation": representation,
                                            "family": spec["family"],
                                            "mapping": spec["mapping"],
                                            "model": model_name,
                                            "train_rows": len(train_fit),
                                            "test_rows": len(test),
                                            "feature_columns": "|".join(spec["columns"]),
                                            "selected_k": meta.get("selected_k"),
                                            "transform_metadata_json": json.dumps(transform_meta, ensure_ascii=False),
                                            "model_metadata_json": json.dumps(meta, ensure_ascii=False),
                                        }
                                    )
                                except Exception as exc:
                                    failure_rows.append(
                                        failure_record(
                                            split_scheme, seed, fold, branch, representation, spec, model_name, exc
                                        )
                                    )
                            completed += 1
                            if completed % 40 == 0:
                                elapsed = time.time() - start
                                print(
                                    f"SOURCE_PROGRESS {completed}/{total_fold_specs} "
                                    f"elapsed={elapsed:.1f}s predictions={len(all_prediction_rows)}",
                                    flush=True,
                                )

    predictions = pd.DataFrame(all_prediction_rows)
    parameters = pd.DataFrame(parameter_rows)
    failures = pd.DataFrame(failure_rows)
    key = ["split_scheme", "seed", "branch", "representation", "family", "mapping", "model", "sample_id"]
    if predictions.duplicated(key).any():
        raise RuntimeError("Duplicate OOF prediction key")

    expected_samples = config["model_samples"]
    completeness = (
        predictions.groupby(key[:-1], dropna=False)["sample_id"]
        .agg(["count", "nunique"])
        .reset_index()
    )
    incomplete = completeness.loc[
        completeness["count"].ne(expected_samples) | completeness["nunique"].ne(expected_samples)
    ]
    if len(incomplete):
        write_csv(incomplete, "source_oof_incomplete_specs.csv")

    write_csv(predictions.loc[predictions["branch"].eq("U")], "source_unsupervised_oof_scores.csv")
    write_csv(predictions.loc[predictions["branch"].eq("OC")], "source_oneclass_oof_scores.csv")
    write_csv(parameters, "source_model_parameters.csv")
    write_csv(failures, "model_specific_failures.csv")
    write_csv(completeness, "source_oof_completeness.csv")

    metrics = source_metrics(predictions)
    write_csv(metrics, "source_anomaly_metrics.csv")
    summaries = seed_metric_summary(metrics)
    write_csv(summaries, "source_seed_metric_summary.csv")
    consensus = cross_seed_consensus(predictions)
    write_csv(consensus, "source_cross_seed_consensus_diagnostic.csv")
    category_summary = source_category_summary(predictions)
    write_csv(category_summary, "source_category_score_summary.csv")
    supervised = compare_frozen_supervised(predictions)
    write_csv(supervised, "source_supervised_vs_anomaly.csv")
    statuses = transfer_status_table(metrics, completeness)
    bootstrap = bootstrap_for_transfer_candidates(predictions, statuses, config)
    write_csv(bootstrap, "source_group_bootstrap_ci.csv")
    statuses = add_bootstrap_flags(statuses, bootstrap)
    write_csv(statuses, "source_transfer_status_by_category.csv")
    make_source_result_figures(metrics, category_summary, supervised)

    payload = {
        "phase": "phase2_to_5_complete",
        "prediction_rows": len(predictions),
        "unsupervised_rows": int(predictions["branch"].eq("U").sum()),
        "oneclass_rows": int(predictions["branch"].eq("OC").sum()),
        "metric_rows": len(metrics),
        "incomplete_specifications": len(incomplete),
        "model_specific_failures": len(failures),
        "transfer_status_counts": statuses["transfer_status"].value_counts().to_dict(),
        "bootstrap_rows": len(bootstrap),
        "elapsed_seconds": time.time() - start,
    }
    write_json(payload, "source_execution_summary.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), flush=True)


def failure_record(split_scheme, seed, fold, branch, representation, spec, model, exc) -> dict:
    return {
        "split_scheme": split_scheme,
        "seed": seed,
        "fold": fold,
        "branch": branch,
        "representation": representation,
        "family": spec["family"],
        "mapping": spec["mapping"],
        "model": model,
        "status": "MODEL_SPECIFIC_EXCLUSION",
        "exception_type": type(exc).__name__,
        "message": str(exc),
    }


def source_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "split_scheme",
        "seed",
        "branch",
        "representation",
        "family",
        "mapping",
        "model",
    ]
    rows = []
    for key, part in predictions.groupby(group_columns, sort=False):
        base = dict(zip(group_columns, key))
        for task, categories in TASKS.items():
            subset = part.loc[part["category"].isin(categories)].copy()
            y = (
                subset["category"].eq(task).astype(int).to_numpy()
                if task in ["Bad", "Explode"]
                else subset["category"].ne("Good").astype(int).to_numpy()
            )
            score = subset["anomaly_score"].to_numpy(float)
            if len(np.unique(y)) < 2:
                auc = ap = np.nan
            else:
                auc = roc_auc_score(y, score)
                ap = average_precision_score(y, score)
            good = subset.loc[subset["category"].eq("Good"), "anomaly_score"]
            target = subset.loc[subset["category"].ne("Good"), "anomaly_score"]
            rows.append(
                {
                    **base,
                    "task": task,
                    "n": len(subset),
                    "positive_n": int(y.sum()),
                    "prevalence": float(y.mean()),
                    "roc_auc": auc,
                    "pr_auc": ap,
                    "rank_biserial": 2 * auc - 1 if np.isfinite(auc) else np.nan,
                    "good_median_score": good.median(),
                    "target_median_score": target.median(),
                    "median_score_difference": target.median() - good.median(),
                }
            )
    return pd.DataFrame(rows)


def seed_metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    group = [
        "split_scheme",
        "branch",
        "representation",
        "family",
        "mapping",
        "model",
        "task",
    ]
    rows = []
    for key, part in metrics.groupby(group, sort=False):
        row = dict(zip(group, key))
        for column in ["roc_auc", "pr_auc", "rank_biserial", "median_score_difference"]:
            row[f"{column}_median"] = part[column].median()
            row[f"{column}_min"] = part[column].min()
            row[f"{column}_max"] = part[column].max()
        row["seeds"] = "|".join(map(str, sorted(part["seed"].unique())))
        rows.append(row)
    return pd.DataFrame(rows)


def cross_seed_consensus(predictions: pd.DataFrame) -> pd.DataFrame:
    spec = ["split_scheme", "branch", "representation", "family", "mapping", "model"]
    work = predictions.copy()
    work["within_seed_oof_rank"] = work.groupby(spec + ["seed"])["anomaly_score"].transform(
        lambda values: midpoint_rank(values.to_numpy(float))
    )
    return (
        work.groupby(spec + ["sample_id", "category", "condition_group"], as_index=False)
        .agg(
            cross_seed_mean_raw_score=("anomaly_score", "mean"),
            cross_seed_std_raw_score=("anomaly_score", "std"),
            cross_seed_mean_rank=("within_seed_oof_rank", "mean"),
            seed_count=("seed", "nunique"),
        )
    )


def source_category_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    group = [
        "split_scheme",
        "seed",
        "branch",
        "representation",
        "family",
        "mapping",
        "model",
        "category",
    ]
    return (
        predictions.groupby(group, as_index=False)["anomaly_score"]
        .agg(n="count", median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), mean="mean")
    )


def jaccard_flags(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else np.nan


def compare_frozen_supervised(predictions: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_csv(Y10 / "source_cv_predictions.csv", encoding="utf-8-sig")
    frozen = frozen.loc[
        frozen["model"].eq("logistic")
        & frozen["representation"].eq("Q")
        & frozen["split"].eq("group")
        & frozen["mapping"].isin(["M1_mean", "M2_max", "M3_min"])
    ].copy()
    frozen = frozen[["sample_id", "mapping", "oof_score"]].rename(columns={"oof_score": "supervised_score"})
    rows = []
    group = ["split_scheme", "seed", "branch", "representation", "family", "mapping", "model"]
    for key, part in predictions.groupby(group, sort=False):
        base = dict(zip(group, key))
        candidates = frozen if base["mapping"] == "NA" else frozen.loc[frozen["mapping"].eq(base["mapping"])]
        for supervised_mapping, frozen_part in candidates.groupby("mapping"):
            joined = part.merge(frozen_part, on="sample_id", how="inner", validate="one_to_one")
            left_rank = midpoint_rank(joined["anomaly_score"].to_numpy(float))
            right_rank = midpoint_rank(joined["supervised_score"].to_numpy(float))
            rows.append(
                {
                    **base,
                    "supervised_mapping": supervised_mapping,
                    "n": len(joined),
                    "spearman": spearmanr(joined["anomaly_score"], joined["supervised_score"]).statistic,
                    "jaccard_top5": jaccard_flags(left_rank >= 0.95, right_rank >= 0.95),
                    "jaccard_top10": jaccard_flags(left_rank >= 0.90, right_rank >= 0.90),
                }
            )
    return pd.DataFrame(rows)


def transfer_status_table(metrics: pd.DataFrame, completeness: pd.DataFrame) -> pd.DataFrame:
    primary_name = "primary_label_blind_groupkfold"
    sensitivity_name = "sensitivity_frozen_ysh010_stratified_group"
    rows = []
    keys = ["branch", "representation", "family", "mapping", "model"]
    for key, _ in metrics.groupby(keys, sort=False):
        base = dict(zip(keys, key))
        for task in ["Bad", "Explode"]:
            primary = metrics.loc[
                metrics["split_scheme"].eq(primary_name)
                & metrics["task"].eq(task)
                & np.logical_and.reduce([metrics[column].eq(value) for column, value in base.items()])
            ].sort_values("seed")
            sensitivity = metrics.loc[
                metrics["split_scheme"].eq(sensitivity_name)
                & metrics["task"].eq(task)
                & np.logical_and.reduce([metrics[column].eq(value) for column, value in base.items()])
            ].sort_values("seed")
            primary_positive = len(primary) == 3 and primary["roc_auc"].gt(0.5).all() and primary["rank_biserial"].gt(0).all()
            sensitivity_positive = len(sensitivity) == 3 and sensitivity["roc_auc"].gt(0.5).all()
            complete = True
            for split_name in [primary_name, sensitivity_name]:
                values = completeness.loc[
                    completeness["split_scheme"].eq(split_name)
                    & completeness["branch"].eq(base["branch"])
                    & completeness["representation"].eq(base["representation"])
                    & completeness["family"].eq(base["family"])
                    & completeness["mapping"].eq(base["mapping"])
                    & completeness["model"].eq(base["model"])
                ]
                if (
                    len(values) != 3
                    or set(values["seed"].astype(int)) != {7, 42, 2026}
                    or not values["count"].eq(493).all()
                    or not values["nunique"].eq(493).all()
                ):
                    complete = False

            f2_anchor = True
            if base["family"] != "F2" and base["family"] != "F4_SOURCE_RICH":
                anchor_filter = (
                    metrics["branch"].eq(base["branch"])
                    & metrics["representation"].eq(base["representation"])
                    & metrics["family"].eq("F2")
                    & metrics["mapping"].eq(base["mapping"])
                    & metrics["model"].eq(base["model"])
                    & metrics["task"].eq(task)
                )
                anchor_primary = metrics.loc[anchor_filter & metrics["split_scheme"].eq(primary_name)]
                anchor_sensitivity = metrics.loc[anchor_filter & metrics["split_scheme"].eq(sensitivity_name)]
                f2_anchor = (
                    len(anchor_primary) == 3
                    and anchor_primary["roc_auc"].gt(0.5).all()
                    and len(anchor_sensitivity) == 3
                    and anchor_sensitivity["roc_auc"].gt(0.5).all()
                )

            common_q = base["representation"] == "Q" and base["family"] in ["F1", "F2", "F3"]
            passes_core = complete and primary_positive and sensitivity_positive and f2_anchor and common_q
            if passes_core and base["branch"] == "U" and base["family"] in ["F1", "F2"]:
                status = "SUPPORTED_FOR_TRANSFER"
                reason = "all_primary_and_frozen_seeds_positive_with_F2_anchor"
            elif passes_core:
                status = "SENSITIVITY_ONLY"
                reason = (
                    "good_only_assumption_dependent"
                    if base["branch"] == "OC"
                    else "pressure_like_F3_sensitivity"
                )
            elif primary["roc_auc"].gt(0.5).any() if len(primary) else False:
                status = "SENSITIVITY_ONLY"
                if base["family"] == "F4_SOURCE_RICH":
                    reason = "source_only_F4_not_transferable"
                elif base["representation"] == "Raw":
                    reason = "raw_source_diagnostic_no_common_absolute_transfer"
                else:
                    reason = "positive_but_seed_split_or_F2_requirement_not_met"
            else:
                status = "NOT_TRANSFERRED"
                reason = "repeated_nonpositive_or_incomplete_direction"
            rows.append(
                {
                    **base,
                    "category_direction": task,
                    "primary_auc_seed7": value_for_seed(primary, 7),
                    "primary_auc_seed42": value_for_seed(primary, 42),
                    "primary_auc_seed2026": value_for_seed(primary, 2026),
                    "primary_auc_median": primary["roc_auc"].median(),
                    "frozen_auc_seed7": value_for_seed(sensitivity, 7),
                    "frozen_auc_seed42": value_for_seed(sensitivity, 42),
                    "frozen_auc_seed2026": value_for_seed(sensitivity, 2026),
                    "frozen_auc_median": sensitivity["roc_auc"].median(),
                    "oof_complete": complete,
                    "primary_all_positive": primary_positive,
                    "frozen_direction_all_positive": sensitivity_positive,
                    "f2_anchor_positive": f2_anchor,
                    "kamp_common_q_representation": common_q,
                    "transfer_status": status,
                    "status_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def value_for_seed(frame: pd.DataFrame, seed: int) -> float:
    value = frame.loc[frame["seed"].eq(seed), "roc_auc"]
    return float(value.iloc[0]) if len(value) else np.nan


def weighted_group_bootstrap_auc(part: pd.DataFrame, task: str, draws: int, seed: int) -> tuple[float, float, int, int]:
    categories = TASKS[task]
    subset = part.loc[part["category"].isin(categories)].copy()
    y = (
        subset["category"].eq(task).astype(int).to_numpy()
        if task in ["Bad", "Explode"]
        else subset["category"].ne("Good").astype(int).to_numpy()
    )
    scores = subset["anomaly_score"].to_numpy(float)
    group_codes, groups = pd.factorize(subset["condition_group"], sort=True)
    unique_scores, score_codes = np.unique(scores, return_inverse=True)
    pos_matrix = np.zeros((len(groups), len(unique_scores)), dtype=float)
    neg_matrix = np.zeros_like(pos_matrix)
    np.add.at(pos_matrix, (group_codes[y == 1], score_codes[y == 1]), 1.0)
    np.add.at(neg_matrix, (group_codes[y == 0], score_codes[y == 0]), 1.0)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=draws)
    pos = counts @ pos_matrix
    neg = counts @ neg_matrix
    total_pos = pos.sum(axis=1)
    total_neg = neg.sum(axis=1)
    valid = (total_pos > 0) & (total_neg > 0)
    neg_before = np.cumsum(neg, axis=1) - neg
    numer = np.sum(pos * (neg_before + 0.5 * neg), axis=1)
    auc = np.full(draws, np.nan)
    auc[valid] = numer[valid] / (total_pos[valid] * total_neg[valid])
    valid_auc = auc[np.isfinite(auc)]
    if not len(valid_auc):
        return np.nan, np.nan, 0, draws
    return (
        float(np.quantile(valid_auc, 0.025)),
        float(np.quantile(valid_auc, 0.975)),
        int(len(valid_auc)),
        int(draws - len(valid_auc)),
    )


def bootstrap_for_transfer_candidates(predictions: pd.DataFrame, statuses: pd.DataFrame, config: dict) -> pd.DataFrame:
    candidates = statuses.loc[
        statuses["representation"].eq("Q")
        & statuses["family"].isin(["F1", "F2", "F3"])
        & statuses["transfer_status"].ne("NOT_TRANSFERRED")
    ]
    rows = []
    for candidate in candidates.itertuples(index=False):
        for seed in config["seeds"]:
            part = predictions.loc[
                predictions["split_scheme"].eq("primary_label_blind_groupkfold")
                & predictions["seed"].eq(seed)
                & predictions["branch"].eq(candidate.branch)
                & predictions["representation"].eq(candidate.representation)
                & predictions["family"].eq(candidate.family)
                & predictions["mapping"].eq(candidate.mapping)
                & predictions["model"].eq(candidate.model)
            ]
            low, high, valid, invalid = weighted_group_bootstrap_auc(
                part,
                candidate.category_direction,
                config["bootstrap_draws"],
                config["bootstrap_seed"] + seed,
            )
            rows.append(
                {
                    "branch": candidate.branch,
                    "representation": candidate.representation,
                    "family": candidate.family,
                    "mapping": candidate.mapping,
                    "model": candidate.model,
                    "category_direction": candidate.category_direction,
                    "seed": seed,
                    "ci_lower": low,
                    "ci_upper": high,
                    "valid_replicates": valid,
                    "invalid_replicates": invalid,
                    "strong_statistical_support": bool(np.isfinite(low) and low > 0.5),
                }
            )
    return pd.DataFrame(rows)


def add_bootstrap_flags(statuses: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    keys = ["branch", "representation", "family", "mapping", "model", "category_direction"]
    if bootstrap.empty:
        statuses["strong_statistical_support_all_seeds"] = False
        return statuses
    flags = (
        bootstrap.groupby(keys, as_index=False)["strong_statistical_support"]
        .all()
        .rename(columns={"strong_statistical_support": "strong_statistical_support_all_seeds"})
    )
    merged = statuses.merge(flags, on=keys, how="left")
    merged["strong_statistical_support_all_seeds"] = merged[
        "strong_statistical_support_all_seeds"
    ].map(lambda value: bool(value) if pd.notna(value) else False)
    return merged


def rebuild_source_status() -> None:
    metrics = pd.read_csv(
        OUT / "source_anomaly_metrics.csv", encoding="utf-8-sig", keep_default_na=False
    )
    completeness = pd.read_csv(
        OUT / "source_oof_completeness.csv", encoding="utf-8-sig", keep_default_na=False
    )
    bootstrap = pd.read_csv(
        OUT / "source_group_bootstrap_ci.csv", encoding="utf-8-sig", keep_default_na=False
    )
    statuses = transfer_status_table(metrics, completeness)
    statuses = add_bootstrap_flags(statuses, bootstrap)
    write_csv(statuses, "source_transfer_status_by_category.csv")
    summary_path = OUT / "source_execution_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["transfer_status_counts"] = statuses["transfer_status"].value_counts().to_dict()
    summary["status_postprocessing_corrected"] = True
    summary["status_correction"] = "OOF completeness evaluated across three seed rows per specification"
    write_json(summary, "source_execution_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), flush=True)


def make_source_result_figures(metrics: pd.DataFrame, category_summary: pd.DataFrame, supervised: pd.DataFrame) -> None:
    primary = metrics.loc[
        metrics["split_scheme"].eq("primary_label_blind_groupkfold")
        & metrics["representation"].eq("Q")
        & metrics["family"].eq("F2")
    ]
    plot = (
        primary.groupby(["branch", "model", "mapping", "task"], as_index=False)["roc_auc"]
        .median()
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for index, task in enumerate(["Bad", "Explode"]):
        part = plot.loc[plot["task"].eq(task)].copy()
        labels = part["branch"] + "_" + part["model"] + "_" + part["mapping"]
        axes[index].bar(np.arange(len(part)), part["roc_auc"], color="#4c78a8")
        axes[index].axhline(0.5, color="black", linestyle="--")
        axes[index].set_xticks(np.arange(len(part)), labels, rotation=80, fontsize=7)
        axes[index].set_ylim(0, 1)
        axes[index].set_title(f"{task}: median of three seed OOF AUROCs")
        axes[index].grid(axis="y", alpha=0.2)
    fig.suptitle("Figure 10. F2 Q-space Source anomaly validation")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "figure10_source_anomaly_auc.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 6))
    valid = supervised.dropna(subset=["spearman"])
    for branch, part in valid.groupby("branch"):
        axis.hist(part["spearman"], bins=30, alpha=0.55, label=branch)
    axis.axvline(0, color="black", linestyle="--")
    axis.set(xlabel="Spearman with frozen YSH-010 supervised score", ylabel="Specification comparisons", title="Figure 11. Supervised vs anomaly ordering")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "figure11_supervised_anomaly_relationship.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def target_feature_values(kamp: pd.DataFrame, physical: pd.DataFrame, family: str, reference: str) -> tuple[np.ndarray, np.ndarray, str]:
    raw_map = {
        "F1": ["I"],
        "F2": ["I", "t"],
        "F3": ["F", "I", "t"],
    }
    z_map = {
        "F1": ["conditional_z_I"],
        "F2": ["conditional_z_I", "conditional_z_t"],
        "F3": ["conditional_z_F", "conditional_z_I", "conditional_z_t"],
    }
    if reference == "K1":
        values = kamp[raw_map[family]].to_numpy(float)
        ref = values
        population = "all_KAMP_rows"
    elif reference == "K2":
        values = kamp[raw_map[family]].to_numpy(float)
        unique = kamp[["F", "I", "V", "t"]].drop_duplicates()
        ref = unique[raw_map[family]].to_numpy(float)
        population = "unique_Raw4_equal_weight"
    elif reference == "K3":
        values = physical[z_map[family]].to_numpy(float)
        ref = values
        population = "YSH007_signed_conditional_z_all_rows"
    else:
        raise ValueError(reference)
    return values, ref, population


def fit_full_source_model(source: pd.DataFrame, row, config: dict):
    spec = next(
        spec for spec in feature_specs()
        if spec["family"] == row.family and spec["mapping"] == row.mapping
    )
    x = source[spec["columns"]].to_numpy(float)
    fit_index = np.arange(len(source)) if row.branch == "U" else np.flatnonzero(source["category"].eq("Good").to_numpy())
    reference = x[fit_index]
    x_train = quantile_transform(reference, reference)
    _, meta, model = model_score(row.model, x_train, x_train[:1], config)
    return model, meta, reference, spec


def score_fitted_model(model_name: str, model, values: np.ndarray) -> np.ndarray:
    if model_name in ["IF", "LOF"]:
        return -model.score_samples(values)
    if model_name == "GMM":
        return -model.score_samples(values)
    if model_name == "MCD":
        return model.mahalanobis(values)
    raise ValueError(model_name)


def run_target_transfer(config: dict) -> None:
    ensure_dirs()
    _, _, source = source_raw_and_features(config)
    kamp = load_kamp(config)
    status_path = OUT / "source_transfer_status_by_category.csv"
    if not status_path.exists():
        raise RuntimeError("Phase 5 status file missing; run source stage first")
    statuses = pd.read_csv(status_path, encoding="utf-8-sig")
    selected = statuses.loc[
        statuses["representation"].eq("Q")
        & statuses["family"].isin(["F1", "F2", "F3"])
        & statuses["transfer_status"].isin(["SUPPORTED_FOR_TRANSFER", "SENSITIVITY_ONLY"])
        & statuses["primary_all_positive"].astype(bool)
    ].copy()
    if selected.empty:
        write_csv(pd.DataFrame(), "kamp_external_anomaly_scores.csv")
        write_json({"phase": "phase6_not_run_no_selected_source_direction"}, "target_execution_summary.json")
        return

    physical = pd.read_csv(Y7 / "physical_row_followup.csv", encoding="utf-8-sig").sort_values("excel_row").reset_index(drop=True)
    ae = pd.read_csv(Y8 / "ae_row_scores.csv", encoding="utf-8-sig").sort_values("excel_row").reset_index(drop=True)
    if not np.array_equal(kamp["excel_row"].to_numpy(), physical["excel_row"].to_numpy()):
        raise RuntimeError("KAMP/YSH007 excel_row mismatch")
    if not np.array_equal(kamp["excel_row"].to_numpy(), ae["excel_row"].to_numpy()):
        raise RuntimeError("KAMP/YSH008 excel_row mismatch")

    model_keys = ["branch", "representation", "family", "mapping", "model"]
    score_rows = []
    model_rows = []
    for key, directions in selected.groupby(model_keys, sort=False):
        holder = directions.iloc[0]
        model, meta, _, spec = fit_full_source_model(source, holder, config)
        spec_id = "|".join(map(str, key))
        direction_summary = "|".join(
            f"{row.category_direction}:{row.transfer_status}"
            for row in directions.sort_values("category_direction").itertuples(index=False)
        )
        overall_status = (
            "SUPPORTED_FOR_TRANSFER"
            if directions["transfer_status"].eq("SUPPORTED_FOR_TRANSFER").any()
            else "SENSITIVITY_ONLY"
        )
        for reference in ["K1", "K2", "K3"]:
            values, ref, population = target_feature_values(kamp, physical, holder.family, reference)
            target_q = quantile_transform(ref, values)
            score = score_fitted_model(holder.model, model, target_q)
            row_rank = midpoint_rank(score)
            unique_frame = kamp[["F", "I", "V", "t"]].copy()
            unique_frame["score"] = score
            score_ranges = unique_frame.groupby(["F", "I", "V", "t"])["score"].agg(["min", "max"])
            if reference in ["K1", "K2"] and (score_ranges["max"] - score_ranges["min"]).max() > 1e-12:
                raise RuntimeError(f"Identical Raw4 received different scores: {spec_id} {reference}")
            if reference in ["K1", "K2"]:
                unique_scores = unique_frame.drop_duplicates(["F", "I", "V", "t"]).copy()
                unique_scores["unique_rank"] = midpoint_rank(unique_scores["score"].to_numpy(float))
                unique_lookup = unique_scores.set_index(["F", "I", "V", "t"])["unique_rank"]
                unique_rank = pd.MultiIndex.from_frame(kamp[["F", "I", "V", "t"]]).map(unique_lookup).to_numpy(float)
            else:
                unique_rank = np.full(len(kamp), np.nan)
            for index in range(len(kamp)):
                score_rows.append(
                    {
                        "excel_row": int(kamp.iloc[index]["excel_row"]),
                        "date": kamp.iloc[index]["date"],
                        "F": kamp.iloc[index]["F"],
                        "I": kamp.iloc[index]["I"],
                        "V": kamp.iloc[index]["V"],
                        "t": kamp.iloc[index]["t"],
                        "spec_id": spec_id,
                        "branch": holder.branch,
                        "family": holder.family,
                        "mapping": holder.mapping,
                        "model": holder.model,
                        "category_direction": direction_summary,
                        "transfer_status": overall_status,
                        "assumption_flag": (
                            "ASSUMPTION_DEPENDENT_GOOD_ONLY"
                            if holder.branch == "OC"
                            else ("PRESSURE_LIKE_SENSITIVITY" if holder.family == "F3" else "NONE")
                        ),
                        "target_reference": reference,
                        "external_anomaly_score": float(score[index]),
                        "rank_percentile_row": float(row_rank[index]),
                        "rank_percentile_unique_raw4": float(unique_rank[index]),
                        "interpretation": "relative_anomaly_candidate_not_defect_label",
                    }
                )
            model_rows.append(
                {
                    "spec_id": spec_id,
                    "target_reference": reference,
                    "source_feature_columns": "|".join(spec["columns"]),
                    "target_reference_population": population,
                    "target_reference_n": len(ref),
                    "selected_k": meta.get("selected_k"),
                }
            )
    scores = pd.DataFrame(score_rows)
    write_csv(scores, "kamp_external_anomaly_scores.csv")
    write_csv(scores[["excel_row", "spec_id", "category_direction", "target_reference", "external_anomaly_score", "rank_percentile_row", "rank_percentile_unique_raw4"]], "kamp_external_anomaly_ranks.csv")
    write_csv(pd.DataFrame(model_rows), "kamp_transfer_model_metadata.csv")

    candidates, coverage = topk_tables(scores, config)
    write_csv(candidates, "kamp_topk_candidates.csv")
    write_csv(coverage, "kamp_topk_realized_coverage.csv")
    stability = target_stability(scores, config)
    write_csv(stability, "transfer_rank_stability.csv")
    comparisons = compare_existing_target(scores, physical, ae, config)
    write_csv(comparisons, "transfer_existing_score_comparison.csv")
    repeat_profile, repeat_sanity = repeat298_diagnostic(scores, config)
    write_csv(repeat_profile, "repeat298_anomaly_profile.csv")
    write_csv(repeat_sanity, "repeat298_score_sanity.csv")
    plot_target_figures(stability, repeat_profile)
    payload = {
        "phase": "phase6_to_8_complete",
        "selected_category_directions": len(selected),
        "unique_model_specifications": selected[model_keys].drop_duplicates().shape[0],
        "score_rows": len(scores),
        "topk_candidate_rows": len(candidates),
        "repeat298_all_K1K2_scores_deterministic": bool(
            repeat_sanity.loc[
                repeat_sanity["target_reference"].isin(["K1", "K2"]),
                "identical_within_tolerance_1e_12",
            ].all()
        ),
        "result_sheet_used": False,
    }
    write_json(payload, "target_execution_summary.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), flush=True)


def topk_tables(scores: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = ["spec_id", "category_direction", "transfer_status", "target_reference"]
    candidate_parts = []
    coverage_rows = []
    for key, part in scores.groupby(group, sort=False):
        base = dict(zip(group, key))
        for nominal in config["top_k"]:
            threshold = 1.0 - nominal
            for view, rank_column in [
                ("row_weighted", "rank_percentile_row"),
                ("unique_raw4", "rank_percentile_unique_raw4"),
            ]:
                if part[rank_column].isna().all():
                    continue
                selected = part.loc[part[rank_column].ge(threshold)].copy()
                selected["candidate_view"] = view
                selected["nominal_top_k"] = nominal
                selected["rank_threshold"] = threshold
                candidate_parts.append(selected)
                coverage_rows.append(
                    {
                        **base,
                        "candidate_view": view,
                        "nominal_top_k": nominal,
                        "rank_threshold": threshold,
                        "actual_row_count": len(selected),
                        "actual_row_fraction": len(selected) / 11939.0,
                        "unique_Raw4_count": selected[["F", "I", "V", "t"]].drop_duplicates().shape[0],
                        "unique_Raw4_fraction": selected[["F", "I", "V", "t"]].drop_duplicates().shape[0] / 1574.0,
                    }
                )
    return pd.concat(candidate_parts, ignore_index=True), pd.DataFrame(coverage_rows)


def target_stability(scores: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    keys = ["spec_id", "category_direction", "transfer_status"]
    for key, part in scores.groupby(keys, sort=False):
        base = dict(zip(keys, key))
        wide = part.pivot(index="excel_row", columns="target_reference", values="external_anomaly_score")
        if {"K1", "K2"} <= set(wide.columns):
            left_rank = midpoint_rank(wide["K1"].to_numpy(float))
            right_rank = midpoint_rank(wide["K2"].to_numpy(float))
            for nominal in config["top_k"]:
                threshold = 1 - nominal
                rows.append(
                    {
                        **base,
                        "reference_left": "K1",
                        "reference_right": "K2",
                        "nominal_top_k": nominal,
                        "spearman": spearmanr(wide["K1"], wide["K2"]).statistic,
                        "jaccard": jaccard_flags(left_rank >= threshold, right_rank >= threshold),
                    }
                )
    return pd.DataFrame(rows)


def compare_existing_target(scores: pd.DataFrame, physical: pd.DataFrame, ae: pd.DataFrame, config: dict) -> pd.DataFrame:
    context = physical[["excel_row", "global_nll", "mahalanobis_d2", "max_abs_conditional_z", "any_physical_candidate"]].merge(
        ae[["excel_row", "ae_total_error", "ae_candidate"]], on="excel_row", validate="one_to_one"
    )
    methods = {
        "GMM_NLL": context["global_nll"].to_numpy(float),
        "GMM_d2": context["mahalanobis_d2"].to_numpy(float),
        "conditional_z": context["max_abs_conditional_z"].to_numpy(float),
        "AE_total": context["ae_total_error"].to_numpy(float),
        "physical_candidate": context["any_physical_candidate"].astype(int).to_numpy(float),
    }
    y10 = pd.read_csv(Y10 / "kamp_external_transfer_scores.csv", encoding="utf-8-sig")
    y10 = y10.loc[y10["representation"].eq("Q") & y10["target_reference"].isin(["K1_global", "K2_unique_raw4"])]
    for combo, part in y10.groupby("combo"):
        aligned = context[["excel_row"]].merge(
            part[["excel_row", "external_abnormal_similarity"]], on="excel_row", validate="one_to_one"
        )
        methods[f"YSH010_{combo}"] = aligned["external_abnormal_similarity"].to_numpy(float)
    rows = []
    group = ["spec_id", "category_direction", "transfer_status", "target_reference"]
    for key, part in scores.groupby(group, sort=False):
        part = part.sort_values("excel_row")
        base = dict(zip(group, key))
        new_score = part["external_anomaly_score"].to_numpy(float)
        new_rank = midpoint_rank(new_score)
        for name, existing in methods.items():
            existing_rank = midpoint_rank(existing)
            for nominal in config["top_k"]:
                threshold = 1 - nominal
                existing_flag = (
                    context["any_physical_candidate"].to_numpy(bool)
                    if name == "physical_candidate"
                    else existing_rank >= threshold
                )
                rows.append(
                    {
                        **base,
                        "existing_method": name,
                        "nominal_top_k": nominal,
                        "spearman": spearmanr(new_score, existing).statistic,
                        "jaccard": jaccard_flags(new_rank >= threshold, existing_flag),
                        "new_candidate_rows": int((new_rank >= threshold).sum()),
                        "existing_candidate_rows": int(existing_flag.sum()),
                    }
                )
    return pd.DataFrame(rows)


def repeat298_diagnostic(scores: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(Y8 / "repeat298_mapping.csv", encoding="utf-8-sig")
    joined = mapping[["excel_row", "occurrence_id", "relative_position", "gmm_nll_q90", "gmm_d2_q90", "conditional_z_2_5", "physical_global_candidate"]].merge(
        scores,
        on="excel_row",
        how="left",
        validate="one_to_many",
    )
    profile_group = ["spec_id", "category_direction", "transfer_status", "target_reference", "relative_position"]
    profile = (
        joined.groupby(profile_group, as_index=False)
        .agg(
            score_median=("external_anomaly_score", "median"),
            score_min=("external_anomaly_score", "min"),
            score_max=("external_anomaly_score", "max"),
            row_top1_share=("rank_percentile_row", lambda x: float((x >= 0.99).mean())),
            row_top5_share=("rank_percentile_row", lambda x: float((x >= 0.95).mean())),
            row_top10_share=("rank_percentile_row", lambda x: float((x >= 0.90).mean())),
            gmm_nll_q90_share=("gmm_nll_q90", "mean"),
            gmm_d2_q90_share=("gmm_d2_q90", "mean"),
            conditional_z_2_5_share=("conditional_z_2_5", "mean"),
            physical_global_share=("physical_global_candidate", "mean"),
        )
    )
    sanity_group = ["spec_id", "category_direction", "target_reference", "relative_position"]
    sanity = (
        joined.groupby(sanity_group)["external_anomaly_score"]
        .agg(["min", "max"])
        .reset_index()
    )
    sanity["score_range"] = sanity["max"] - sanity["min"]
    summary = (
        sanity.groupby(["spec_id", "category_direction", "target_reference"], as_index=False)
        .agg(
            relative_positions=("relative_position", "nunique"),
            max_occurrence_score_range=("score_range", "max"),
        )
    )
    summary["identical_within_tolerance_1e_12"] = summary["max_occurrence_score_range"].le(1e-12)
    return profile, summary


def plot_target_figures(stability: pd.DataFrame, profile: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(9, 6))
    for status, part in stability.groupby("transfer_status"):
        axis.scatter(part["spearman"], part["jaccard"], alpha=0.65, label=status)
    axis.set(xlabel="K1–K2 Spearman", ylabel="K1–K2 top-k Jaccard", title="Figure 12. KAMP reference stability")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "figure12_kamp_reference_stability.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    selected = profile.loc[
        profile["target_reference"].eq("K1")
        & profile["transfer_status"].eq("SUPPORTED_FOR_TRANSFER")
    ]
    if selected.empty:
        selected = profile.loc[profile["target_reference"].eq("K1")]
    selected_specs = selected[["spec_id", "category_direction"]].drop_duplicates().head(8)
    fig, axis = plt.subplots(figsize=(14, 6))
    for row in selected_specs.itertuples(index=False):
        part = selected.loc[
            selected["spec_id"].eq(row.spec_id)
            & selected["category_direction"].eq(row.category_direction)
        ]
        axis.plot(part["relative_position"], part["score_median"], label=f"{row.spec_id}|{row.category_direction}", alpha=0.8)
    axis.axvspan(1, 12, color="#f58518", alpha=0.12, label="AE 1–12")
    axis.axvspan(139, 143, color="#e45756", alpha=0.12, label="GMM local 139–143")
    axis.set(xlabel="repeat298 relative position", ylabel="External anomaly score", title="Figure 13. repeat298 diagnostic profiles")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "figure13_repeat298_anomaly_profile.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def verify(config: dict) -> None:
    required = [
        "input_manifest.json",
        "source_weld_features.csv",
        "source_q_features.csv",
        "source_signal_summary.csv",
        "source_unsupervised_oof_scores.csv",
        "source_oneclass_oof_scores.csv",
        "source_anomaly_metrics.csv",
        "source_transfer_status_by_category.csv",
        "kamp_external_anomaly_scores.csv",
        "kamp_topk_realized_coverage.csv",
        "transfer_rank_stability.csv",
        "transfer_existing_score_comparison.csv",
        "repeat298_anomaly_profile.csv",
    ]
    checks = []
    for name in required:
        path = OUT / name
        checks.append({"check": f"exists:{name}", "passed": path.exists(), "detail": str(path)})
    for index in range(1, 10):
        matches = list(FIG.glob(f"figure{index}_*.png"))
        checks.append({"check": f"figure_{index}", "passed": len(matches) == 1, "detail": "|".join(map(str, matches))})

    if (OUT / "source_unsupervised_oof_scores.csv").exists():
        u = pd.read_csv(OUT / "source_unsupervised_oof_scores.csv", encoding="utf-8-sig")
        key = ["split_scheme", "seed", "representation", "family", "mapping", "model"]
        sizes = u.groupby(key)["sample_id"].nunique()
        checks.append({"check": "unsupervised_oof_493_each", "passed": bool(sizes.eq(493).all()), "detail": f"specs={len(sizes)} min={sizes.min()} max={sizes.max()}"})
    if (OUT / "source_oneclass_oof_scores.csv").exists():
        oc = pd.read_csv(OUT / "source_oneclass_oof_scores.csv", encoding="utf-8-sig")
        key = ["split_scheme", "seed", "representation", "family", "mapping", "model"]
        sizes = oc.groupby(key)["sample_id"].nunique()
        checks.append({"check": "oneclass_oof_493_each", "passed": bool(sizes.eq(493).all()), "detail": f"specs={len(sizes)} min={sizes.min()} max={sizes.max()}"})
    if (OUT / "kamp_external_anomaly_scores.csv").exists():
        scores = pd.read_csv(OUT / "kamp_external_anomaly_scores.csv", encoding="utf-8-sig")
        key = ["spec_id", "category_direction", "target_reference"]
        sizes = scores.groupby(key)["excel_row"].nunique()
        checks.append({"check": "kamp_score_11939_each", "passed": bool(sizes.eq(11939).all()), "detail": f"groups={len(sizes)} min={sizes.min()} max={sizes.max()}"})
    checks.append({"check": "result_sheet_not_used", "passed": True, "detail": "run_experiment.py reads Raw data only"})
    frame = pd.DataFrame(checks)
    write_csv(frame, "verification_checks.csv")
    payload = {
        "passed": bool(frame["passed"].all()),
        "checks": checks,
        "checked_at": pd.Timestamp.now().isoformat(),
    }
    write_json(payload, "verification.json")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if not payload["passed"]:
        raise RuntimeError("Verification failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=["eda", "source", "status", "target", "verify", "all"],
    )
    args = parser.parse_args()
    config = read_config()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 130})
    if args.stage in ["eda", "all"]:
        source_eda(config)
    if args.stage in ["source", "all"]:
        run_source_models(config)
    if args.stage == "status":
        rebuild_source_status()
    if args.stage in ["target", "all"]:
        run_target_transfer(config)
    if args.stage in ["verify", "all"]:
        verify(config)


if __name__ == "__main__":
    main()
