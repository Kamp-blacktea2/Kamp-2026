from __future__ import annotations

import json
import warnings
from itertools import combinations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    FIG,
    HERE,
    KAMP_FEATURES,
    MODELS,
    OUT,
    Y7,
    Y8,
    aggregate_source,
    classification_metrics,
    ensure_output_dirs,
    episode_ids,
    json_default,
    load_kamp,
    load_source_raw,
    optimal_bacc_threshold,
    percentile_rank,
    quantile_transform,
    read_config,
    regression_metrics,
    robust_fit,
    robust_transform,
    sha256,
    source_transform_apply,
    source_transform_fit,
    source_model_cohort,
    write_csv,
    write_json,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 130})

MAPPINGS = {
    "M1_mean": "current_mean_ka",
    "M2_max": "current_max_ka",
    "M3_min": "current_min_ka",
}
REPRESENTATIONS = ["RZ", "Q"]


def transfer_matrix(source: pd.DataFrame, mapping: str) -> np.ndarray:
    return source[["pressure_bar", MAPPINGS[mapping], "welding_time_ms"]].to_numpy(float)


def classification_splits(source: pd.DataFrame, split: str, config: dict):
    y = source["abnormal"].to_numpy(int)
    groups = source["condition_group"].to_numpy()
    for seed in config["seeds"]:
        if split == "stratified":
            splitter = StratifiedKFold(config["folds"], shuffle=True, random_state=seed)
            iterator = splitter.split(source, y)
        elif split == "group":
            splitter = StratifiedGroupKFold(config["folds"], shuffle=True, random_state=seed)
            iterator = splitter.split(source, y, groups)
        else:
            raise ValueError(split)
        for fold, (train, test) in enumerate(iterator):
            yield seed, fold, train, test


def regression_splits(source: pd.DataFrame, split: str, config: dict):
    groups = source["condition_group"].to_numpy()
    for seed in config["seeds"]:
        if split == "sample":
            iterator = KFold(config["folds"], shuffle=True, random_state=seed).split(source)
        elif split == "group":
            iterator = GroupKFold(config["folds"], shuffle=True, random_state=seed).split(source, groups=groups)
        else:
            raise ValueError(split)
        for fold, (train, test) in enumerate(iterator):
            yield seed, fold, train, test


def fit_logistic(config: dict) -> LogisticRegression:
    return LogisticRegression(
        C=config["logistic"]["C"],
        class_weight=config["logistic"]["class_weight"],
        max_iter=config["logistic"]["max_iter"],
        random_state=42,
    )


def fit_rf_classifier(config: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config["random_forest"]["n_estimators"],
        class_weight="balanced_subsample",
        min_samples_leaf=config["random_forest"]["min_samples_leaf"],
        max_features=config["random_forest"]["max_features"],
        random_state=config["random_forest"]["random_state"],
        n_jobs=config["random_forest"]["n_jobs"],
    )


def bootstrap_classification(
    y: np.ndarray,
    score: np.ndarray,
    threshold: float,
    groups: np.ndarray | None,
    draws: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    values = {"roc_auc": [], "pr_auc": [], "balanced_accuracy": []}
    rejected = 0
    unique_groups = np.unique(groups) if groups is not None else None
    for _ in range(draws):
        if unique_groups is None:
            index = rng.integers(0, len(y), len(y))
        else:
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            index = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        if np.unique(y[index]).size < 2:
            rejected += 1
            continue
        values["roc_auc"].append(roc_auc_score(y[index], score[index]))
        values["pr_auc"].append(average_precision_score(y[index], score[index]))
        values["balanced_accuracy"].append(
            balanced_accuracy_score(y[index], score[index] >= threshold)
        )
    result = {"bootstrap_valid": len(values["roc_auc"]), "bootstrap_rejected": rejected}
    for metric, metric_values in values.items():
        result[f"{metric}_ci_low"] = float(np.quantile(metric_values, 0.025))
        result[f"{metric}_ci_high"] = float(np.quantile(metric_values, 0.975))
    return result


def normalized_classification_cv(source: pd.DataFrame, config: dict):
    y = source["abnormal"].to_numpy(int)
    groups = source["condition_group"].to_numpy()
    predictions = []
    metrics = []
    parameters = []
    split_manifest = []

    for split in ["stratified", "group"]:
        split_cache = list(classification_splits(source, split, config))
        if not split_manifest:
            pass
        for seed, fold, train, test in split_cache:
            for index in test:
                split_manifest.append(
                    {
                        "split": split,
                        "seed": seed,
                        "fold": fold,
                        "sample_id": int(source.iloc[index]["sample_id"]),
                        "condition_group": source.iloc[index]["condition_group"],
                        "abnormal": int(y[index]),
                    }
                )

        for representation in REPRESENTATIONS:
            for mapping in MAPPINGS:
                x = transfer_matrix(source, mapping)
                for model_name in ["logistic", "random_forest"]:
                    score_sum = np.zeros(len(source), dtype=float)
                    score_count = np.zeros(len(source), dtype=int)
                    invalid_folds = 0
                    for seed, fold, train, test in split_cache:
                        try:
                            transform = source_transform_fit(x[train], y[train], representation)
                            x_train = source_transform_apply(x[train], transform)
                            x_test = source_transform_apply(x[test], transform)
                        except ValueError:
                            invalid_folds += 1
                            continue
                        if model_name == "logistic":
                            model = fit_logistic(config).fit(x_train, y[train])
                            fold_score = model.decision_function(x_test)
                            for feature, coefficient in zip(["pressure_like", "current", "time"], model.coef_[0]):
                                parameters.append(
                                    {
                                        "split": split,
                                        "seed": seed,
                                        "fold": fold,
                                        "model": model_name,
                                        "representation": representation,
                                        "mapping": mapping,
                                        "feature": feature,
                                        "coefficient": coefficient,
                                        "intercept": model.intercept_[0],
                                    }
                                )
                        else:
                            model = fit_rf_classifier(config).fit(x_train, y[train])
                            probability = model.predict_proba(x_test)[:, 1]
                            fold_score = np.log(np.clip(probability, 1e-9, 1 - 1e-9) / np.clip(1 - probability, 1e-9, 1))
                        score_sum[test] += fold_score
                        score_count[test] += 1
                    if invalid_folds:
                        metrics.append(
                            {
                                "model": model_name,
                                "representation": representation,
                                "mapping": mapping,
                                "split": split,
                                "invalid_folds": invalid_folds,
                                "samples": len(source),
                                "roc_auc": np.nan,
                                "pr_auc": np.nan,
                                "balanced_accuracy": np.nan,
                                "f1": np.nan,
                                "recall": np.nan,
                                "precision": np.nan,
                                "specificity": np.nan,
                                "brier": np.nan,
                                "tn": np.nan,
                                "fp": np.nan,
                                "fn": np.nan,
                                "tp": np.nan,
                                "threshold": 0.0,
                                "invalid_reason": "non_positive_good_iqr_in_at_least_one_fold",
                            }
                        )
                        continue
                    if np.any(score_count == 0):
                        raise AssertionError(f"Missing OOF score: {split}/{representation}/{mapping}/{model_name}")
                    score = score_sum / score_count
                    metric = classification_metrics(y, score, 0.0)
                    metric.update(
                        {
                            "model": model_name,
                            "representation": representation,
                            "mapping": mapping,
                            "split": split,
                            "invalid_folds": invalid_folds,
                            "samples": len(source),
                        }
                    )
                    if model_name == "logistic":
                        metric.update(
                            bootstrap_classification(
                                y,
                                score,
                                0.0,
                                groups if split == "group" else None,
                                config["bootstrap_draws"],
                                config["bootstrap_seed"],
                            )
                        )
                    metrics.append(metric)
                    predictions.append(
                        pd.DataFrame(
                            {
                                "sample_id": source["sample_id"],
                                "category": source["category"],
                                "abnormal": y,
                                "condition_group": groups,
                                "model": model_name,
                                "representation": representation,
                                "mapping": mapping,
                                "split": split,
                                "oof_score": score,
                                "oof_probability_for_brier_only": 1 / (1 + np.exp(-score)),
                            }
                        )
                    )
    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(metrics),
        pd.DataFrame(parameters),
        pd.DataFrame(split_manifest).drop_duplicates(),
    )


def raw_baseline_cv(source: pd.DataFrame, config: dict):
    feature_sets = {
        "time_only": ["welding_time_ms"],
        "setting_only": ["pressure_bar", "welding_time_ms", "angle_deg"],
        "common_M1": ["pressure_bar", "current_mean_ka", "welding_time_ms"],
        "common_M2": ["pressure_bar", "current_max_ka", "welding_time_ms"],
        "common_M3": ["pressure_bar", "current_min_ka", "welding_time_ms"],
        "full_source": [
            "pressure_bar",
            "welding_time_ms",
            "angle_deg",
            "current_min_ka",
            "current_mean_ka",
            "current_max_ka",
            "force_min",
            "force_mean",
            "force_max",
            "thickness_a_mm",
            "thickness_b_mm",
        ],
    }
    y = source["abnormal"].to_numpy(int)
    rows = []
    prediction_rows = []
    for split in ["stratified", "group"]:
        splits = list(classification_splits(source, split, config))
        for name, columns in feature_sets.items():
            x = source[columns].to_numpy(float)
            score_sum = np.zeros(len(source))
            score_count = np.zeros(len(source), dtype=int)
            for _, _, train, test in splits:
                model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    LogisticRegression(
                        C=config["logistic"]["C"],
                        class_weight="balanced",
                        max_iter=config["logistic"]["max_iter"],
                        random_state=42,
                    ),
                ).fit(x[train], y[train])
                score_sum[test] += model.decision_function(x[test])
                score_count[test] += 1
            score = score_sum / score_count
            metric = classification_metrics(y, score, 0.0)
            metric.update(
                {
                    "model": "logistic_raw_baseline",
                    "representation": "RAW_STANDARD",
                    "mapping": name,
                    "split": split,
                    "invalid_folds": 0,
                    "samples": len(source),
                }
            )
            rows.append(metric)
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "sample_id": source["sample_id"],
                        "category": source["category"],
                        "abnormal": y,
                        "condition_group": source["condition_group"],
                        "model": "logistic_raw_baseline",
                        "representation": "RAW_STANDARD",
                        "mapping": name,
                        "split": split,
                        "oof_score": score,
                        "oof_probability_for_brier_only": 1 / (1 + np.exp(-score)),
                    }
                )
            )
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(rows)


def regression_cv(source: pd.DataFrame, config: dict):
    metric_rows = []
    prediction_rows = []
    targets = {
        "nugget_mm": source["nugget_mm"].notna(),
        "pulltest_n": source["pulltest_n"].notna() & ~source["sample_id"].isin([161, 185, 213]),
    }
    for target, valid in targets.items():
        subset = source.loc[valid].reset_index(drop=True)
        y = subset[target].to_numpy(float)
        for representation in REPRESENTATIONS:
            for mapping in MAPPINGS:
                x = transfer_matrix(subset, mapping)
                for split in ["sample", "group"]:
                    splits = list(regression_splits(subset, split, config))
                    for model_name in ["ridge", "random_forest"]:
                        pred_sum = np.zeros(len(subset))
                        dummy_sum = np.zeros(len(subset))
                        pred_count = np.zeros(len(subset), dtype=int)
                        invalid_folds = 0
                        for _, _, train, test in splits:
                            try:
                                transform = source_transform_fit(
                                    x[train], subset.iloc[train]["abnormal"].to_numpy(int), representation
                                )
                            except ValueError:
                                invalid_folds += 1
                                continue
                            x_train = source_transform_apply(x[train], transform)
                            x_test = source_transform_apply(x[test], transform)
                            if model_name == "ridge":
                                model = Ridge(alpha=config["ridge_alpha"]).fit(x_train, y[train])
                            else:
                                model = RandomForestRegressor(
                                    n_estimators=config["random_forest"]["n_estimators"],
                                    min_samples_leaf=config["random_forest"]["min_samples_leaf"],
                                    max_features=config["random_forest"]["max_features"],
                                    random_state=config["random_forest"]["random_state"],
                                    n_jobs=config["random_forest"]["n_jobs"],
                                ).fit(x_train, y[train])
                            dummy = DummyRegressor(strategy="mean").fit(x_train, y[train])
                            pred_sum[test] += model.predict(x_test)
                            dummy_sum[test] += dummy.predict(x_test)
                            pred_count[test] += 1
                        if invalid_folds:
                            metric_rows.append(
                                {
                                    "target": target,
                                    "model": model_name,
                                    "representation": representation,
                                    "mapping": mapping,
                                    "split": split,
                                    "samples": len(subset),
                                    "mae": np.nan,
                                    "rmse": np.nan,
                                    "r2": np.nan,
                                    "spearman": np.nan,
                                    "dummy_mae": np.nan,
                                    "dummy_mae_improvement": np.nan,
                                    "invalid_folds": invalid_folds,
                                    "invalid_reason": "non_positive_good_iqr_in_at_least_one_fold",
                                }
                            )
                            continue
                        prediction = pred_sum / pred_count
                        dummy_prediction = dummy_sum / pred_count
                        metrics = regression_metrics(y, prediction, dummy_prediction)
                        metrics.update(
                            {
                                "target": target,
                                "model": model_name,
                                "representation": representation,
                                "mapping": mapping,
                                "split": split,
                                "samples": len(subset),
                            }
                        )
                        metric_rows.append(metrics)
                        prediction_rows.append(
                            pd.DataFrame(
                                {
                                    "sample_id": subset["sample_id"],
                                    "target": target,
                                    "actual": y,
                                    "prediction": prediction,
                                    "dummy_prediction": dummy_prediction,
                                    "model": model_name,
                                    "representation": representation,
                                    "mapping": mapping,
                                    "split": split,
                                }
                            )
                        )
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(metric_rows)


def gate_decisions(metrics: pd.DataFrame, regression: pd.DataFrame, config: dict, prevalence: float):
    gate = config["gate"]
    logistic = metrics.loc[metrics["model"].eq("logistic")].copy()
    wide = logistic.pivot_table(
        index=["representation", "mapping"],
        columns="split",
        values=["roc_auc", "pr_auc", "balanced_accuracy"],
        aggfunc="first",
    )
    rows = []
    combo_status = {}
    for representation in REPRESENTATIONS:
        for mapping in MAPPINGS:
            key = (representation, mapping)
            if key not in wide.index:
                combo_status[key] = {"A": False, "B": False, "auc_drop": None}
                rows.extend(
                    [
                        {
                            "gate": gate_name,
                            "scope": f"{representation}_{mapping}",
                            "status": "fail",
                            "metric_summary": "invalid_non_positive_good_iqr_or_missing_oof",
                            "consequence": consequence,
                        }
                        for gate_name, consequence in [
                            ("A", "no_kamp_normalized_scoring"),
                            ("B", "recipe_dependent_or_weak"),
                        ]
                    ]
                )
                continue
            values = wide.loc[key]
            required = [
                (metric, split)
                for metric in ["roc_auc", "pr_auc", "balanced_accuracy"]
                for split in ["stratified", "group"]
            ]
            if any(pd.isna(values.get(item, np.nan)) for item in required):
                combo_status[key] = {"A": False, "B": False, "auc_drop": None}
                rows.extend(
                    [
                        {
                            "gate": gate_name,
                            "scope": f"{representation}_{mapping}",
                            "status": "fail",
                            "metric_summary": "invalid_non_positive_good_iqr_or_missing_oof",
                            "consequence": consequence,
                        }
                        for gate_name, consequence in [
                            ("A", "no_kamp_normalized_scoring"),
                            ("B", "recipe_dependent_or_weak"),
                        ]
                    ]
                )
                continue
            gate_a = (
                values[("roc_auc", "stratified")] > gate["roc_auc_min_exclusive"]
                and values[("pr_auc", "stratified")] > prevalence
                and values[("balanced_accuracy", "stratified")] > gate["balanced_accuracy_min_exclusive"]
            )
            auc_drop = values[("roc_auc", "stratified")] - values[("roc_auc", "group")]
            gate_b = gate_a and (
                values[("roc_auc", "group")] > gate["roc_auc_min_exclusive"]
                and values[("pr_auc", "group")] > prevalence
                and values[("balanced_accuracy", "group")] > gate["balanced_accuracy_min_exclusive"]
                and auc_drop <= gate["group_auc_drop_max"]
            )
            combo_status[key] = {"A": gate_a, "B": gate_b, "auc_drop": auc_drop}
            rows.extend(
                [
                {
                    "gate": "A",
                    "scope": f"{representation}_{mapping}",
                    "status": "pass" if gate_a else "fail",
                    "metric_summary": f"stratified_auc={values[('roc_auc','stratified')]:.6f};pr={values[('pr_auc','stratified')]:.6f};bacc0={values[('balanced_accuracy','stratified')]:.6f}",
                    "consequence": "eligible_for_gate_B" if gate_a else "no_kamp_normalized_scoring",
                },
                {
                    "gate": "B",
                    "scope": f"{representation}_{mapping}",
                    "status": "pass" if gate_b else "fail",
                    "metric_summary": f"group_auc={values[('roc_auc','group')]:.6f};pr={values[('pr_auc','group')]:.6f};bacc0={values[('balanced_accuracy','group')]:.6f};auc_drop={auc_drop:.6f}",
                    "consequence": "eligible_for_gate_C" if gate_b else "recipe_dependent_or_weak",
                },
                ]
            )
    family_counts = {
        representation: sum(combo_status[(representation, mapping)]["B"] for mapping in MAPPINGS)
        for representation in REPRESENTATIONS
    }
    gate_a_global = any(status["A"] for status in combo_status.values())
    gate_b_global = any(status["B"] for status in combo_status.values())
    gate_c_global = any(count >= 2 for count in family_counts.values())
    for representation, count in family_counts.items():
        rows.append(
            {
                "gate": "C",
                "scope": representation,
                "status": "pass" if count >= 2 else "fail",
                "metric_summary": f"gate_B_current_mappings={count}/3",
                "consequence": "normalized_family_supported" if count >= 2 else "mapping_sensitive_or_unsupported",
            }
        )

    ridge_group = regression.loc[
        regression["model"].eq("ridge") & regression["split"].eq("group")
    ].copy()
    regression_status = {}
    for row in ridge_group.itertuples(index=False):
        passed = (
            np.isfinite(row.r2)
            and row.r2 > gate["regression_r2_min_exclusive"]
            and row.spearman > gate["regression_spearman_min_exclusive"]
            and row.dummy_mae_improvement >= gate["regression_dummy_mae_improvement_min"]
        )
        regression_status[(row.target, row.representation, row.mapping)] = passed
        rows.append(
            {
                "gate": "regression",
                "scope": f"{row.target}_{row.representation}_{row.mapping}",
                "status": "pass" if passed else "fail",
                "metric_summary": f"r2={row.r2:.6f};rho={row.spearman:.6f};mae_improvement={row.dummy_mae_improvement:.6f}",
                "consequence": "relative_quality_direction_allowed" if passed else "relative_quality_direction_blocked",
            }
        )
    summary = {
        "gate_A": gate_a_global,
        "gate_B": gate_b_global,
        "gate_C": gate_c_global,
        "gate_C_family_counts": family_counts,
        "combo_status": {
            f"{representation}_{mapping}": value
            for (representation, mapping), value in combo_status.items()
        },
        "regression_pass_count": int(sum(regression_status.values())),
    }
    return pd.DataFrame(rows), summary, combo_status, regression_status


def threshold_table(predictions: pd.DataFrame, combo_status: dict) -> pd.DataFrame:
    rows = []
    selected = predictions.loc[
        predictions["model"].eq("logistic") & predictions["split"].eq("group")
    ]
    for (representation, mapping), frame in selected.groupby(["representation", "mapping"]):
        threshold, metrics = optimal_bacc_threshold(
            frame["abnormal"].to_numpy(int), frame["oof_score"].to_numpy(float)
        )
        rows.append(
            {
                "representation": representation,
                "mapping": mapping,
                "criterion": "max_group_oof_balanced_accuracy",
                "threshold": threshold,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "specificity": metrics["specificity"],
                "recall": metrics["recall"],
                "gate_A_pass": combo_status[(representation, mapping)]["A"],
                "gate_B_pass": combo_status[(representation, mapping)]["B"],
            }
        )
    return pd.DataFrame(rows)


def source_reference_table(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = source["abnormal"].to_numpy(int)
    for mapping in MAPPINGS:
        x = transfer_matrix(source, mapping)
        good = x[y == 0]
        for feature_index, feature in enumerate(["pressure_like", "current", "time"]):
            values = good[:, feature_index]
            rows.append(
                {
                    "representation": "RZ",
                    "mapping": mapping,
                    "feature": feature,
                    "reference_population": "source_Good",
                    "n": len(values),
                    "median": np.median(values),
                    "iqr": np.quantile(values, 0.75) - np.quantile(values, 0.25),
                    "q10": np.quantile(values, 0.10),
                    "q25": np.quantile(values, 0.25),
                    "q50": np.quantile(values, 0.50),
                    "q75": np.quantile(values, 0.75),
                    "q90": np.quantile(values, 0.90),
                }
            )
            rows.append(
                {
                    "representation": "Q",
                    "mapping": mapping,
                    "feature": feature,
                    "reference_population": "source_Good_ECDF",
                    "n": len(values),
                    "median": np.median(values),
                    "iqr": np.quantile(values, 0.75) - np.quantile(values, 0.25),
                    "q10": np.quantile(values, 0.10),
                    "q25": np.quantile(values, 0.25),
                    "q50": np.quantile(values, 0.50),
                    "q75": np.quantile(values, 0.75),
                    "q90": np.quantile(values, 0.90),
                }
            )
    return pd.DataFrame(rows)


def figure_source_map(source: pd.DataFrame) -> None:
    colors = source["category"].map({"Good": "#4c78a8", "Bad": "#e45756", "Explode": "#f58518"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].scatter(source["pressure_psi"], source["welding_time_ms"], c=colors, s=18, alpha=0.7)
    axes[0].set(xlabel="Pressure (PSI)", ylabel="Welding time (ms)", title="DOE positions and Category")
    scatter = axes[1].scatter(
        source["pressure_psi"], source["current_mean_ka"], c=source["nugget_mm"], cmap="viridis", s=24
    )
    axes[1].set(xlabel="Pressure (PSI)", ylabel="Current mean (kA)", title="Nugget diameter (mm)")
    fig.colorbar(scatter, ax=axes[1], label="mm")
    scatter = axes[2].scatter(
        source["welding_time_ms"], source["current_mean_ka"], c=source["pulltest_n"], cmap="magma", s=24
    )
    axes[2].set(xlabel="Welding time (ms)", ylabel="Current mean (kA)", title="PullTest (N)")
    fig.colorbar(scatter, ax=axes[2], label="N")
    for axis in axes:
        axis.grid(alpha=0.18)
    fig.suptitle("Figure 2. External RSW source quality map (493 model samples)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "figure2_source_quality_map.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_model_validation(metrics: pd.DataFrame) -> None:
    selected = metrics.loc[
        metrics["model"].isin(["logistic", "logistic_raw_baseline"])
        & (
            metrics["representation"].isin(REPRESENTATIONS)
            | metrics["mapping"].isin(["setting_only", "common_M1", "full_source"])
        )
    ].copy()
    selected["label"] = np.where(
        selected["representation"].isin(REPRESENTATIONS),
        selected["representation"] + "-" + selected["mapping"],
        selected["mapping"],
    )
    labels = selected["label"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharex=True)
    for axis, metric, title in zip(
        axes,
        ["roc_auc", "pr_auc", "balanced_accuracy"],
        ["ROC-AUC", "PR-AUC", "Balanced accuracy at score 0"],
    ):
        for offset, (split, color) in enumerate([("stratified", "#4c78a8"), ("group", "#f58518")]):
            part = selected.loc[selected["split"].eq(split)].set_index("label").reindex(labels)
            positions = np.arange(len(labels)) + (offset - 0.5) * 0.36
            axis.bar(positions, part[metric], width=0.34, label=split, color=color, alpha=0.85)
        axis.set_title(title)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=50, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend()
    fig.suptitle("Figure 3. Source model validation by split", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "figure3_source_model_validation.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_quality_outputs(source: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    specs = [
        ("nugget_mm", "Nugget (mm)"),
        ("pulltest_n", "PullTest (N)"),
        ("pressure_bar", "Pressure (bar)"),
        ("current_mean_ka", "Current mean (kA)"),
        ("welding_time_ms", "Time (ms)"),
    ]
    for axis, (column, title) in zip(axes, specs):
        axis.boxplot(
            [source.loc[source["abnormal"].eq(0), column].dropna(), source.loc[source["abnormal"].eq(1), column].dropna()],
            tick_labels=["Good", "Bad+Explode"],
            patch_artist=True,
            boxprops={"facecolor": "#9ecae9", "alpha": 0.65},
            medianprops={"color": "black"},
        )
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Figure 4. Source quality outputs and common process features", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "figure4_source_quality_outputs.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    config = read_config()
    ensure_output_dirs()
    source = source_model_cohort(aggregate_source(load_source_raw(config)), config)

    normalized_predictions, normalized_metrics, parameters, split_manifest = normalized_classification_cv(source, config)
    baseline_predictions, baseline_metrics = raw_baseline_cv(source, config)
    all_predictions = pd.concat([normalized_predictions, baseline_predictions], ignore_index=True)
    all_metrics = pd.concat([normalized_metrics, baseline_metrics], ignore_index=True, sort=False)
    regression_predictions, regression_metric_table = regression_cv(source, config)
    gates, gate_summary, combo_status, regression_status = gate_decisions(
        all_metrics, regression_metric_table, config, float(source["abnormal"].mean())
    )
    thresholds = threshold_table(all_predictions, combo_status)
    references = source_reference_table(source)

    write_csv(split_manifest, "source_split_manifest.csv")
    write_csv(all_predictions, "source_cv_predictions.csv")
    write_csv(all_metrics, "source_model_metrics.csv")
    write_csv(parameters, "source_model_parameters.csv")
    write_csv(regression_predictions, "source_regression_predictions.csv")
    write_csv(regression_metric_table, "source_regression_metrics.csv")
    write_csv(thresholds, "source_thresholds.csv")
    write_csv(references, "source_normalization_references.csv")
    write_csv(gates, "transfer_gate_decisions.csv")

    figure_source_map(source)
    figure_model_validation(all_metrics)
    figure_quality_outputs(source)

    write_json(gate_summary, "source_gate_summary.json")
    print(json.dumps(gate_summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
