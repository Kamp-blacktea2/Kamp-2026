from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
Y12_OUT = HERE.parent / "ysh-012" / "outputs"
Y11_CONFIG = HERE.parent / "ysh-011" / "config.json"
KAMP_PATH = ROOT / "data" / "Welding_Data_Set_01.xlsx"
OUT = HERE / "outputs"
FIG = OUT / "figures"
RAW4 = ["F", "I", "V", "t"]

BURDENS = [
    "strict_row_rate",
    "strict_unique_rate",
    "sensitivity_3of4_row_rate",
    "sensitivity_3of4_unique_rate",
    "row_median_external_rank",
    "unique_median_external_rank",
]
OUTCOMES = ["total_recorded_defects", "type1", "type2", "type3"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.12g")


def write_json(payload: dict, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
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
    raise TypeError(type(value).__name__)


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(spearmanr(x.to_numpy(), y.to_numpy()).statistic)


def sign(value: float) -> int | None:
    if not np.isfinite(value):
        return None
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def load_frozen_rows() -> pd.DataFrame:
    verification = json.loads((Y12_OUT / "verification.json").read_text(encoding="utf-8-sig"))
    if not verification.get("passed"):
        raise RuntimeError("YSH-012 verification did not pass")
    rows = pd.read_csv(Y12_OUT / "robust_candidate_rows.csv", encoding="utf-8-sig")
    required = {
        "excel_row",
        "date",
        *RAW4,
        "median_rank",
        "strict_4of4_top5",
        "sensitivity_3of4_top5",
    }
    if required - set(rows.columns):
        raise RuntimeError(f"YSH-012 candidate columns missing: {sorted(required - set(rows.columns))}")
    if len(rows) != 11939 or rows["excel_row"].nunique() != 11939:
        raise RuntimeError("Unexpected YSH-012 row count")
    if rows[RAW4].drop_duplicates().shape[0] != 1574:
        raise RuntimeError("Unexpected YSH-012 unique Raw4 count")
    if int(rows["strict_4of4_top5"].sum()) != 277:
        raise RuntimeError("Frozen strict candidate count changed")
    return rows


def candidate_burden(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for date, part in rows.groupby("date", sort=True):
        unique = part.drop_duplicates(RAW4)
        output.append(
            {
                "date": date,
                "date_rows": len(part),
                "date_unique_Raw4": len(unique),
                "strict_candidate_rows": int(part["strict_4of4_top5"].sum()),
                "strict_candidate_unique_Raw4": int(unique["strict_4of4_top5"].sum()),
                "sensitivity_3of4_candidate_rows": int(part["sensitivity_3of4_top5"].sum()),
                "sensitivity_3of4_candidate_unique_Raw4": int(unique["sensitivity_3of4_top5"].sum()),
                "strict_row_rate": part["strict_4of4_top5"].mean(),
                "strict_unique_rate": unique["strict_4of4_top5"].mean(),
                "sensitivity_3of4_row_rate": part["sensitivity_3of4_top5"].mean(),
                "sensitivity_3of4_unique_rate": unique["sensitivity_3of4_top5"].mean(),
                "row_median_external_rank": part["median_rank"].median(),
                "unique_median_external_rank": unique["median_rank"].median(),
            }
        )
    return pd.DataFrame(output)


def result_counts(kamp_dates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = json.loads(Y11_CONFIG.read_text(encoding="utf-8"))
    if sha256(KAMP_PATH) != config["kamp_sha256"]:
        raise RuntimeError("KAMP workbook SHA-256 mismatch")
    raw_result = pd.read_excel(KAMP_PATH, sheet_name="result")
    required = {"working time", "defect", "defect type"}
    if required - set(raw_result.columns):
        raise RuntimeError(f"Result columns missing: {sorted(required - set(raw_result.columns))}")
    work = raw_result[["working time", "defect", "defect type"]].copy()
    work["date"] = pd.to_datetime(work["working time"]).dt.strftime("%Y-%m-%d")
    if work.duplicated(["date", "defect type"]).any():
        raise RuntimeError("Duplicate date/defect-type records in Result")
    observed_types = set(work["defect type"].astype(int))
    if observed_types != {1, 2, 3}:
        raise RuntimeError(f"Unexpected defect types: {observed_types}")

    pivot = work.pivot(index="date", columns="defect type", values="defect").rename(
        columns={1: "type1", 2: "type2", 3: "type3"}
    )
    pivot = pivot.reindex(kamp_dates)
    pivot.index.name = "date"
    pivot = pivot.reset_index()
    pivot["observed_type_count"] = pivot[["type1", "type2", "type3"]].notna().sum(axis=1)
    pivot["total_recorded_defects"] = pivot[["type1", "type2", "type3"]].sum(
        axis=1, min_count=3
    )
    return pivot, raw_result


def associations(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    lodo_rows = []
    for burden in BURDENS:
        for outcome in OUTCOMES:
            complete = joined[["date", burden, outcome]].dropna().copy()
            rho = safe_spearman(complete[burden], complete[outcome])
            current_lodo = []
            for omitted in complete["date"]:
                remaining = complete.loc[~complete["date"].eq(omitted)]
                lodo_rho = safe_spearman(remaining[burden], remaining[outcome])
                current_lodo.append(lodo_rho)
                lodo_rows.append(
                    {
                        "burden": burden,
                        "outcome": outcome,
                        "omitted_date": omitted,
                        "n_dates_after_omission": len(remaining),
                        "spearman_rho": lodo_rho,
                        "rho_sign": sign(lodo_rho),
                    }
                )
            finite = np.asarray([value for value in current_lodo if np.isfinite(value)], dtype=float)
            overall_sign = sign(rho)
            lodo_signs = [sign(value) for value in finite]
            summaries.append(
                {
                    "burden": burden,
                    "burden_role": (
                        "primary"
                        if burden in {"strict_row_rate", "strict_unique_rate"}
                        else ("sensitivity" if "sensitivity_3of4" in burden else "supplementary")
                    ),
                    "outcome": outcome,
                    "outcome_role": "primary" if outcome == "total_recorded_defects" else "type_specific",
                    "n_dates": len(complete),
                    "spearman_rho": rho,
                    "rho_sign": overall_sign,
                    "lodo_valid_count": len(finite),
                    "lodo_min_rho": finite.min() if len(finite) else np.nan,
                    "lodo_max_rho": finite.max() if len(finite) else np.nan,
                    "lodo_positive_count": int((finite > 0).sum()),
                    "lodo_negative_count": int((finite < 0).sum()),
                    "lodo_zero_count": int((finite == 0).sum()),
                    "lodo_na_count": len(current_lodo) - len(finite),
                    "lodo_sign_stable": bool(
                        overall_sign is not None
                        and overall_sign != 0
                        and len(lodo_signs) == len(current_lodo)
                        and all(value == overall_sign for value in lodo_signs)
                    ),
                    "lodo_all_positive": bool(
                        len(finite) == len(current_lodo) and len(finite) > 0 and (finite > 0).all()
                    ),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(lodo_rows)


def primary_judgment(summary: pd.DataFrame) -> str:
    primary = summary.loc[
        summary["outcome"].eq("total_recorded_defects")
        & summary["burden"].isin(["strict_row_rate", "strict_unique_rate"])
    ].set_index("burden")
    row = primary.loc["strict_row_rate"]
    unique = primary.loc["strict_unique_rate"]
    if row["n_dates"] < 4 or unique["n_dates"] < 4 or not np.isfinite(row["spearman_rho"]) or not np.isfinite(unique["spearman_rho"]):
        return "untestable"
    row_rho = float(row["spearman_rho"])
    unique_rho = float(unique["spearman_rho"])
    if row_rho > 0 and unique_rho > 0:
        if bool(row["lodo_all_positive"]) and bool(unique["lodo_all_positive"]):
            return "directionally_supported_stable"
        return "positive_but_unstable"
    if sign(row_rho) != sign(unique_rho):
        return "mixed_row_vs_unique"
    return "not_supported"


def make_figures(
    joined: pd.DataFrame,
    summary: pd.DataFrame,
    lodo: pd.DataFrame,
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 130})
    colors = {"strict": "#4c78a8", "sensitivity": "#f58518"}

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    x = np.arange(len(joined))
    axes[0].plot(x, joined["strict_row_rate"], marker="o", label="strict row rate", color=colors["strict"])
    axes[0].plot(x, joined["sensitivity_3of4_row_rate"], marker="o", label=">=3/4 row rate", color=colors["sensitivity"])
    axes[0].set(ylabel="candidate row rate", title="Row-weighted burden")
    axes[1].plot(x, joined["strict_unique_rate"], marker="o", label="strict unique rate", color=colors["strict"])
    axes[1].plot(x, joined["sensitivity_3of4_unique_rate"], marker="o", label=">=3/4 unique rate", color=colors["sensitivity"])
    axes[1].set(ylabel="candidate unique-Raw4 rate", title="Unique-Raw4 burden")
    axes[1].set_xticks(x, joined["date"], rotation=35, ha="right")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    fig.suptitle("Figure 1. Frozen candidate burden by date")
    fig.tight_layout()
    fig.savefig(FIG / "figure1_date_candidate_burden.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    complete = joined.dropna(subset=["total_recorded_defects"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, burden, title in [
        (axes[0], "strict_row_rate", "Strict row rate"),
        (axes[1], "strict_unique_rate", "Strict unique-Raw4 rate"),
    ]:
        axis.scatter(complete[burden], complete["total_recorded_defects"], s=55, color="#4c78a8")
        for row in complete.itertuples(index=False):
            axis.annotate(row.date[5:], (getattr(row, burden), row.total_recorded_defects), xytext=(4, 4), textcoords="offset points", fontsize=8)
        rho = summary.loc[
            summary["burden"].eq(burden) & summary["outcome"].eq("total_recorded_defects"),
            "spearman_rho",
        ].iloc[0]
        axis.set(title=f"{title} (rho={rho:.3f}, n={len(complete)})", xlabel=burden, ylabel="total recorded defects")
        axis.grid(alpha=0.2)
    fig.suptitle("Figure 2. Strict burden versus complete-case total defects")
    fig.tight_layout()
    fig.savefig(FIG / "figure2_strict_vs_total.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    rho_pivot = summary.pivot(index="burden", columns="outcome", values="spearman_rho").loc[BURDENS, OUTCOMES]
    n_pivot = summary.pivot(index="burden", columns="outcome", values="n_dates").loc[BURDENS, OUTCOMES]
    fig, axis = plt.subplots(figsize=(11, 7))
    image = axis.imshow(rho_pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(np.arange(len(rho_pivot.columns)), rho_pivot.columns, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(rho_pivot.index)), rho_pivot.index)
    for i in range(len(rho_pivot.index)):
        for j in range(len(rho_pivot.columns)):
            axis.text(j, i, f"{rho_pivot.iloc[i, j]:.2f}\nn={int(n_pivot.iloc[i, j])}", ha="center", va="center", fontsize=8)
    axis.set_title("Figure 3. Date-level Spearman associations — all prespecified combinations")
    fig.colorbar(image, ax=axis, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(FIG / "figure3_association_heatmap.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    selected_burdens = BURDENS[:4]
    total_summary = summary.loc[
        summary["outcome"].eq("total_recorded_defects") & summary["burden"].isin(selected_burdens)
    ].set_index("burden").loc[selected_burdens]
    fig, axis = plt.subplots(figsize=(11, 5.5))
    y = np.arange(len(total_summary))
    axis.hlines(y, total_summary["lodo_min_rho"], total_summary["lodo_max_rho"], color="#9ecae9", linewidth=5)
    axis.scatter(total_summary["spearman_rho"], y, color="#e45756", s=65, label="overall rho", zorder=3)
    axis.axvline(0, color="black", linewidth=1, linestyle="--")
    axis.set_yticks(y, total_summary.index)
    axis.set(xlabel="Spearman rho", title="Figure 4. Leave-one-date-out range for total defects", xlim=(-1.05, 1.05))
    axis.legend()
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "figure4_lodo_stability.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def verify(
    rows: pd.DataFrame,
    burden: pd.DataFrame,
    result: pd.DataFrame,
    raw_result: pd.DataFrame,
    joined: pd.DataFrame,
    summary: pd.DataFrame,
    lodo: pd.DataFrame,
    judgment: str,
) -> dict:
    required_csv = [
        "date_candidate_burden.csv",
        "date_result_counts.csv",
        "date_joined_analysis.csv",
        "spearman_associations.csv",
        "lodo_associations.csv",
    ]
    figures = [
        "figure1_date_candidate_burden.png",
        "figure2_strict_vs_total.png",
        "figure3_association_heatmap.png",
        "figure4_lodo_stability.png",
    ]
    result_by_date = result.set_index("date")
    checks = {
        "frozen_input_rows_11939": len(rows) == 11939,
        "frozen_strict_rows_277": int(rows["strict_4of4_top5"].sum()) == 277,
        "kamp_dates_9": len(burden) == 9 and burden["date"].nunique() == 9,
        "raw_result_rows_23": len(raw_result) == 23,
        "result_dates_joined_9": len(joined) == 9,
        "missing_not_filled_with_zero": bool(
            result_by_date.loc["2020-03-27", ["type1", "type2", "type3", "total_recorded_defects"]].isna().all()
            and pd.isna(result_by_date.loc["2020-03-31", "type3"])
            and pd.isna(result_by_date.loc["2020-03-31", "total_recorded_defects"])
        ),
        "type_sample_sizes_expected": bool(
            result["type1"].notna().sum() == 8
            and result["type2"].notna().sum() == 8
            and result["type3"].notna().sum() == 7
            and result["total_recorded_defects"].notna().sum() == 7
        ),
        "all_24_associations_reported": len(summary) == len(BURDENS) * len(OUTCOMES),
        "lodo_rows_expected": len(lodo) == len(BURDENS) * (7 + 8 + 8 + 7),
        "no_p_value_column": not any("p_value" in column.lower() for column in summary.columns),
        "required_csv_present": all((OUT / name).exists() for name in required_csv),
        "four_figures_present": all((FIG / name).exists() for name in figures),
        "new_model_training": False,
        "new_threshold_selection": False,
        "result_used_for_candidate_definition": False,
        "result_used_for_association_only": True,
    }
    positive_checks = [
        key
        for key in checks
        if key
        not in {
            "new_model_training",
            "new_threshold_selection",
            "result_used_for_candidate_definition",
        }
    ]
    passed = all(checks[key] for key in positive_checks) and not any(
        checks[key]
        for key in ["new_model_training", "new_threshold_selection", "result_used_for_candidate_definition"]
    )
    return {
        "experiment": "YSH-013",
        "checks": checks,
        "primary_judgment": judgment,
        "passed": bool(passed),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load_frozen_rows()
    burden = candidate_burden(rows)
    result, raw_result = result_counts(burden["date"].tolist())
    joined = burden.merge(result, on="date", validate="one_to_one")
    summary, lodo = associations(joined)
    judgment = primary_judgment(summary)

    write_csv(burden, "date_candidate_burden.csv")
    write_csv(result, "date_result_counts.csv")
    write_csv(joined, "date_joined_analysis.csv")
    write_csv(summary, "spearman_associations.csv")
    write_csv(lodo, "lodo_associations.csv")
    make_figures(joined, summary, lodo)

    verification = verify(rows, burden, result, raw_result, joined, summary, lodo, judgment)
    write_json(verification, "verification.json")
    if not verification["passed"]:
        raise RuntimeError(f"Verification failed: {verification}")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
