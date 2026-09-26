"""F: freeze process metrics before joining aggregate quality records."""

from common import *
from scipy.stats import spearmanr


def process_metrics(raw, segments):
    rows = read("row_scores")
    candidate = read("candidate_scores")
    candidate_rows = read("candidate_row_scores")
    counts = read("candidate_counts")
    profiles = {}
    for q in range(4):
        profiles[q] = raw.loc[raw.train & ((raw.idx - 1) % 4 == q), NAMES].median().to_numpy()
    daily = []
    for date, part in raw.groupby("date", sort=True):
        record = dict(
            date=date,
            split="train" if part.train.iloc[0] else "test",
            records=len(part),
            force_above3_share=float((part.force > 3).mean()),
            force_7p8_8p0_count=int(part.force.between(7.8, 8.0).sum()),
            force_median=part.force.median(),
            current_median=part.current.median(),
            current_iqr=part.current.quantile(0.75) - part.current.quantile(0.25),
        )
        dv = []
        for s in segments[(segments.condition == "all") & (segments.date == date)].itertuples():
            dv.extend(np.abs(np.diff(segment_part(raw, s).voltage)))
        record["voltage_abs_diff_median"] = np.median(dv) if dv else np.nan
        for value in sorted(raw.time.unique()):
            record[f"time_{value:g}_share"] = float((part.time == value).mean())
        row_scores = rows[rows.date == date]
        for col in row_scores:
            if col.endswith("_score"):
                record[f"{col}_median"] = row_scores[col].median()
                record[f"{col}_p95"] = row_scores[col].quantile(0.95)
            elif col.endswith("_flag"):
                record[f"{col}_count"] = int(row_scores[col].sum())
                record[f"{col}_share"] = float(row_scores[col].mean())
        for r in range(4):
            count = counts[
                (counts.condition == "all") & (counts.r == r) & (counts.date == date)
            ].iloc[0]
            record[f"r{r}_groups"] = int(count.groups)
            record[f"r{r}_unassigned"] = int(count.unassigned)
        daily.append(record)
    save(daily, "daily_process")
    candidate_daily, correspondence = [], []
    for (key, date), part in candidate.groupby(["key", "date"], sort=False):
        values = group_values(raw, part)
        q = ((part.idx_start.to_numpy(int)[:, None] - 1) + np.arange(4)) % 4
        template = np.array([[profiles[pos] for pos in row] for row in q])
        record = dict(
            key=key,
            date=date,
            condition=part.condition.iloc[0],
            r=int(part.r.iloc[0]),
            split="train" if part.train.iloc[0] else "test",
            groups=len(part),
            force_above3_any_share=np.any(values[:, :, 0] > 3, axis=1).mean(),
            posterior_uncertain_share=(part.posterior < 0.8).mean(),
            density_flag99_count=int(part.density_flag99.sum()),
        )
        for j, name in enumerate(NAMES):
            record[f"{name}_within_variance"] = np.var(values[:, :, j], axis=1).mean()
            record[f"{name}_residual_variance"] = np.var(
                (values - template)[:, :, j], axis=1
            ).mean()
        all_states = sorted(candidate.loc[candidate.key == key, "state"].unique())
        for state in all_states:
            record[f"state_{state}_share"] = (part.state == state).mean()
        ordered = part.sort_values("excel_start")
        adjacent = (ordered.segment.to_numpy()[1:] == ordered.segment.to_numpy()[:-1]) & (
            np.diff(ordered.excel_start) == 4
        )
        changes = ordered.state.to_numpy()[1:] != ordered.state.to_numpy()[:-1]
        record["transition_pairs"] = int(adjacent.sum())
        record["transition_rate"] = changes[adjacent].mean() if adjacent.any() else np.nan
        if "_S8_" in key:
            record["if_score_median"] = part.if_score.median()
            record["if_score_p95"] = part.if_score.quantile(0.95)
            record["if_flag95_count"] = int(part.if_flag95.sum())
            record["if_flag95_share"] = part.if_flag95.mean()
            record["if_flag99_count"] = int(part.if_flag99.sum())
        candidate_daily.append(record)
    for key, part in candidate[candidate.key.str.contains("_S8_")].groupby("key"):
        matched = part.merge(candidate_rows, on="candidate_id", suffixes=("", "_row"))
        for col in candidate_rows:
            if col.endswith("_score_mean") or col.endswith("_score_max"):
                correspondence.append(
                    dict(
                        key=key,
                        row_aggregation=col,
                        n=len(matched),
                        spearman=spearmanr(matched.if_score, matched[col]).statistic,
                    )
                )
    save(candidate_daily, "daily_candidates")
    save(correspondence, "row_candidate_correspondence")
    return pd.DataFrame(daily), pd.DataFrame(candidate_daily)


def safe_spearman(a, b, minimum=5):
    mask = pd.notna(a) & pd.notna(b)
    a, b = np.asarray(a)[mask], np.asarray(b)[mask]
    if len(a) < minimum or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)


def quality_comparison(daily, candidate_daily):
    result = pd.read_excel(ROOT / "data" / "Welding_Data_Set_01.xlsx", sheet_name="result")
    result["excel_row"] = np.arange(2, len(result) + 2)
    result["date"] = pd.to_datetime(result["working time"]).dt.strftime("%Y-%m-%d")
    assert not result.duplicated(["date", "defect type"]).any()
    assert len(result) == 23 and result.defect.sum() == 39
    save(result, "result_audit")
    quality = result.pivot(index="date", columns="defect type", values="defect").reindex(daily.date)
    quality.columns = [f"type_{int(c)}" for c in quality.columns]
    quality["recorded_sum"] = quality.sum(axis=1, min_count=1)
    quality["known_types"] = quality[["type_1", "type_2", "type_3"]].notna().sum(axis=1)
    quality["complete_sum"] = quality.recorded_sum.where(quality.known_types == 3)
    combined = daily.merge(quality, on="date", how="left")
    for kind in ("type_1", "type_2", "type_3"):
        combined[f"{kind}_per_record"] = combined[kind] / combined.records
        for r in range(4):
            combined[f"{kind}_per_r{r}_candidate"] = combined[kind] / combined[f"r{r}_groups"]
    save(combined, "daily_quality_comparison")
    save(candidate_daily.merge(quality, on="date", how="left"), "candidate_quality_comparison")
    # Fixed list; no search for the strongest date correlation.
    columns = [
        "records",
        "force_above3_share",
        "force_median",
        "current_median",
        "current_iqr",
        "voltage_abs_diff_median",
        "time_71_share",
        "time_72_share",
        "time_73_share",
        "if_standard_42_flag_count",
        "if_standard_42_flag_share",
        "if_standard_42_score_median",
        "ae_42_flag_count",
        "ae_42_flag_share",
        "ae_42_score_median",
    ]
    correlation = []
    for metric in columns:
        if metric not in combined:
            continue
        for kind in ("type_1", "type_2", "type_3", "complete_sum"):
            for target_form in ("count", "per_record"):
                a = combined[metric]
                b = combined[kind] if target_form == "count" else combined[kind] / combined.records
                valid = a.notna() & b.notna()
                rho = safe_spearman(a, b)
                leave = [
                    safe_spearman(a[valid].drop(i), b[valid].drop(i)) for i in combined.index[valid]
                ]
                finite = [v for v in leave if np.isfinite(v)]
                correlation.append(
                    dict(
                        metric=metric,
                        target=kind,
                        target_form=target_form,
                        n=int(valid.sum()),
                        spearman=rho,
                        leave_one_min=min(finite) if finite else np.nan,
                        leave_one_max=max(finite) if finite else np.nan,
                        scope="all_dates_descriptive",
                    )
                )
    for key, metric in [
        (f"all_r{r}_S8_standard", metric)
        for r in range(4)
        for metric in (
            "if_flag95_count",
            "if_flag95_share",
            "transition_rate",
            "voltage_residual_variance",
            "force_above3_any_share",
        )
    ]:
        part = candidate_daily[candidate_daily.key == key].merge(quality, on="date")
        for kind in ("type_1", "type_2", "type_3", "complete_sum"):
            a, b = part[metric], part[kind]
            mask = a.notna() & b.notna()
            leave = [safe_spearman(a[mask].drop(i), b[mask].drop(i)) for i in part.index[mask]]
            finite = [v for v in leave if np.isfinite(v)]
            correlation.append(
                dict(
                    metric=f"{key}:{metric}",
                    target=kind,
                    target_form="count",
                    n=int(mask.sum()),
                    spearman=safe_spearman(a, b),
                    leave_one_min=min(finite) if finite else np.nan,
                    leave_one_max=max(finite) if finite else np.nan,
                    scope="all_dates_descriptive",
                )
            )
    save(correlation, "quality_correlations")


if __name__ == "__main__":
    raw, segments = load()
    daily, candidate_daily = process_metrics(raw, segments)
    quality_comparison(daily, candidate_daily)
    print("QUALITY COMPARISON COMPLETE", flush=True)
