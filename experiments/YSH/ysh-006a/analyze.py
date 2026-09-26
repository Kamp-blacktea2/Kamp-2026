"""Process features are frozen and saved before aggregate quality is loaded."""

from common import *
import argparse
import itertools
import platform
import time
from scipy.stats import spearmanr


def process_features(config, process, transition):
    reference_rows, distribution, instability, daily = [], [], [], []
    original_counts = process.groupby("date").size()
    for condition in config["conditions"]:
        subset, differences = select_condition(process, transition, condition)
        # controlled100 keeps the all-data reference, rather than refitting on its altered population.
        reference_condition = "all" if condition == "controlled100" else condition
        reference_process, reference_diff = select_condition(process, transition, reference_condition)
        for reference in config["reference_scopes"]:
            rp = reference_process if reference == "full" else reference_process[reference_process.reference_period]
            rd = reference_diff if reference == "full" else reference_diff[reference_diff.reference_period]
            centers = {f: float(np.median(rp[f])) for f in FEATURES}
            for tail in config["tail_quantiles"]:
                thresholds = {f: np.quantile(rp[f], [tail, 1 - tail], method="linear") for f in FEATURES}
                jumps = {f: float(np.quantile(np.abs(rd[f]), 1 - tail, method="linear")) for f in DIFFERENCES}
                mode = dict(condition=condition, reference=reference, tail=tail)
                for feature in FEATURES:
                    lo, hi = thresholds[feature]
                    reference_rows.append(dict(mode, feature=feature, reference_condition=reference_condition,
                                               reference_records=len(rp), reference_transitions=len(rd),
                                               median=centers[feature], low=lo, high=hi,
                                               jump=jumps.get(feature, np.nan),
                                               low_ties=int((rp[feature] == lo).sum()), high_ties=int((rp[feature] == hi).sum())))
                for date, part in subset.groupby("date", sort=True):
                    delta = differences[differences.date == date]
                    record = dict(mode, date=date, original_records=int(original_counts.loc[date]),
                                  distribution_records=len(part), transitions=len(delta),
                                  coverage=len(part) / original_counts.loc[date],
                                  split="reference" if date <= config["reference_last_date"] else "later")
                    for feature in FEATURES:
                        vals = part[feature].to_numpy()
                        stats = summary(vals)
                        lo, hi = thresholds[feature]
                        stats.update(low_share=float(np.mean(vals < lo)), high_share=float(np.mean(vals > hi)),
                                     reference_deviation=float(np.median(vals) - centers[feature]),
                                     low_ties=int(np.sum(vals == lo)), high_ties=int(np.sum(vals == hi)))
                        distribution.append(dict(mode, date=date, feature=feature, unit=UNITS[feature], n=len(vals), **stats))
                        for name, value in stats.items():
                            record[f"{feature}__{name}"] = value
                        if feature in DIFFERENCES:
                            values = delta[feature].to_numpy()
                            absolute = np.abs(values)
                            stats_delta = dict(
                                abs_median=np.median(absolute) if len(values) else np.nan,
                                abs_mean=np.mean(absolute) if len(values) else np.nan,
                                abs_p90=np.quantile(absolute, .9, method="linear") if len(values) else np.nan,
                                abs_max=np.max(absolute) if len(values) else np.nan,
                                delta_std=np.std(values, ddof=0) if len(values) else np.nan,
                                jump_share=np.mean(absolute > jumps[feature]) if len(values) else np.nan,
                            )
                            instability.append(dict(mode, date=date, feature=feature, n_pairs=len(values),
                                                    threshold=jumps[feature], **stats_delta))
                            for name, value in stats_delta.items():
                                record[f"{feature}__{name}"] = value
                    high_force = part.F.to_numpy() > thresholds["F"][1]
                    record["highF_highE"] = np.mean(high_force & (part.E.to_numpy() > thresholds["E"][1]))
                    record["highF_lowE"] = np.mean(high_force & (part.E.to_numpy() < thresholds["E"][0]))
                    record["highF_lowH"] = np.mean(high_force & (part.H.to_numpy() < thresholds["H"][0]))
                    daily.append(record)
    save(reference_rows, "reference_thresholds")
    save(distribution, "daily_distribution")
    save(instability, "daily_instability")
    save(daily, "daily_features")
    return pd.DataFrame(daily)


def load_quality(process):
    result = pd.read_excel(RAW, sheet_name="result")
    result["result_excel_row"] = np.arange(2, len(result) + 2)
    result["date"] = pd.to_datetime(result["working time"]).dt.strftime("%Y-%m-%d")
    assert len(result) == 23 and result.defect.sum() == 39
    assert not result.duplicated(["date", "defect type"]).any()
    assert set(result["defect type"]) == {1, 2, 3}
    assert (result.defect >= 0).all() and np.all(result.defect == result.defect.astype(int))
    save(result, "result_audit")
    quality = result.pivot(index="date", columns="defect type", values="defect")
    quality.columns = [f"type{int(c)}" for c in quality.columns]
    quality = quality.reindex(sorted(process.date.unique()))
    quality["original_records"] = process.groupby("date").size()
    quality["known_types"] = quality[["type1", "type2", "type3"]].notna().sum(axis=1)
    quality["recorded_sum"] = quality[["type1", "type2", "type3"]].sum(axis=1, min_count=1)
    for kind in (1, 2, 3):
        quality[f"type{kind}_per1000"] = quality[f"type{kind}"] / quality.original_records * 1000
    assert quality.type1.notna().sum() == 8 and quality.type2.notna().sum() == 8 and quality.type3.notna().sum() == 7
    assert np.isnan(quality.loc["2020-03-27", "type1"]) and np.isnan(quality.loc["2020-03-31", "type3"])
    assert quality.loc["2020-04-07", "type1"] == 0
    save(quality.reset_index(), "daily_quality")
    return quality.reset_index()


def rho(a, b, minimum=5):
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = np.asarray(a)[valid], np.asarray(b)[valid]
    if len(a) < minimum:
        return np.nan, "fewer_than_5_dates"
    if np.ptp(a) == 0:
        return np.nan, "constant_feature"
    if np.ptp(b) == 0:
        return np.nan, "constant_target"
    return float(spearmanr(a, b).statistic), "ok"


def evaluate_candidates(config, daily, quality):
    comparisons, leave_rows, contrasts, counterexamples = [], [], [], []
    merged = daily.merge(quality.drop(columns="original_records"), on="date", how="left", validate="many_to_one")
    save(merged, "daily_quality_comparison")
    for mode, group in merged.groupby(["condition", "reference", "tail"], sort=False):
        condition, reference, tail = mode
        for candidate in config["candidates"]:
            for target in ("count", "per1000"):
                column = f"type{candidate['type']}" + ("_per1000" if target == "per1000" else "")
                a, b = group[candidate["metric"]].to_numpy(float), group[column].to_numpy(float)
                dates = group.date.to_numpy()
                valid = np.isfinite(a) & np.isfinite(b)
                av, bv, dv = a[valid], b[valid], dates[valid]
                correlation, reason = rho(av, bv)
                identity = dict(condition=condition, reference=reference, tail=tail,
                                candidate=candidate["id"], defect_type=candidate["type"],
                                metric=candidate["metric"], target=target)
                leave_values, weak_dates = [], []
                for index, date in enumerate(dv):
                    keep = np.arange(len(av)) != index
                    value, why = rho(av[keep], bv[keep])
                    leave_rows.append(dict(identity, excluded_date=date, n=int(keep.sum()), rho=value, reason=why))
                    leave_values.append(value)
                    if not np.isfinite(value) or abs(value) < .2:
                        weak_dates.append(date)
                finite = np.array([v for v in leave_values if np.isfinite(v)])
                matches = int(np.sum((np.sign(finite) == np.sign(correlation)) & (finite != 0))) if np.isfinite(correlation) else 0
                comparisons.append(dict(identity, n=len(av), rho=correlation, reason=reason,
                                        loo_valid=len(finite), loo_total=len(leave_values),
                                        loo_min=finite.min() if len(finite) else np.nan,
                                        loo_median=np.median(finite) if len(finite) else np.nan,
                                        loo_max=finite.max() if len(finite) else np.nan,
                                        sign_matches=matches,
                                        sign_fraction=matches / len(finite) if len(finite) else np.nan,
                                        influential_dates=";".join(weak_dates)))
                if condition == "all" and reference == "train" and tail == .1:
                    if len(bv):
                        low, high = np.quantile(bv, [.25, .75], method="linear")
                        lower, upper = bv <= low, bv >= high
                        valid_groups = lower.sum() >= 2 and upper.sum() >= 2 and not np.any(lower & upper)
                        contrasts.append(dict(identity, low_dates=";".join(dv[lower]), high_dates=";".join(dv[upper]),
                                              n_low=int(lower.sum()), n_high=int(upper.sum()),
                                              high_minus_low=np.median(av[upper]) - np.median(av[lower]) if valid_groups else np.nan,
                                              reason="ok" if valid_groups else "overlap_or_fewer_than_2_dates"))
                        for i, j in itertools.combinations(range(len(av)), 2):
                            if (av[i] - av[j]) * (bv[i] - bv[j]) < 0:
                                counterexamples.append(dict(identity, date_a=dv[i], date_b=dv[j],
                                                            feature_a=av[i], feature_b=av[j], target_a=bv[i], target_b=bv[j]))
    comparison = pd.DataFrame(comparisons)
    save(comparison, "candidate_correlations")
    save(leave_rows, "leave_one_date_out")
    save(contrasts, "high_low_dates")
    save(counterexamples, "counterexamples")
    return comparison


def classify(config, correlations):
    results = []
    primary = correlations[(correlations.condition == "all") & (correlations.reference == "train") & (correlations["tail"] == .1)]
    for candidate in config["candidates"]:
        count = primary[(primary.candidate == candidate["id"]) & (primary.target == "count")].iloc[0]
        normalized = primary[(primary.candidate == candidate["id"]) & (primary.target == "per1000")].iloc[0]
        weighted = correlations[(correlations.candidate == candidate["id"]) & (correlations.condition == "unique4") &
                                (correlations.reference == "train") & (correlations["tail"] == .1)]
        sensitivity = correlations[(correlations.candidate == candidate["id"]) & (correlations.condition == "all") &
                                   ((correlations.reference == "full") | (correlations["tail"] == .05))]
        reasons = []
        if count.n < 7:
            reasons.append("fewer_than_7_dates")
        if not np.isfinite(count.rho) or count.rho < .5:
            reasons.append("count_rho_below_positive_0.5")
        for label, row in (("count", count), ("per1000", normalized)):
            if not np.isfinite(row.rho) or row.rho <= 0:
                reasons.append(f"{label}_nonpositive_or_NA")
            if not np.isfinite(row.sign_fraction) or row.sign_fraction < .8:
                reasons.append(f"{label}_loo_sign_below_0.8")
            if row.influential_dates:
                reasons.append(f"{label}_one_date_influence")
        if not (np.isfinite(weighted.rho).all() and (weighted.rho > 0).all()):
            reasons.append("weighted_nonpositive_or_NA")
        if (sensitivity.rho.dropna() < 0).any():
            reasons.append("reference_or_tail_sign_reversal")
        if not np.isfinite(count.rho):
            status = "판단 불가"
        elif not reasons:
            status = "상대적으로 일관된 후속 후보"
        elif max(count.rho, normalized.rho if np.isfinite(normalized.rho) else -1, weighted.rho.max()) >= .5:
            status = "조건 의존 후보"
        else:
            status = "지지 부족"
        row = dict(candidate=candidate["id"], defect_type=candidate["type"], metric=candidate["metric"],
                   category=candidate["category"], n=count.n, rho_count=count.rho, rho_per1000=normalized.rho,
                   loo_min=count.loo_min, loo_max=count.loo_max, loo_sign_matches=count.sign_matches,
                   loo_valid=count.loo_valid, loo_total=count.loo_total,
                   influential_dates_count=count.influential_dates, influential_dates_per1000=normalized.influential_dates,
                   rho_weighted_count=weighted[weighted.target == "count"].rho.iloc[0],
                   rho_weighted_per1000=weighted[weighted.target == "per1000"].rho.iloc[0],
                   status=status, failed_criteria=";".join(reasons))
        results.append(row)
    save(results, "type_candidates")


def baseline_and_lineage(config, daily, quality, process):
    primary = daily[(daily.condition == "all") & (daily.reference == "train") & (daily["tail"] == .1)]
    primary = primary.merge(quality.drop(columns="original_records"), on="date", how="left")
    baseline = []
    for feature, stat, kind, target in itertools.product(BASE, ("median", "mad", "jump_share"), (1, 2, 3), ("count", "per1000")):
        col = f"type{kind}" + ("_per1000" if target == "per1000" else "")
        coefficient, reason = rho(primary[f"{feature}__{stat}"].to_numpy(), primary[col].to_numpy())
        baseline.append(dict(metric=f"{feature}__{stat}", defect_type=kind, target=target, rho=coefficient, reason=reason))
    save(baseline, "raw_baseline")
    blocks = pd.read_csv(HERE.parent / "ysh-001/outputs/exact_repeated_blocks.csv")
    blocks = blocks[(blocks.length >= 100) & (blocks.a_date != blocks.b_date)].copy()
    blocks = blocks.sort_values("length", ascending=False).drop_duplicates(["a_date", "b_date"])
    merged = blocks.merge(quality.add_prefix("a_"), on="a_date", how="left")
    merged = merged.merge(quality.add_prefix("b_"), on="b_date", how="left")
    save(merged, "repeated_blocks_quality")
    # No target is attached to a raw row; quality is only compared at the date level.
    algebra = []
    for a, b in itertools.combinations(FEATURES, 2):
        algebra.append(dict(a=a, b=b, row_spearman=spearmanr(process[a], process[b]).statistic,
                            daily_median_spearman=rho(primary[f"{a}__median"].to_numpy(), primary[f"{b}__median"].to_numpy())[0]))
    save(algebra, "feature_dependence")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    start = time.time()
    before = {str(p.relative_to(ROOT)): digest(p) for p in (
        RAW, ROOT / "docs/data_dictionary.md", ROOT / "docs/공정방식.md", ROOT / "data/용접방식.png",
        ROOT / "requirements.txt", HERE / "실험계획.md", HERE / "config.json",
        HERE.parent / "ysh-003/outputs/tables/segments.csv",
        HERE.parent / "ysh-001/outputs/exact_repeated_blocks.csv")}
    json_save({"input_hashes": before, "code_hashes": {p.name: digest(p) for p in HERE.glob("*.py")}}, HERE / "outputs/pre_run.json")
    process, transition, segments = load_process(config)
    save(process[["excel_row", "idx", "date", "segment", "tuple_id", "reference_period", "controlled100"] + FEATURES], "row_proxies")
    save(transition, "transitions")
    save([dict(condition=c, records=len(select_condition(process, transition, c)[0]),
               transitions=len(select_condition(process, transition, c)[1]),
               dates=select_condition(process, transition, c)[0].date.nunique()) for c in config["conditions"]], "input_audit")
    save([dict(feature=f, unit=UNITS[f], formula=config["formulas"][f]) for f in FEATURES], "proxy_definitions")
    daily = process_features(config, process, transition)
    feature_hash = digest(TABLES / "daily_features.csv")
    print("Process features frozen before reading Result", flush=True)
    quality = load_quality(process)
    correlations = evaluate_candidates(config, daily, quality)
    classify(config, correlations)
    baseline_and_lineage(config, daily, quality, process)
    assert feature_hash == digest(TABLES / "daily_features.csv")
    import scipy
    manifest = dict(computed=True, verified=False, seconds=time.time() - start, python=platform.python_version(),
                    numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                    config_sha256=digest(HERE / "config.json"), process_features_before_result_sha256=feature_hash,
                    input_hashes_before=before, input_hashes_after={p: digest(ROOT / p) for p in before},
                    code_hashes={p.name: digest(p) for p in HERE.glob("*.py")},
                    table_hashes={p.name: digest(p) for p in TABLES.glob("*.csv")},
                    result_used_only_at_date_level=True, b_outputs_read=False)
    assert manifest["input_hashes_before"] == manifest["input_hashes_after"]
    json_save(manifest, HERE / "outputs/manifest.json")
    print(f"ANALYSIS COMPLETE: {len(correlations)} comparisons, {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()

