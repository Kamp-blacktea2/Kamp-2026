"""Independent arithmetic checks against raw inputs and saved tables."""

from common import *


checks = []


def check(name, actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12, equal_nan=True)
    checks.append(name)


def rank_correlation(x, y):
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pair) < 5 or pair.x.nunique() < 2 or pair.y.nunique() < 2:
        return np.nan
    ranks = pair.rank(method="average").to_numpy()
    return np.corrcoef(ranks.T)[0, 1]


def main():
    raw = pd.read_excel(RAW, sheet_name="Raw data")
    p, tr, daily = read("row_proxies"), read("transitions"), read("daily_features")
    quality, correlations = read("daily_quality"), read("candidate_correlations")
    config = load_config()
    check("raw values and row mapping", p[BASE], raw[COLS])
    check("Excel rows", p.excel_row, np.arange(2, 11941))
    f, i, v, t = raw[COLS].to_numpy().T
    expected = dict(P=v*i*1000, E=v*i*t, H=i*i*t*1000,
                    R=v/i/1000, E_F=v*i*t/f, H_F=i*i*t*1000/f)
    for name, values in expected.items():
        check(f"SI formula {name}", p[name], values)
    dates = pd.to_datetime(raw["working time"]).dt.strftime("%Y-%m-%d")
    valid = (dates == dates.shift()) & (raw.idx.diff() == 1)
    conflict = raw.duplicated(["working time", "idx"], keep=False)
    valid &= ~conflict & ~conflict.shift(fill_value=False)
    check("independently reconstructed adjacency", tr.current_excel, np.flatnonzero(valid)+2)
    for name in DIFFERENCES:
        check(f"original adjacent differences {name}", tr[name], p[name].diff()[valid])
    result = pd.read_excel(RAW, sheet_name="result")
    for row in result[["working time", "defect", "defect type"]].itertuples(index=False, name=None):
        date, count, kind = row
        date = pd.Timestamp(date).strftime("%Y-%m-%d")
        saved = quality.set_index("date").loc[date]
        check(f"Result {date} type{kind}", saved[f"type{kind}"], count)
    check("missing cells", quality[["type1", "type2", "type3"]].isna().sum(), [1,1,2])
    check("explicit zero", quality.set_index("date").loc["2020-04-07", "type1"], 0)
    for kind in (1,2,3):
        check(f"original denominator type{kind}", quality[f"type{kind}_per1000"],
              quality[f"type{kind}"]/quality.original_records*1000)
    thresholds = read("reference_thresholds")
    for mode, group in daily.groupby(["condition", "reference", "tail"]):
        condition, scope, tail = mode
        rows, diffs = p, tr
        if condition == "unique4":
            rows = p.drop_duplicates(["date"]+BASE)
            diffs = tr.drop_duplicates(["date", "previous_id", "current_id"])
        elif condition == "controlled100":
            rows, diffs = p[p.controlled100], tr[tr.controlled100]
        ref_rows, ref_diff = (p, tr) if condition == "controlled100" else (rows, diffs)
        if scope == "train":
            ref_rows, ref_diff = ref_rows[ref_rows.reference_period], ref_diff[ref_diff.reference_period]
        saved_thresholds = thresholds[(thresholds.condition==condition) &
                                      (thresholds.reference==scope) & (thresholds["tail"]==tail)].set_index("feature")
        for name in FEATURES:
            lo, hi = ref_rows[name].quantile([tail,1-tail]).to_numpy()
            check(f"threshold {mode} {name}", saved_thresholds.loc[name,["low","high"]].to_numpy(float), [lo,hi])
        for row in group.to_dict("records"):
            values = rows[rows.date==row["date"]]
            delta = diffs[diffs.date==row["date"]]
            for name in FEATURES:
                x = values[name]
                check(f"median/MAD {mode} {row['date']} {name}",
                      [row[name+"__median"],row[name+"__mad"]], [x.median(),(x-x.median()).abs().median()])
                lo,hi = saved_thresholds.loc[name,["low","high"]]
                check(f"tail {mode} {row['date']} {name}",
                      [row[name+"__low_share"],row[name+"__high_share"]],[(x<lo).mean(),(x>hi).mean()])
                if name in DIFFERENCES:
                    threshold = ref_diff[name].abs().quantile(1-tail)
                    check(f"jump {mode} {row['date']} {name}", row[name+"__jump_share"],
                          (delta[name].abs()>threshold).mean())
            hf = values.F > saved_thresholds.loc["F","high"]
            for metric, name, side in [("highF_highE","E","high"),("highF_lowE","E","low"),("highF_lowH","H","low")]:
                boundary = saved_thresholds.loc[name,side]
                mask = values[name]>boundary if side=="high" else values[name]<boundary
                check(f"joint {mode} {row['date']} {metric}",row[metric],(hf & mask).mean())
    merged = daily.merge(quality.drop(columns="original_records"), on="date")
    for row in correlations.itertuples():
        group = merged[(merged.condition==row.condition) & (merged.reference==row.reference) & (merged["tail"]==row.tail)]
        target = f"type{row.defect_type}" + ("_per1000" if row.target=="per1000" else "")
        check(f"rank correlation {row.Index}",row.rho,rank_correlation(group[row.metric],group[target]))
    loo = read("leave_one_date_out")
    for row in loo.itertuples():
        group = merged[(merged.condition==row.condition) & (merged.reference==row.reference) &
                       (merged["tail"]==row.tail) & (merged.date!=row.excluded_date)]
        target = f"type{row.defect_type}" + ("_per1000" if row.target=="per1000" else "")
        check(f"leave-date rank correlation {row.Index}",row.rho,rank_correlation(group[row.metric],group[target]))
    check("complete candidate grid",len(correlations),600)
    check("controlled correlation unavailable",correlations[correlations.condition=="controlled100"].rho.notna().sum(),0)
    for candidate in read("type_candidates").itertuples():
        rows = correlations[correlations.candidate == candidate.candidate]
        main_rows = rows[(rows.condition == "all") & (rows.reference == "train") & (rows["tail"] == .1)]
        count = main_rows[main_rows.target == "count"].iloc[0]
        weighted = rows[(rows.condition == "unique4") & (rows.reference == "train") & (rows["tail"] == .1)]
        sensitivities = rows[(rows.condition == "all") & ((rows.reference == "full") | (rows["tail"] == .05))]
        stable = count.n >= 7 and count.rho >= .5
        stable &= bool((main_rows.rho > 0).all() and (main_rows.sign_fraction >= .8).all())
        stable &= bool(main_rows.influential_dates.isna().all())
        stable &= bool(weighted.rho.notna().all() and (weighted.rho > 0).all())
        stable &= not bool((sensitivities.rho < 0).any())
        check(f"candidate classification {candidate.candidate}",
              candidate.status == "상대적으로 일관된 후속 후보", stable)
    old = pd.read_csv(HERE.parent / "ysh-005/outputs/tables/daily_quality_comparison.csv")
    for kind in (1,2,3):
        check(f"YSH005 quality type{kind}",quality.set_index("date")[f"type{kind}"],old.set_index("date")[f"type_{kind}"])
    manifest_path = HERE / "outputs/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, value in manifest["table_hashes"].items():
        assert digest(TABLES/name)==value, name
    for name in ("analyze.py","common.py"):
        assert digest(HERE/name)==manifest["code_hashes"][name]
    manifest["verified"] = True
    json_save(manifest,manifest_path)
    json_save(dict(passed=True, checks=len(checks), details=checks,
                   verify_sha256=digest(__file__), tolerance="rtol=1e-11, atol=1e-12"),HERE/"outputs/verification.json")
    print(f"VERIFIED {len(checks)} checks")


if __name__ == "__main__":
    main()
