"""Independent numerical and boundary audit of B outputs."""

from engine import *

checks = []


def check(name, actual, expected):
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-11,equal_nan=True)
    checks.append(name)


def main():
    from order_scores import build
    order_scores = build()
    config,raw,x,segments,keep,median,scale,q = load()
    saved = read("scales")
    check("saved medians",saved["median"],median)
    check("saved scales",saved.scale,scale)
    check("saved observed resolution",saved.observed_resolution,q)
    conflict = raw.duplicated(["date","idx"],keep=False)
    boundaries = (raw.date.ne(raw.date.shift()) | raw.idx.diff().ne(1) |
                  conflict | conflict.shift(fill_value=False)).to_numpy()
    check("independent segment starts",np.flatnonzero(boundaries)+2,segments.excel_start)
    end = np.r_[np.flatnonzero(boundaries)[1:]-1,len(raw)-1]
    check("independent segment ends",end+2,segments.excel_end)
    audit = read("window_audit")
    for row in audit.itertuples():
        check(f"window count {row.Index}",row.windows,max(row.rows-row.width+1,0))
    anchor = read("anchors")
    for row in anchor.itertuples():
        source = segments.iloc[row.segment]
        assert row.pos>=row.burn+row.phase
        assert (row.pos-row.burn-row.phase)%4==0
        assert row.pos+row.width<=int(.6*source.n)
        assert row.excel==source.excel_start+row.pos
    checks.append("all anchor positions and discovery boundaries")
    _, window_meta = windows(x,segments,keep,4)
    rng = np.random.default_rng(42)
    for trial in range(5):
        ids = np.sort(rng.choice(len(window_meta),80,replace=False))
        distance = rng.random(len(window_meta))
        actual = greedy(ids,distance,window_meta,4)
        expected = []
        for index in ids[np.lexsort((window_meta.start.to_numpy()[ids],distance[ids]))]:
            if all(window_meta.segment.iloc[index]!=window_meta.segment.iloc[j] or
                   abs(window_meta.start.iloc[index]-window_meta.start.iloc[j])>=4 for j in expected):
                expected.append(index)
        check(f"cached greedy versus independent intervals {trial}",actual,expected)
    occurrence = read("occurrences")
    for key, group in occurrence.groupby(["family","target_segment"]):
        assert np.all(np.diff(np.sort(group.target_excel.to_numpy()))>=4),key
    checks.append("all selected occurrences are nonoverlapping within original segments")
    assert not occurrence.exact_view.any() and not occurrence.exact4.any()
    assert occurrence.distance.le(.25+1e-12).all()
    checks.append("all saved occurrences exclude exact views and meet fixed threshold")
    samples = occurrence.iloc[np.linspace(0,len(occurrence)-1,min(1500,len(occurrence))).astype(int)]
    for row in samples.itertuples():
        view = VIEWS[row.view]
        a,b = row.anchor_excel-2,row.target_excel-2
        aa,bb = x[a:a+4][:,view],x[b:b+4][:,view]
        if row.representation=="shape":
            aa,bb = aa-aa.mean(axis=0),bb-bb.mean(axis=0)
        elif row.representation=="delta":
            aa,bb = np.diff(aa,axis=0),np.diff(bb,axis=0)
        expected = np.linalg.norm((aa-bb)/scale[view])/np.sqrt(aa.size)
        check(f"direct raw distance {row.Index}",row.distance,expected)
        s = segments.iloc[row.target_segment]
        assert s.excel_start<=row.target_excel and row.target_excel+3<=s.excel_end
        if row.condition=="controlled100":
            assert keep[a:a+4].all() and keep[b:b+4].all()
    summary = read("match_summary")
    main = summary[(summary.setting=="primary")&(summary.threshold==.25)&
                   (summary.exclusion==1)&~summary.initial_excluded]
    counts = occurrence.groupby(["view","representation","anchor_excel","condition"]).size()
    for row in main.itertuples():
        key = row.view,row.representation,row.anchor_excel,row.condition
        check(f"selected count {row.Index}",row.selected,counts.get(key,0))
    catalog = read("motif_catalog")
    expected = catalog.sort_values(["discovery","discovery_median","anchor_excel","view","representation"],
                                  ascending=[False,True,True,True,True])
    assert expected.family.tolist()==catalog.family.tolist()
    checks.append("catalog discovery-only ranking")
    for row in read("representative_raw_values").itertuples():
        check(f"representative raw row {row.Index}",[row.F,row.I,row.V,row.t],x[row.excel-2])
    contexts = read("context_extension")
    selected_context = contexts.iloc[np.linspace(0,len(contexts)-1,min(600,len(contexts))).astype(int)]
    lookup = catalog.set_index("family")
    for row in selected_context.itertuples():
        info = lookup.loc[row.family]
        d,active,reason = context(x,segments,keep,info.anchor_excel-2,row.target_excel-2,
                                 VIEWS[info["view"]],info.representation,median,scale,q,row.length,row.before,
                                 row.condition=="controlled100")
        check(f"context distance {row.Index}",row.distance,d)
        assert row.reason==reason and row.passed==(reason=="ok" and active and d<=.25)
    t = read("null_T_replicates")
    check("T grid",len(t),101*18)
    for row in t[t.replicate==0].itertuples():
        found = order_scores[(order_scores["view"]==row.view)&
                             (order_scores.representation==row.representation)&(order_scores.matches>=3)]
        check(f"T observed max independently recalculated {row.view}/{row.representation}",
              row.max_rate,found.rate.max() if len(found) else 0)
    g = read("null_G_replicates")
    assert (g[g.valid].drawn==g[g.valid].n).all()
    assert (g[~g.valid].drawn<g[~g.valid].n).all()
    check("G each attempted group repeats",g.groupby(["family","segment"]).size().to_numpy(),100)
    g_groups = g.groupby(["family","segment"])
    g_summary = read("null_G_summary")
    subset = g_summary[g_summary.null_valid>0]
    subset = subset.iloc[np.linspace(0,len(subset)-1,min(100,len(subset))).astype(int)]
    for row in subset.itertuples():
        trials = g_groups.get_group((row.family,row.segment))
        valid = trials[trials.valid]
        check(f"G valid count {row.Index}",row.null_valid,len(valid))
        check(f"G peak quantiles {row.Index}",[row.peak_p05,row.peak_median,row.peak_p95],
              valid.max_share.quantile([.05,.5,.95]).to_numpy())
    for row in read("null_summary").itertuples():
        subset = t[(t.view==row.view)&(t.representation==row.representation)&(t.replicate>0)][row.metric].dropna()
        check(f"T quantiles {row.Index}",[row.p05,row.median,row.p95],
              subset.quantile([.05,.5,.95]).to_numpy() if len(subset) else [np.nan]*3)
    for script in ("engine.py","analyze.py","nulls.py"):
        code = (HERE/script).read_text(encoding="utf-8")
        assert "ysh-006a" not in code and 'sheet_name="result"' not in code
    checks.append("B code uses Raw data sheet and no A paths")
    manifest = json.loads((HERE/"outputs/manifest.json").read_text(encoding="utf-8"))
    for name,value in manifest["input_hashes"].items():
        assert digest(ROOT/name)==value,name
    for name,value in manifest["table_hashes"].items():
        assert digest(TABLES/name)==value,name
    null_manifest = json.loads((HERE/"outputs/null_manifest.json").read_text(encoding="utf-8"))
    assert null_manifest["completed"] and null_manifest["replicates"]==100
    assert null_manifest["config_sha256"]==digest(HERE/"config.json")
    for name,value in null_manifest["tables"].items():
        assert digest(TABLES/name)==value,name
    for name,value in null_manifest["code_hashes"].items():
        assert digest(HERE/name)==value,name
    checks.append("completed controls and their source/config/output hashes")
    manifest["verified"] = True
    write_json(manifest,"manifest.json")
    write_json(dict(passed=True,checks=len(checks),details=checks,verify_sha256=digest(__file__)),"verification.json")
    print(f"VERIFIED {len(checks)} numerical groups/checks",flush=True)


if __name__=="__main__":
    main()
