"""Fixed-grid search and discovery-only motif catalog; no quality inputs."""

from engine import *
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.parse_args()
    config, raw, x, segments, keep, median, scale, q = load()
    started = time.time()
    inputs = [ROOT / name for name in config["inputs"]]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    write_json(dict(input_hashes=hashes, config_sha256=digest(HERE / "config.json"),
                    code_hashes={p.name:digest(p) for p in HERE.glob("*.py")}), "pre_run.json")
    reference = x[raw.date <= "2020-03-31"]
    save([dict(variable=n, median=median[j], iqr=np.ptp(np.quantile(reference[:,j],[.25,.75])),
               scale=scale[j], observed_resolution=q[j], reference_rows=len(reference)) for j,n in enumerate(NAMES)], "scales")
    summaries, occurrence_rows, anchor_rows, audit, contexts = [], [], [], [], []
    for setting in config["settings"]:
        width, burn, phase = setting["width"], setting["burn"], setting["phase"]
        array, meta = windows(x, segments, keep, width)
        ids = anchors(meta, segments, width, burn, phase)
        # Long windows retain w4 anchor positions, then drop invalid extensions.
        if width > 4:
            _, short_meta = windows(x, segments, keep, 4)
            short_ids = anchors(short_meta, segments, 4)
            starts = set(short_meta.start.iloc[short_ids])
            ids = np.array([i for i in meta.index if meta.start.iloc[i] in starts and meta.front.iloc[i]], int)
        for s, segment in segments.iterrows():
            audit.append(dict(setting=setting["name"], width=width, segment=segment.segment,
                              date=segment.date, rows=segment.n, windows=int((meta.segment==s).sum()),
                              controlled_windows=int(((meta.segment==s)&meta.controlled).sum()),
                              anchors=int(np.sum(meta.segment.iloc[ids].to_numpy()==s))))
        for a in ids:
            anchor_rows.append(dict(setting=setting["name"], width=width, burn=burn, phase=phase,
                                    window_index=a, excel=int(meta.start.iloc[a])+2,
                                    idx=int(raw.idx.iloc[meta.start.iloc[a]]),
                                    segment=int(meta.segment.iloc[a]), pos=int(meta.pos.iloc[a]),
                                    date=meta.date.iloc[a], controlled=bool(meta.controlled.iloc[a])))
        for view_name, view in VIEWS.items():
            for rep in REPS:
                z = transform(array, view, rep, median, scale)
                for chunk_start in range(0, len(ids), 32):
                    chunk = ids[chunk_start:chunk_start+32]
                    distances = np.empty((len(chunk),len(meta)))
                    for target in range(0,len(meta),2048):
                        distances[:,target:target+2048] = np.sqrt(
                            cdist(z[chunk],z[target:target+2048],metric="sqeuclidean") / z.shape[1])
                    for a, distance in zip(chunk, distances):
                        cases = [("all",.25,1,False)]
                        if setting["name"] == "primary":
                            cases += [("all",.125,1,False),("all",.5,1,False),
                                      ("all",.25,2,False),("all",.25,1,True),
                                      ("controlled100",.25,1,False),
                                      ("controlled100",.125,1,False),("controlled100",.5,1,False)]
                        for condition, threshold, exclusion, initial in cases:
                            eligible, matches, selected, ev, e4, near, active = compare(
                                array, meta, a, view, rep, distance, q, threshold,
                                exclusion, condition=="controlled100", initial)
                            same = meta.segment.to_numpy()==meta.segment.iloc[a]
                            discovery = same & meta.front.to_numpy()
                            later = same & meta.back.to_numpy()
                            identity = dict(setting=setting["name"], width=width, view=view_name, representation=rep,
                                            anchor_excel=int(meta.start.iloc[a])+2, anchor_segment=int(meta.segment.iloc[a]),
                                            anchor_date=meta.date.iloc[a], condition=condition, threshold=threshold,
                                            exclusion=exclusion, initial_excluded=initial)
                            discovery_indices = np.flatnonzero(matches & ~ev & active & discovery)
                            selected_discovery = greedy(discovery_indices, distance, meta, width)
                            row = dict(identity, eligible=int(eligible.sum()), matches=int(matches.sum()),
                                       exact4=int((matches&e4).sum()), exact_view=int((matches&ev).sum()),
                                       near_exact4=int((matches&near).sum()), low_change=int((matches&~active).sum()),
                                       nonexact_raw=int((matches&~ev&active).sum()), selected=len(selected),
                                       discovery=len(selected_discovery),
                                       discovery_eligible=int((eligible&discovery).sum()),
                                       discovery_median=float(np.median(distance[selected_discovery])) if len(selected_discovery) else np.nan,
                                       same_back=int(later[selected].sum()),
                                       other_segment=int((~same[selected]).sum()),
                                       other_date=int((meta.date.iloc[selected].to_numpy()!=meta.date.iloc[a]).sum()),
                                       initial_targets=int((meta.pos.iloc[selected]<100).sum()),
                                       temporal_confirmation=int(((meta.date.iloc[selected]>"2020-03-31") &
                                                                   (meta.date.iloc[a]<="2020-03-31")).sum()))
                            summaries.append(row)
                            primary = (setting["name"]=="primary" and threshold==.25 and exclusion==1 and not initial)
                            if primary:
                                family = f"{view_name}_{rep}_{identity['anchor_excel']}_{condition}"
                                for b in selected:
                                    start_b = int(meta.start.iloc[b])
                                    occurrence_rows.append(dict(family=family, view=view_name, representation=rep,
                                        condition=condition, anchor_excel=identity["anchor_excel"],
                                        target_excel=start_b+2, target_idx=int(raw.idx.iloc[start_b]),
                                        target_segment=int(meta.segment.iloc[b]), target_date=meta.date.iloc[b],
                                        target_pos=int(meta.pos.iloc[b]), distance=distance[b],
                                        discovery=bool(discovery[b]), same_back=bool(later[b]),
                                        initial=bool(meta.pos.iloc[b]<100), exact4=bool(e4[b]),
                                        exact_view=bool(ev[b]), near_exact4=bool(near[b]),
                                        pair_id=f"{min(identity['anchor_excel'],start_b+2)}:{max(identity['anchor_excel'],start_b+2)}"))
        print(f"Search {setting['name']} complete: {len(ids)} anchors, {time.time()-started:.1f}s",flush=True)
    save(audit,"window_audit")
    save(anchor_rows,"anchors")
    save(summaries,"match_summary")
    save(occurrence_rows,"occurrences")
    summary = pd.DataFrame(summaries)
    occurrences = pd.DataFrame(occurrence_rows)
    base = summary[(summary.setting=="primary") & (summary.condition=="all") &
                   (summary.threshold==.25) & (summary.exclusion==1) & ~summary.initial_excluded].copy()
    base["family"] = [f"{r.view}_{r.representation}_{r.anchor_excel}_all" for r in base.itertuples()]
    # Rank only by discovery scores, never by later or cross-date outcomes.
    base = base.sort_values(["discovery","discovery_median","anchor_excel","view","representation"],
                            ascending=[False,True,True,True,True])
    base["discovery_rank"] = np.arange(1,len(base)+1)
    selected, used = [], []
    for row in base.itertuples():
        if row.discovery < 3:
            continue
        if any(abs(row.anchor_excel-a)<4 for a in used):
            continue
        selected.append(row.family)
        used.append(row.anchor_excel)
        if len(selected)==20:
            break
    base["detailed"] = base.family.isin(selected)
    save(base,"motif_catalog")
    write_json(dict(families=selected, rule="discovery-only, nonoverlap, max20 text examples",
                    selection_sha256=digest(TABLES/"motif_catalog.csv")),"catalog_frozen.json")
    print("Discovery-only catalog frozen",flush=True)
    blocks = pd.read_csv(HERE.parent / "ysh-001/outputs/exact_repeated_blocks.csv")
    raw_values, details, lineage, gaps, gap_summary, force = [], [], [], [], [], []
    for row in base[base.detailed].itertuples():
        view = VIEWS[row.view]
        a_start = row.anchor_excel-2
        selected_rows = occurrences[occurrences.family==row.family]
        for offset, values in enumerate(x[a_start:a_start+4]):
            raw_values.append(dict(family=row.family, excel=a_start+offset+2,
                                   **dict(zip(NAMES,values))))
        for match in selected_rows.itertuples():
            b_start = match.target_excel-2
            a_raw, b_raw = x[a_start:a_start+4], x[b_start:b_start+4]
            for j in view:
                aa, bb = a_raw[:,j], b_raw[:,j]
                correlation = np.corrcoef(aa,bb)[0,1] if np.ptp(aa)>0 and np.ptp(bb)>0 else np.nan
                cosine = np.dot(aa,bb)/(np.linalg.norm(aa)*np.linalg.norm(bb)) if np.linalg.norm(aa)*np.linalg.norm(bb)>0 else np.nan
                details.append(dict(family=row.family, anchor_excel=row.anchor_excel,target_excel=match.target_excel,
                                    variable=NAMES[j], distance=match.distance, raw_rmse=np.sqrt(np.mean((aa-bb)**2)),
                                    max_error=np.max(np.abs(aa-bb)), mean_difference=bb.mean()-aa.mean(),
                                    range_ratio=np.ptp(bb)/np.ptp(aa) if np.ptp(aa)>0 else np.nan,
                                    correlation=correlation, cosine=cosine,
                                    shape_exact_raw_nonexact=bool(np.array_equal(aa-aa.mean(),bb-bb.mean()) and not np.array_equal(aa,bb))))
            for length, before in [(8,0),(16,0),(8,2)]:
                for condition in ("all","controlled100"):
                    d, active, reason = context(x,segments,keep,a_start,b_start,view,row.representation,
                                               median,scale,q,length,before,condition=="controlled100")
                    contexts.append(dict(family=row.family,target_excel=match.target_excel,length=length,before=before,
                                         condition=condition,distance=d,active=active,reason=reason,
                                         passed=bool(reason=="ok" and active and d<=.25)))
            a_member = blocks[(blocks.length>=100) & (((blocks.a_start<=row.anchor_excel)&(blocks.a_end>=row.anchor_excel+3)) |
                                                     ((blocks.b_start<=row.anchor_excel)&(blocks.b_end>=row.anchor_excel+3)))]
            b_member = blocks[(blocks.length>=100) & (((blocks.a_start<=match.target_excel)&(blocks.a_end>=match.target_excel+3)) |
                                                     ((blocks.b_start<=match.target_excel)&(blocks.b_end>=match.target_excel+3)))]
            lineage.append(dict(family=row.family,target_excel=match.target_excel,
                                anchor_long_blocks=";".join(map(str,a_member.index)),
                                target_long_blocks=";".join(map(str,b_member.index)),
                                controlled_pair=bool(keep[a_start:a_start+4].all() and keep[b_start:b_start+4].all())))
        for segment, group in selected_rows.groupby("target_segment"):
            stats, values = gap_stats(group.target_pos.to_numpy())
            gap_summary.append(dict(family=row.family, segment=segment, **stats))
            unique, count = np.unique(values,return_counts=True)
            for gap,n in zip(unique,count):
                gaps.append(dict(family=row.family,segment=segment,gap=gap,count=n,tail=gap>256))
        _, m = windows(x,segments,keep,4)
        for cutoff in (2.8,3.0,3.2):
            for population, starts in [("occurrences",selected_rows.target_excel.to_numpy()-2),("all_windows",m.start.to_numpy())]:
                count = (x[starts[:,None]+np.arange(4),0]>cutoff).sum(axis=1)
                force.append(dict(family=row.family,cutoff=cutoff,population=population,n=len(starts),
                                  full=int((count==4).sum()),partial=int(((count>0)&(count<4)).sum()),none=int((count==0).sum())))
    save(raw_values,"representative_raw_values")
    save(details,"selected_match_details")
    save(contexts,"context_extension")
    save(lineage,"exact_lineage")
    save(gaps,"gap_distribution")
    save(gap_summary,"gap_summary")
    save(force,"force_context")
    opportunities = []
    for width in (4,8,16):
        for lag in range(1,int(segments.n.max())-width+1):
            opportunities.append(dict(width=width,lag=lag,valid_pairs=int(np.maximum(segments.n-width+1-lag,0).sum())))
    save(opportunities,"lag_opportunities")
    save(summary[[c for c in summary if c not in ["exact4","exact_view","near_exact4"]]],"settings_sensitivity")
    save(base[["family","anchor_date","discovery","same_back","other_date","temporal_confirmation"]],"confirmation")
    assert hashes == {str(p.relative_to(ROOT)):digest(p) for p in inputs}
    write_json(dict(computed=True,verified=False,seconds=time.time()-started,
                    input_hashes=hashes,code_hashes={p.name:digest(p) for p in HERE.glob("*.py")},
                    config_sha256=digest(HERE/"config.json"), read_sheets=["Raw data"],
                    quality_inputs=False, a_outputs_read=False,
                    table_hashes={p.name:digest(p) for p in TABLES.glob("*.csv")}),"manifest.json")
    print(f"SEARCH COMPLETE {len(summary)} summaries; {len(occurrences)} nonoverlap occurrences",flush=True)


if __name__ == "__main__":
    main()
