"""Order and conditional-position controls, with fixed seed and 100 replicates."""

from engine import *
import time
import argparse
import pickle


def block_labels(segments, total):
    labels = np.full(total, -1, int)
    blocks = []
    for row in segments.itertuples():
        start = row.excel_start-2
        cut = start+int(.6*row.n)
        end = row.excel_end-1
        for lo,hi in [(start,cut),(cut,end)]:
            for left in range(lo,hi,32):
                right = min(left+32,hi)
                labels[left:right] = len(blocks)
                blocks.append((left,right))
    return labels,blocks


def order_control(config, x, segments, keep, median, scale, q):
    rng = np.random.default_rng(config["seed"])
    labels, blocks = block_labels(segments,len(x))
    original_array, full_meta = windows(x,segments,keep,4)
    anchor_ids = anchors(full_meta,segments,4)
    anchor_starts = set(full_meta.start.iloc[anchor_ids])
    safe = labels[full_meta.start.to_numpy()] == labels[full_meta.start.to_numpy()+3]
    records = []
    started = time.time()
    for replicate in range(config["null_repeats"]+1):
        values = x.copy()
        if replicate:
            for lo,hi in blocks:
                values[lo:hi] = values[lo:hi][rng.permutation(hi-lo)]
        array, meta = windows(values,segments,keep,4)
        best = {(v,r): None for v in VIEWS for r in REPS}
        for segment in range(len(segments)):
            eligible_ids = meta.index[(meta.segment==segment)&meta.front&safe].to_numpy()
            local_meta = meta.iloc[eligible_ids].reset_index(drop=True)
            if not len(local_meta):
                continue
            local = array[eligible_ids]
            local_anchors = np.flatnonzero(local_meta.start.isin(anchor_starts))
            for view_name, view in VIEWS.items():
                for rep in REPS:
                    z = transform(local,view,rep,median,scale)
                    matrix = np.sqrt(cdist(z[local_anchors],z,metric="sqeuclidean")/z.shape[1])
                    for a, distance in zip(local_anchors,matrix):
                        allowed = np.abs(local_meta.pos.to_numpy()-local_meta.pos.iloc[a])>=4
                        active = active_pair(local,a,view,q)
                        exact = np.all(local[:,:,view]==local[a:a+1,:,view],axis=(1,2))
                        selected = greedy(np.flatnonzero(allowed&active&~exact&(distance<=.25)),distance,local_meta,4)
                        if len(selected)<3:
                            continue
                        rate = len(selected)/allowed.sum() if allowed.any() else np.nan
                        median_distance = float(np.median(distance[selected]))
                        key = (rate,-median_distance,-int(local_meta.start.iloc[a]))
                        previous = best[(view_name,rep)]
                        if previous is not None and key<=previous["key"]:
                            continue
                        a_start = int(local_meta.start.iloc[a])
                        successes = valid_context = 0
                        for b in selected:
                            b_start = int(local_meta.start.iloc[b])
                            if a_start+7>=len(x) or b_start+7>=len(x):
                                continue
                            if labels[a_start]!=labels[a_start+7] or labels[b_start]!=labels[b_start+7]:
                                continue
                            d, changing, why = context(values,segments,keep,a_start,b_start,view,rep,median,scale,q)
                            if why=="ok":
                                valid_context += 1
                                successes += changing and d<=.25
                        best[(view_name,rep)] = dict(key=key,max_rate=rate,anchor_excel=a_start+2,
                                                    matches=len(selected),eligible=int(allowed.sum()),
                                                    context_valid=valid_context,context_pass=successes,
                                                    context_rate=successes/valid_context if valid_context else np.nan)
        for (view,rep), outcome in best.items():
            outcome = outcome or dict(max_rate=0.,anchor_excel=np.nan,matches=0,eligible=0,
                                      context_valid=0,context_pass=0,context_rate=np.nan)
            records.append(dict(control="T",replicate=replicate,view=view,representation=rep,
                                **{k:v for k,v in outcome.items() if k!="key"}))
        if replicate%10==0:
            save(records,"null_T_replicates")
            print(f"T {replicate}/100: {time.time()-started:.1f}s",flush=True)
    return pd.DataFrame(records)


def position_control(config,x,segments,keep,resume=False):
    rng = np.random.default_rng(config["seed"])
    occurrences = read("occurrences")
    catalog = read("motif_catalog")
    anchors_by_family = catalog.set_index("family")
    _, meta = windows(x,segments,keep,4)
    records, summaries, all_gaps = [], [], []
    checkpoint_path = HERE / "outputs/G_checkpoint.pkl"
    last_group = -1
    processed = 0
    if resume and checkpoint_path.exists():
        with checkpoint_path.open("rb") as stream:
            checkpoint = pickle.load(stream)
        assert checkpoint["config_sha256"] == digest(HERE/"config.json")
        assert checkpoint["code_sha256"] == digest(__file__)
        records = checkpoint["records"]
        summaries = checkpoint["summaries"]
        all_gaps = checkpoint["all_gaps"]
        last_group = checkpoint["last_group"]
        processed = checkpoint["processed"]
        rng.bit_generator.state = checkpoint["rng"]
        print(f"Resuming G after {processed} attempted groups",flush=True)
    eligible_occ = occurrences[occurrences.condition=="all"]
    groups = eligible_occ.groupby(["family","target_segment"],sort=True)
    started = time.time()
    for group_index,((family, segment), group) in enumerate(groups):
        if group_index<=last_group:
            continue
        anchor = anchors_by_family.loc[family]
        if segment!=anchor.anchor_segment and not anchor.detailed:
            continue
        positions = group.target_pos.to_numpy(int)
        observed, gaps = gap_stats(positions)
        for gap,count in zip(*np.unique(gaps,return_counts=True)):
            all_gaps.append(dict(family=family,segment=segment,gap=gap,count=count))
        if observed["gaps"]<5:
            summaries.append(dict(family=family,segment=segment,n=len(positions),**observed,
                                  null_valid=0,reason="fewer_than_5_gaps"))
            continue
        local_meta = meta[meta.segment==segment].reset_index(drop=True)
        allowed = np.ones(len(local_meta),bool)
        if segment==anchor.anchor_segment:
            allowed &= np.abs(local_meta.start.to_numpy()-(anchor.anchor_excel-2))>=4
        ids = np.flatnonzero(allowed)
        peaks, multiples = [], []
        for replicate in range(1,config["null_repeats"]+1):
            priorities = rng.random(len(local_meta))
            sampled = greedy(ids,priorities,local_meta,4,len(positions))
            valid = len(sampled)==len(positions)
            stat,_ = gap_stats(local_meta.pos.iloc[sampled].to_numpy()) if valid else ({},None)
            peak, multiple = stat.get("max_share",np.nan),stat.get("multiple4",np.nan)
            records.append(dict(family=family,segment=segment,replicate=replicate,n=len(positions),
                                drawn=len(sampled),valid=valid,max_share=peak,multiple4=multiple))
            if valid:
                peaks.append(peak)
                multiples.append(multiple)
        row = dict(family=family,segment=segment,n=len(positions),**observed,
                   null_valid=len(peaks),reason="ok" if peaks else "no_complete_draw")
        for name, values in [("peak",peaks),("multiple4",multiples)]:
            for label,quantile in [("p05",.05),("median",.5),("p95",.95)]:
                row[f"{name}_{label}"] = np.quantile(values,quantile) if values else np.nan
            value = observed["max_share" if name=="peak" else "multiple4"]
            row[f"{name}_rank"] = sum(v<value for v in values)/len(values) if values else np.nan
        summaries.append(row)
        processed += 1
        if processed%100==0:
            print(f"G {processed} families/segments: {time.time()-started:.1f}s",flush=True)
        if processed%250==0:
            checkpoint = dict(config_sha256=digest(HERE/"config.json"),code_sha256=digest(__file__),
                              records=records,summaries=summaries,all_gaps=all_gaps,
                              last_group=group_index,processed=processed,rng=rng.bit_generator.state)
            temporary = checkpoint_path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                pickle.dump(checkpoint,stream,protocol=5)
            temporary.replace(checkpoint_path)
    save(records,"null_G_replicates")
    save(summaries,"null_G_summary")
    save(all_gaps,"all_gap_distribution")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-T",action="store_true")
    args = parser.parse_args()
    config,raw,x,segments,keep,median,scale,q = load()
    if args.resume_T:
        result = read("null_T_replicates")
        assert len(result)==101*18 and result.replicate.max()==100
    else:
        result = order_control(config,x,segments,keep,median,scale,q)
    summaries = []
    for (view,rep), group in result.groupby(["view","representation"]):
        observed = group[group.replicate==0].iloc[0]
        null = group[group.replicate>0]
        for metric in ("max_rate","context_rate"):
            vals = null[metric].dropna().to_numpy()
            summaries.append(dict(control="T",view=view,representation=rep,metric=metric,
                                  observed=observed[metric],valid=len(vals),
                                  p05=np.quantile(vals,.05) if len(vals) else np.nan,
                                  median=np.median(vals) if len(vals) else np.nan,
                                  p95=np.quantile(vals,.95) if len(vals) else np.nan,
                                  rank=np.mean(vals<observed[metric]) if len(vals) and np.isfinite(observed[metric]) else np.nan,
                                  observed_anchor=observed.anchor_excel,observed_context_n=observed.context_valid))
    save(summaries,"null_summary")
    position_control(config,x,segments,keep,resume=args.resume_T)
    write_json(dict(completed=True,seed=config["seed"],replicates=config["null_repeats"],
                    config_sha256=digest(HERE/"config.json"),
                    code_hashes={name:digest(HERE/name) for name in ("engine.py","nulls.py")},
                    J="not run: optional joint alignment control; no claim of physical synchrony",
                    tables={p.name:digest(p) for p in TABLES.glob("null*.csv")}),"null_manifest.json")
    print("NULL CONTROLS COMPLETE",flush=True)


if __name__=="__main__":
    main()
