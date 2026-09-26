"""All-candidate context and gap summaries; representative detail is not a sample filter."""

from engine import *


def main():
    config,raw,x,segments,keep,median,scale,q = load()
    occurrences = read("occurrences")
    contexts, gaps, distributions, coverage, wide = [], [], [], [], []
    bounds_lo = segments.excel_start.to_numpy()-2
    bounds_hi = segments.excel_end.to_numpy()-1
    row_segment = np.empty(len(x),int)
    for i,(lo,hi) in enumerate(zip(bounds_lo,bounds_hi)):
        row_segment[lo:hi] = i
    prefix = np.r_[0,np.cumsum(~keep)]
    for family,group in occurrences.groupby("family",sort=False):
        first = group.iloc[0]
        a = int(first.anchor_excel)-2
        starts = group.target_excel.to_numpy(int)-2
        view = VIEWS[first["view"]]
        for length,before in [(8,0),(16,0),(8,2)]:
            a_start = a-before
            b_starts = starts-before
            a_segment = row_segment[a]
            b_segments = row_segment[starts]
            valid = ((a_start>=bounds_lo[a_segment]) & (a_start+length<=bounds_hi[a_segment]) &
                     (b_starts>=bounds_lo[b_segments]) & (b_starts+length<=bounds_hi[b_segments]))
            if first.condition=="controlled100":
                for index in np.flatnonzero(valid):
                    b = b_starts[index]
                    valid[index] &= prefix[a_start+length]==prefix[a_start] and prefix[b+length]==prefix[b]
            selected = b_starts[valid]
            passed = 0
            if len(selected):
                aa = x[a_start:a_start+length][None,:,:]
                bb = x[selected[:,None]+np.arange(length)]
                za = transform(aa,view,first.representation,median,scale)
                zb = transform(bb,view,first.representation,median,scale)
                distance = np.sqrt(np.mean((zb-za)**2,axis=1))
                active = ((np.ptp(bb[:,:,view],axis=1)>q[view]) &
                          (np.ptp(aa[:,:,view],axis=1)>q[view])).sum(axis=1)>=(1 if len(view)==1 else 2)
                passed = int(np.sum(active & (distance<=.25)))
            contexts.append(dict(family=family,view=first["view"],representation=first.representation,
                                 condition=first.condition,anchor_excel=first.anchor_excel,
                                 length=length,before=before,total=len(starts),valid=int(valid.sum()),
                                 excluded=int((~valid).sum()),passed=passed,
                                 rate=passed/valid.sum() if valid.any() else np.nan))
        coverage.append(dict(family=family,condition=first.condition,dates=group.target_date.nunique(),
                             segments=group.target_segment.nunique(),pairs=len(group),
                             unique_pairs=group.pair_id.nunique()))
        for segment,part in group.groupby("target_segment"):
            stats,values = gap_stats(part.target_pos.to_numpy())
            gaps.append(dict(family=family,condition=first.condition,segment=segment,**stats))
            for value,count in zip(*np.unique(values,return_counts=True)):
                distributions.append(dict(family=family,condition=first.condition,segment=segment,
                                          gap=value,count=count,tail=value>256))
    save(contexts,"all_context_summary")
    save(gaps,"all_gap_summary")
    save(distributions,"all_gap_distribution_full")
    save(coverage,"family_coverage")
    # Wide exclusion is recomputed on each original same-segment search, not inferred from counts.
    catalog = read("motif_catalog")
    for segment,part in catalog.groupby("anchor_segment"):
        local_segments = segments.iloc[[segment]].reset_index(drop=True)
        array,meta = windows(x,local_segments,keep,4)
        lookup = {int(start)+2:i for i,start in enumerate(meta.start)}
        for (view_name,rep),group in part.groupby(["view","representation"]):
            view = VIEWS[view_name]
            z = transform(array,view,rep,median,scale)
            for row in group.itertuples():
                a = lookup[row.anchor_excel]
                distance = np.sqrt(np.mean((z-z[a])**2,axis=1))
                _,_,selected,*_ = compare(array,meta,a,view,rep,distance,q,exclusion=2)
                stats,_ = gap_stats(meta.pos.iloc[selected].to_numpy())
                wide.append(dict(family=row.family,segment=segment,selected=len(selected),**stats))
    save(wide,"wide_gap_summary")
    write_json(dict(completed=True,code_sha256=digest(__file__),
                    engine_sha256=digest(HERE/"engine.py")),"postprocess_manifest.json")
    print("All-candidate context and wide-exclusion summaries complete",flush=True)


if __name__=="__main__":
    main()
