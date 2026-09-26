"""Anchor-wise observed scores on the same safe windows as the maximum-selection T control."""

from engine import *
from nulls import block_labels


def build():
    config,raw,x,segments,keep,median,scale,q = load()
    labels,_ = block_labels(segments,len(x))
    array,meta = windows(x,segments,keep,4)
    ids = anchors(meta,segments,4)
    starts = set(meta.start.iloc[ids])
    safe = labels[meta.start.to_numpy()]==labels[meta.start.to_numpy()+3]
    rows = []
    for segment in range(len(segments)):
        selected = meta.index[(meta.segment==segment)&meta.front&safe].to_numpy()
        local_meta = meta.iloc[selected].reset_index(drop=True)
        if not len(local_meta):
            continue
        local = array[selected]
        candidates = np.flatnonzero(local_meta.start.isin(starts))
        for name,view in VIEWS.items():
            for rep in REPS:
                z = transform(local,view,rep,median,scale)
                for a in candidates:
                    distance = np.sqrt(cdist(z[a:a+1],z,metric="sqeuclidean")[0]/z.shape[1])
                    allowed = np.abs(local_meta.pos.to_numpy()-local_meta.pos.iloc[a])>=4
                    active = active_pair(local,a,view,q)
                    exact = np.all(local[:,:,view]==local[a:a+1,:,view],axis=(1,2))
                    found = greedy(np.flatnonzero(allowed&active&~exact&(distance<=.25)),distance,local_meta,4)
                    excel = int(local_meta.start.iloc[a])+2
                    rows.append(dict(family=f"{name}_{rep}_{excel}_all",view=name,representation=rep,
                                     anchor_excel=excel,matches=len(found),eligible=int(allowed.sum()),
                                     rate=len(found)/allowed.sum() if allowed.any() else np.nan))
    frame = pd.DataFrame(rows)
    save(frame,"order_anchor_scores")
    return frame
