"""Detailed state support, temporal fit, boundary correspondence, and case review."""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from analyze import HERE, X3, gmm_grid, load_config, load_inputs, save, scale, top_set


def main():
    config=load_config(HERE/"config.json")
    raw,values,segments,_,_=load_inputs(config)
    folder=HERE/"outputs/tables"
    pairs=pd.read_csv(folder/"voltage_pair_states.csv")
    detail=pairs[(pairs.state_a>=0)&(pairs.state_a==pairs.state_b)].copy()
    grouped=detail.groupby(["condition","phase","length","offset","state_a"]).agg(
        pairs=("passed","size"),passed=("passed","sum"),segments=("segment_id","nunique"),
        median_corr=("corr","median"),median_rmse=("centered_rmse","median")).reset_index()
    grouped["pass_pct"]=100*grouped.passed/grouped.pairs
    grouped["supported"]=(grouped.pairs>=30)&(grouped.segments>=2)
    grouped["model"]="X3_GMM_voltage";grouped["representation"]="voltage"
    grouped["fit_id"]=grouped.condition+"_X3_standard";grouped["n"]=grouped.pairs
    grouped["setting_id"]="posterior80_window80"
    save(grouped,"voltage_by_specific_state.csv")
    diag=pd.read_csv(folder/"row_diagnostics.csv")
    occupancy=[]; temporal=[]
    for condition in ("all","controlled100"):
        d=diag[diag.condition==condition].set_index("excel_row")
        for r in segments[segments.condition==condition].itertuples():
            sub=d.loc[np.arange(r.excel_start,r.excel_end+1)]
            for rep in ("X3","X4"):
                for state,n in sub[f"{rep}_state"].value_counts().sort_index().items():
                    occupancy.append(dict(condition=condition,model="GMM",representation=rep,
                                          fit_id=f"{condition}_{rep}_standard",n=len(sub),setting_id="segment",
                                          segment_id=r.segment,date=r.date,excel_start=r.excel_start,
                                          excel_end=r.excel_end,state=int(state),state_n=int(n),
                                          share_pct=100*n/len(sub)))
        front=[];rear=[]
        for r in segments[segments.condition==condition].itertuples():
            span=np.arange(r.excel_start-2,r.excel_end-1)
            cut=len(span)//2
            front.extend(span[:cut]);rear.extend(span[cut:])
        front=np.array(front,int);rear=np.array(rear,int)
        scaler,z=scale(values[front][:,X3])
        _,_,gm=gmm_grid(z,fixed_k=6)
        if gm is None:
            temporal.append(dict(condition=condition,model="GMM",representation="X3",
                                 fit_id=f"{condition}_front_halves",n=len(front),setting_id="fixed_K6",
                                 reason="fit failed"))
        else:
            zz=scaler.transform(values[rear][:,X3]);post=gm.predict_proba(zz)
            full=d.loc[rear+2]
            temporal.append(dict(condition=condition,model="GMM",representation="X3",
                                 fit_id=f"{condition}_front_halves",n=len(front),setting_id="fixed_K6",
                                 train_n=len(front),holdout_n=len(rear),
                                 holdout_ari_vs_full=adjusted_rand_score(full.X3_state,post.argmax(axis=1)),
                                 holdout_uncertain_pct=100*np.mean(post.max(axis=1)<.8),
                                 holdout_log_density_median=float(np.median(gm.score_samples(zz)))))
    save(occupancy,"state_occupancy.csv");save(temporal,"temporal_fit.csv")
    cp=pd.read_csv(folder/"change_points.csv")
    matches=[]
    for (condition,rep,segment),g in cp.groupby(["condition","representation","segment_id"]):
        primary=np.sort(g[g.setting_id=="min32_pen3"].boundary_excel.to_numpy(int))
        for setting in ("min16_pen3","min64_pen3","min32_pen1","min32_pen10"):
            other=np.sort(g[g.setting_id==setting].boundary_excel.to_numpy(int))
            for tol in (4,8,16):
                candidates=sorted((abs(int(a-b)),int(a),int(b)) for a in primary for b in other if abs(a-b)<=tol)
                used_a=set();used_b=set();dist=[]
                for delta,a,b in candidates:
                    if a not in used_a and b not in used_b:
                        used_a.add(a);used_b.add(b);dist.append(delta)
                matches.append(dict(condition=condition,model="PELT",representation=rep,
                                    fit_id=f"{condition}_{rep}_standard",n=len(primary),setting_id=setting,
                                    segment_id=segment,tolerance=tol,primary_n=len(primary),other_n=len(other),
                                    matched=len(dist),match_pct=100*len(dist)/len(primary) if len(primary) else np.nan,
                                    median_shift=float(np.median(dist)) if dist else np.nan))
    save(matches,"pelt_stability.csv")
    all_d=diag[diag.condition=="all"].set_index("excel_row")
    cp_all=cp[(cp.condition=="all")&(cp.representation=="X3")&(cp.setting_id=="min32_pen3")]
    cp_pos=cp_all.boundary_excel.to_numpy(int)
    if_top,_=top_set(all_d.if_score,.01);lof_top,_=top_set(all_d.lof_score,.01)
    rows=all_d.index.to_numpy(int)
    categories=[("IF_LOF_common",rows[if_top&lof_top]),
                ("IF_only",rows[if_top&~lof_top]),
                ("LOF_only",rows[lof_top&~if_top]),
                ("frequent_state",rows[(all_d.X3_state==all_d.X3_state.value_counts().idxmax())&~if_top&~lof_top]),
                ("near_change",np.array([x for x in rows if len(cp_pos) and np.min(abs(cp_pos-x))<=2],int))]
    motif=pd.read_csv(folder/"motif_pairs.csv")
    mm=motif[(motif.condition=="all")&(motif.representation=="voltage")&
             (motif.setting_id.isin(["m16_similar","m16_dissimilar"]))]
    for setting in ("m16_similar","m16_dissimilar"):
        s=mm[mm.setting_id==setting].sort_values("distance",ascending=setting=="m16_similar")
        categories.append((setting,s.a_excel.to_numpy(int)))
    selected=[];seen=set()
    for category,candidates in categories:
        # Numerical selection is deterministic and drawn from the requested class.
        for excel in candidates:
            if excel in seen:continue
            seen.add(excel);record=all_d.loc[excel];v=values[excel-2]
            selected.append(dict(condition="all",model="case_review",representation="X4/X3",
                                 fit_id="all_X4_standard",n=len(all_d),setting_id=category,
                                 excel_row=int(excel),date=str(raw.iloc[excel-2]["working time"]),
                                 force=float(v[0]),current=float(v[1]),voltage=float(v[2]),time=float(v[3]),
                                 X3_state=int(record.X3_state),X3_max_p=float(record.X3_max_p),
                                 if_score=float(record.if_score),lof_score=float(record.lof_score),
                                 t2=float(record.t2),nearest_X3_boundary=int(np.min(abs(cp_pos-excel))) if len(cp_pos) else np.nan))
            break
        if len(selected)>=10:break
    save(selected,"case_review.csv")
    print(f"state rows {len(grouped)}; temporal rows {len(temporal)}; cases {len(selected)}")


if __name__=="__main__":main()

