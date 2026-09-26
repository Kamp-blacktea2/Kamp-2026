"""Secondary normalization, boundary context, and exact-block audits."""

import numpy as np
import pandas as pd

from analyze import HERE, NAMES, load_config, load_inputs, save, top_set


def normalized_profile(piece, j, m):
    windows=np.lib.stride_tricks.sliding_window_view(piece[:,j],m)
    centered=windows-windows.mean(axis=1,keepdims=True)
    norm=np.linalg.norm(centered,axis=1)
    valid=norm>1e-12
    standard=np.zeros_like(centered)
    standard[valid]=centered[valid]/norm[valid,None]*np.sqrt(m)
    best=np.full(len(windows),np.nan)
    for first in range(0,len(windows),128):
        last=min(len(windows),first+128)
        # Squared z-normalized RMSE; constant windows are explicitly missing.
        corr=standard[first:last]@standard.T/m
        dist2=2-2*corr
        np.maximum(dist2,0,out=dist2)
        dist2[np.abs(np.arange(first,last)[:,None]-np.arange(len(windows))[None,:])<m]=np.inf
        dist2[:,~valid]=np.inf
        minimum=np.min(dist2,axis=1)
        result=np.sqrt(minimum)
        result[~valid[first:last]]=np.nan
        result[~np.isfinite(result)]=np.nan
        best[first:last]=result
    return best,int((~valid).sum())


def main():
    config=load_config(HERE/"config.json")
    raw,values,segments,_,_=load_inputs(config)
    tables=HERE/"outputs/tables"
    diag=pd.read_csv(tables/"row_diagnostics.csv")
    cp=pd.read_csv(tables/"change_points.csv")
    znorm=[]
    for r in segments.itertuples():
        piece=values[r.excel_start-2:r.excel_end-1]
        for j,name in enumerate(NAMES):
            for m in [16,32,64]+([15,17] if name=="voltage" else []):
                base=dict(condition=r.condition,model="z_normalized_motif",representation=name,
                          fit_id=r.segment,n=len(piece),setting_id=f"m{m}",segment_id=r.segment,
                          excel_start=r.excel_start,excel_end=r.excel_end)
                if len(piece)<2*m:
                    znorm.append(dict(**base,reason="no nonoverlap pair"));continue
                distances,constants=normalized_profile(piece,j,m)
                finite=np.isfinite(distances)
                znorm.append(dict(**base,windows=len(distances),constant_windows=constants,
                                  valid_nearest=int(finite.sum()),median=float(np.median(distances[finite])) if finite.any() else np.nan,
                                  p95=float(np.quantile(distances[finite],.95)) if finite.any() else np.nan))
    save(znorm,"znorm_motif_summary.csv")
    local=[];nearness=[];by_date=[]
    for condition in ("all","controlled100"):
        d=diag[diag.condition==condition].set_index("excel_row")
        main_cp=cp[(cp.condition==condition)&(cp.representation=="X3")&(cp.setting_id=="min32_pen3")]
        for r in main_cp.itertuples():
            for side,start,end in (("before",max(r.excel_start,r.boundary_excel-32),r.boundary_excel-1),
                                   ("after",r.boundary_excel,min(r.excel_end,r.boundary_excel+31))):
                region=d.loc[np.arange(start,end+1)]
                for state,count in region.X3_state.value_counts().sort_index().items():
                    local.append(dict(condition=condition,model="GMM_boundary",representation="X3",
                                      fit_id=f"{condition}_X3_standard",n=len(region),setting_id=side,
                                      segment_id=r.segment_id,excel_start=r.excel_start,excel_end=r.excel_end,
                                      boundary_excel=r.boundary_excel,side=side,state=int(state),state_n=int(count),
                                      share_pct=100*count/len(region)))
        near=np.zeros(len(d),bool)
        for boundary in main_cp.boundary_excel.to_numpy(int):
            near|=abs(d.index.to_numpy(int)-boundary)<=16
        for tag,mask in (("near",near),("far",~near)):
            part=d.iloc[np.flatnonzero(mask)]
            for name,col in (("IF","if_score"),("LOF","lof_score"),("PCA_T2","t2")):
                top,_=top_set(d[col].to_numpy(),.05)
                nearness.append(dict(condition=condition,model=name,representation="X4",
                                     fit_id=f"{condition}_X4_standard",n=len(part),setting_id=tag,
                                     segment_count=main_cp.segment_id.nunique(),boundary_count=len(main_cp),
                                     median=float(part[col].median()) if len(part) else np.nan,
                                     p95=float(part[col].quantile(.95)) if len(part) else np.nan,
                                     top5_count=int(np.sum(top&mask)),
                                     top5_pct=float(100*np.sum(top&mask)/len(part)) if len(part) else np.nan))
        for date,part in raw.iloc[d.index.to_numpy(int)-2].assign(excel_row=d.index.to_numpy(int)).groupby("working time"):
            rows=d.loc[part.excel_row.to_numpy(int)]
            for state,count in rows.X3_state.value_counts().sort_index().items():
                by_date.append(dict(condition=condition,model="GMM",representation="X3",fit_id=f"{condition}_X3_standard",
                                    n=len(rows),setting_id="date",date=str(date),state=int(state),state_n=int(count),
                                    share_pct=100*count/len(rows)))
    save(local,"boundary_state_occupancy.csv")
    save(nearness,"near_boundary_scores.csv")
    save(by_date,"state_by_date.csv")
    blocks=pd.read_csv(config["previous_exact_blocks"])
    checks=[]
    for r in blocks.itertuples():
        a=values[r.a_start-2:r.a_end-1]
        b=values[r.b_start-2:r.b_end-1]
        checks.append(dict(condition="all",model="exact_block_audit",representation="X4",
                           fit_id="YSH001",n=len(a),setting_id="raw_exact",
                           a_start=r.a_start,a_end=r.a_end,b_start=r.b_start,b_end=r.b_end,
                           length=r.length,actual_equal=bool(np.array_equal(a,b))))
    if not all(r["actual_equal"] for r in checks):
        raise ValueError("Prior exact-block list is inconsistent with raw workbook")
    save(checks,"exact_block_audit.csv")
    print(f"z rows {len(znorm)}, X3 local-state rows {len(local)}, exact blocks {len(checks)}")


if __name__=="__main__":main()
