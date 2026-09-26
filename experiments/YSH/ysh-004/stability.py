"""Common-row and contiguous leave-one-fifth-out sensitivity checks."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import LocalOutlierFactor

from analyze import HERE, X3, gmm_grid, load_config, load_inputs, scale, save, top_set


def compare_scores(a, b):
    aa,_=top_set(a,.05);bb,_=top_set(b,.05)
    union=(aa|bb).sum()
    return float(spearmanr(a,b).statistic),float((aa&bb).sum()/union),int(aa.sum()),int(bb.sum())


def main():
    config=load_config(HERE/"config.json")
    _,values,segments,_,_=load_inputs(config)
    d=pd.read_csv(HERE/"outputs/tables/row_diagnostics.csv")
    baseline=d[d.condition=="all"].set_index("excel_row")
    results=[]
    for condition in ("controlled100","unique4"):
        other=d[d.condition==condition].set_index("excel_row")
        common=baseline.loc[other.index]
        for rep in ("X3","X4"):
            results.append(dict(condition=f"all_vs_{condition}",model="GMM",representation=rep,
                                fit_id=f"common_rows_{rep}",n=len(other),setting_id="same_excel_rows",
                                anchor_n=len(other),ari=adjusted_rand_score(common[f"{rep}_state"],other[f"{rep}_state"])))
        for model,col in (("IF","if_score"),("LOF","lof_score"),("PCA_T2","t2")):
            rho,jacc,n1,n2=compare_scores(common[col].to_numpy(),other[col].to_numpy())
            results.append(dict(condition=f"all_vs_{condition}",model=model,fit_id="common_rows",
                                n=len(other),setting_id="same_excel_rows",anchor_n=len(other),
                                spearman=rho,jaccard=jacc,top5_all=n1,top5_other=n2))
    long_segments=segments[(segments.condition=="all")&(segments.n>=200)]
    for fold in range(5):
        removed=[]
        for r in long_segments.itertuples():
            span=np.arange(r.excel_start-2,r.excel_end-1)
            removed.extend(np.array_split(span,5)[fold])
        train=np.setdiff1d(np.arange(len(values)),removed,assume_unique=False)
        x=values[train]
        common=baseline.loc[train+2]
        setting=f"remove_fifth_{fold+1}"
        for rep,cols in (("X3",X3),("X4",[0,1,2,3])):
            _,z=scale(x[:,cols])
            grid,chosen,gm=gmm_grid(z,fixed_k=6)
            if gm is None:
                results.append(dict(condition="all",model="GMM",representation=rep,fit_id=setting,
                                    n=len(train),setting_id="fixed_K6",anchor_n=len(train),reason="fit failed"))
            else:
                results.append(dict(condition="all",model="GMM",representation=rep,fit_id=setting,
                                    n=len(train),setting_id="fixed_K6",anchor_n=len(train),removed=len(removed),
                                    ari=adjusted_rand_score(common[f"{rep}_state"],gm.predict(z))))
        _,z=scale(x)
        iso=IsolationForest(n_estimators=300,max_samples=min(256,len(z)),random_state=42,n_jobs=4).fit(z)
        if_scores=-iso.score_samples(z)
        lof=LocalOutlierFactor(n_neighbors=20,n_jobs=4).fit(z)
        for model,col,score in (("IF","if_score",if_scores),("LOF","lof_score",-lof.negative_outlier_factor_)):
            rho,jacc,n1,n2=compare_scores(common[col].to_numpy(),score)
            results.append(dict(condition="all",model=model,fit_id=setting,n=len(train),
                                setting_id="fit_on_retained_rows",anchor_n=len(train),removed=len(removed),
                                spearman=rho,jaccard=jacc,top5_all=n1,top5_other=n2))
    old=pd.read_csv(HERE/"outputs/tables/stability.csv")
    old=old[~old.fit_id.astype(str).str.startswith(("common_rows","remove_fifth_"))]
    save(pd.concat([old,pd.DataFrame(results)],ignore_index=True),"stability.csv")
    print(pd.DataFrame(results).to_string(index=False))


if __name__=="__main__": main()
