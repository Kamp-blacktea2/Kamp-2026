"""YSH-004: exploratory structure, rarity, and within-segment sequence analysis."""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler
import ruptures as rpt

COLS = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
NAMES = ["force", "current", "voltage", "time"]
X3 = [0, 1, 3]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(table, name):
    path = HERE / "outputs" / "tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(path, index=False, float_format="%.10g")


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("segments", "partitions", "previous_pairs", "previous_scan", "previous_exact_blocks"):
        config[key] = str((HERE / config[key]).resolve())
    return config


def load_inputs(config):
    raw_path = Path(config["raw"])
    actual_hash = digest(raw_path)
    if actual_hash != config["expected_sha256"]:
        raise ValueError(f"raw SHA-256 mismatch: {actual_hash}")
    book = pd.ExcelFile(raw_path)
    if "Raw data" not in book.sheet_names:
        raise ValueError(f"Raw data sheet missing: {book.sheet_names}")
    raw = pd.read_excel(book, sheet_name="Raw data")
    if not set(COLS).issubset(raw.columns):
        raise ValueError(f"Missing process columns: {set(COLS)-set(raw.columns)}")
    if raw.shape != (11939, 10):
        raise ValueError(f"Unexpected raw shape: {raw.shape}")
    if raw[COLS].isna().any().any():
        raise ValueError("Missing process values; no automatic imputation")
    if not all(pd.api.types.is_numeric_dtype(raw[c]) for c in COLS):
        raise ValueError("A process column is not numeric")
    values = raw[COLS].to_numpy(float)
    segments = pd.read_csv(config["segments"])
    if set(segments.condition) != {"all", "controlled100"}:
        raise ValueError("Unexpected segment conditions")
    for r in segments.itertuples():
        piece = raw.iloc[r.excel_start - 2:r.excel_end - 1]
        if len(piece) != r.n or piece["working time"].nunique() != 1:
            raise ValueError(f"Segment length/date mismatch: {r.segment}")
        if len(piece) > 1 and not np.all(np.diff(piece.idx.to_numpy()) == 1):
            raise ValueError(f"Segment idx gap: {r.segment}")
    for condition, expected in [("all", (11939, 12, 9)), ("controlled100", (2416, 9, 4))]:
        part = segments[segments.condition == condition]
        if (part.n.sum(), len(part), part.date.nunique()) != expected:
            raise ValueError(f"Segment audit failed: {condition}")
    if segments[segments.condition == "all"].excel_start.iloc[0] != 2:
        raise ValueError("First Excel data row must be 2")
    unique = pd.DataFrame(values).drop_duplicates().shape[0]
    if unique != 1574:
        raise ValueError(f"Unexpected unique4 count: {unique}")
    return raw, values, segments, book.sheet_names, actual_hash


def condition_indices(segments, condition, values):
    if condition == "unique4":
        return pd.DataFrame(values).drop_duplicates().index.to_numpy(int)
    spans = segments[segments.condition == condition]
    return np.concatenate([np.arange(r.excel_start-2, r.excel_end-1) for r in spans.itertuples()])


def audit(config, raw, values, segments, sheets, raw_hash):
    rows = []
    for condition in ("all", "controlled100", "unique4"):
        idx = condition_indices(segments, condition, values)
        part = values[idx]
        rows.append(dict(condition=condition, model="input", fit_id=condition, n=len(idx),
                         setting_id="raw", segments=int(sum(segments.condition == condition)) if condition != "unique4" else np.nan,
                         dates=int(raw.iloc[idx]["working time"].nunique()) if condition != "unique4" else np.nan,
                         unique4=int(pd.DataFrame(part).drop_duplicates().shape[0]),
                         duplicate_fraction=1-pd.DataFrame(part).drop_duplicates().shape[0]/len(part),
                         missing=int(np.isnan(part).sum()), raw_sha256=raw_hash,
                         sheets=";".join(sheets), shape=f"{raw.shape[0]}x{raw.shape[1]}"))
    for j, name in enumerate(NAMES):
        x = values[:, j]
        rows.append(dict(condition="all", model="variable", fit_id="raw", n=len(x), setting_id=name,
                         median=float(np.median(x)), iqr=float(np.subtract(*np.percentile(x, [75, 25]))),
                         minimum=float(x.min()), maximum=float(x.max()), unique=int(np.unique(x).size),
                         dtype=str(raw[COLS[j]].dtype)))
    save(rows, "input_audit.csv")
    save(pd.DataFrame(values, columns=NAMES).corr().reset_index().rename(columns={"index":"variable"}), "raw_correlation.csv")
    pairs = pd.read_csv(config["previous_pairs"])
    sample = pairs[(pairs.variable == "voltage") & (pairs.length == 16)].head(10)
    for r in sample.itertuples():
        if r.a_excel < 2 or r.b_excel + 15 > 11940:
            raise ValueError("Previous pair Excel mapping invalid")
        a = values[r.a_excel-2:r.a_excel+14, 2]
        b = values[r.b_excel-2:r.b_excel+14, 2]
        if bool(np.array_equal(a, b)) != bool(r.exact_variable):
            raise ValueError("Previous pair exact flag mismatch")
    return rows


def scale(x, kind="standard"):
    scaler = StandardScaler() if kind == "standard" else RobustScaler()
    z = scaler.fit_transform(x)
    bad = ~np.isfinite(z).all(axis=0)
    if bad.any():
        raise ValueError(f"nonfinite scaled columns {np.flatnonzero(bad)}")
    return scaler, z


def top_set(scores, fraction):
    scores = np.asarray(scores, float)
    threshold = np.quantile(scores, 1-fraction)
    return scores >= threshold, threshold


def pca_fit(z):
    pca = PCA(n_components=z.shape[1], svd_solver="full").fit(z)
    scores = pca.transform(z)
    k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), .9) + 1)
    kept = scores[:, :k]
    residual = z - kept @ pca.components_[:k] - pca.mean_
    q_parts = residual ** 2
    valid = pca.explained_variance_[:k] > 1e-12
    t2 = np.sum(kept[:, valid]**2/pca.explained_variance_[:k][valid], axis=1)
    return pca, k, q_parts, t2


def gmm_grid(z, seed=42, covariance="full", fixed_k=None):
    records, fits = [], {}
    for k in ([fixed_k] if fixed_k is not None else range(1, 7)):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = GaussianMixture(n_components=k, covariance_type=covariance,
                                        reg_covar=1e-4, n_init=5, max_iter=500,
                                        random_state=seed).fit(z)
            reg = 1e-4
            if not model.converged_:
                model = GaussianMixture(n_components=k, covariance_type=covariance,
                                        reg_covar=1e-3, n_init=5, max_iter=500,
                                        random_state=seed).fit(z)
                reg = 1e-3
            covs = model.covariances_ if covariance == "full" else [np.diag(v) for v in model.covariances_]
            records.append(dict(k=k, bic=model.bic(z), converged=bool(model.converged_),
                                iterations=int(model.n_iter_), reg_covar=reg,
                                min_weight=float(model.weights_.min()),
                                max_condition=float(max(np.linalg.cond(c) for c in covs)),
                                warnings=";".join(str(w.message) for w in caught)))
            if model.converged_:
                fits[k] = model
        except Exception as e:
            records.append(dict(k=k, bic=np.nan, converged=False, reason=str(e)))
    valid = [r for r in records if r.get("converged")]
    if not valid:
        return records, None, None
    best = min(r["bic"] for r in valid)
    representative = min(r["k"] for r in valid if r["bic"] <= best + 2)
    return records, representative, fits[representative]


def models(config, raw, values, segments):
    summary, profiles, diagnostics, pca_details, gmm_bics, stability = [], [], [], [], [], []
    fitted = {}
    for condition in ("all", "controlled100", "unique4"):
        idx = condition_indices(segments, condition, values)
        x = values[idx]
        scaler, z = scale(x)
        fit_id = f"{condition}_X4_standard"
        pca, k, qparts, t2 = pca_fit(z)
        q = qparts.sum(axis=1)
        for component in range(4):
            pca_details.append(dict(condition=condition, model="PCA", fit_id=fit_id, n=len(idx),
                                    setting_id="standard", component=component+1,
                                    explained=float(pca.explained_variance_ratio_[component]),
                                    cumulative=float(np.sum(pca.explained_variance_ratio_[:component+1])),
                                    **{f"loading_{name}":float(pca.components_[component,j]) for j,name in enumerate(NAMES)}))
        summary.append(dict(condition=condition, model="PCA", fit_id=fit_id, n=len(idx),
                            setting_id="standard", k=k, q_median=float(np.median(q)),
                            q_p95=float(np.quantile(q,.95)), t2_p95=float(np.quantile(t2,.95)),
                            q_interpretable=k<4))
        for kk in (1,2,3):
            rec = z - pca.transform(z)[:,:kk] @ pca.components_[:kk] - pca.mean_
            summary.append(dict(condition=condition, model="PCA_sensitivity", fit_id=fit_id,
                                n=len(idx), setting_id=f"k{kk}", q_median=float(np.median(np.sum(rec*rec,axis=1)))))
        gmm_outputs = {}
        for rep, columns in (("X4", [0,1,2,3]), ("X3", X3)):
            sc, zz = scale(x[:, columns])
            gf_id = f"{condition}_{rep}_standard"
            grid, chosen, gm = gmm_grid(zz)
            for rec in grid:
                gmm_bics.append(dict(condition=condition, model="GMM", representation=rep,
                                     fit_id=gf_id, n=len(idx), setting_id="full_seed42", **rec,
                                     representative=rec["k"] == chosen))
            if gm is None:
                summary.append(dict(condition=condition, model="GMM", representation=rep,
                                    fit_id=gf_id, n=len(idx), setting_id="full_seed42", reason="all fits failed"))
                continue
            post = gm.predict_proba(zz)
            labels = post.argmax(axis=1)
            entropy = -np.sum(np.where(post>0, post*np.log(np.maximum(post,1e-300)), 0),axis=1)
            entropy = entropy/np.log(chosen) if chosen>1 else np.zeros(len(idx))
            gmm_outputs[rep] = (labels, post.max(axis=1), entropy, gm.score_samples(zz))
            summary.append(dict(condition=condition, model="GMM", representation=rep,
                                fit_id=gf_id, n=len(idx), setting_id="full_seed42", k=chosen,
                                bic=float(gm.bic(zz)), uncertain_pct=float(np.mean(post.max(axis=1)<.8)*100),
                                entropy_median=float(np.median(entropy))))
            for label in range(chosen):
                mask = labels==label
                row = dict(condition=condition, model="GMM", representation=rep, fit_id=gf_id,
                           n=len(idx), setting_id="full_seed42", state=f"C{label}", state_n=int(mask.sum()),
                           share_pct=float(mask.mean()*100))
                for j,name in enumerate(NAMES):
                    row[f"{name}_median"] = float(np.median(x[mask,j])) if mask.any() else np.nan
                    row[f"{name}_iqr"] = float(np.subtract(*np.percentile(x[mask,j],[75,25]))) if mask.any() else np.nan
                profiles.append(row)
            if condition == "all":
                for seed,cov in ((7,"full"),(2026,"full"),(42,"diag")):
                    _, _, alt = gmm_grid(zz, seed, cov, chosen)
                    if alt is not None:
                        stability.append(dict(condition=condition, model="GMM", representation=rep,
                                              fit_id=gf_id, n=len(idx), setting_id=f"{cov}_seed{seed}",
                                              anchor_n=len(idx), ari=adjusted_rand_score(labels, alt.predict(zz))))
        iso = IsolationForest(n_estimators=300, max_samples=min(256,len(idx)),
                              max_features=1., bootstrap=False, contamination="auto",
                              random_state=42, n_jobs=4).fit(z)
        if_score = -iso.score_samples(z)
        lof = LocalOutlierFactor(n_neighbors=min(20,len(idx)-1), metric="euclidean", novelty=False, n_jobs=4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lof.fit(z)
            lof_score = -lof.negative_outlier_factor_
        nn = NearestNeighbors(n_neighbors=2, n_jobs=4).fit(z)
        distances = nn.kneighbors(z, return_distance=True)[0][:,1]
        univariate = np.max(np.abs((x-np.median(x,axis=0))/np.where(np.subtract(*np.percentile(x,[75,25],axis=0))>0,
                             np.subtract(*np.percentile(x,[75,25],axis=0)),1)),axis=1)
        scores = {"IF":if_score,"LOF":lof_score,"PCA_Q":q,"PCA_T2":t2,"univariate":univariate}
        for name,s in scores.items():
            for budget in (.01,.05):
                mask, cutoff = top_set(s,budget)
                summary.append(dict(condition=condition, model=name, fit_id=fit_id, n=len(idx),
                                    setting_id=f"top{budget*100:g}pct", selected=int(mask.sum()),
                                    selected_pct=float(mask.mean()*100), cutoff=float(cutoff),
                                    score_median=float(np.median(s)), score_p95=float(np.quantile(s,.95)),
                                    nonfinite=int((~np.isfinite(s)).sum())))
        summary.append(dict(condition=condition, model="LOF_diagnostics", fit_id=fit_id,n=len(idx),
                            setting_id="neighbors20", zero_distance_pct=float(np.mean(distances==0)*100),
                            tied_score_pct=float((1-np.unique(np.round(lof_score,10)).size/len(idx))*100),
                            warnings=";".join(str(w.message) for w in caught)))
        for a,b in (("IF","LOF"),("IF","PCA_Q"),("IF","univariate"),("LOF","PCA_Q"),("PCA_Q","PCA_T2")):
            aa,_=top_set(scores[a],.05); bb,_=top_set(scores[b],.05)
            summary.append(dict(condition=condition, model="overlap", fit_id=fit_id,n=len(idx),
                                setting_id=f"{a}|{b}_top5", intersection=int((aa&bb).sum()),
                                union=int((aa|bb).sum()), jaccard=float((aa&bb).sum()/(aa|bb).sum())))
        if condition=="all":
            for seed,ms in ((7,256),(2026,256),(42,1024)):
                alt=IsolationForest(n_estimators=300,max_samples=min(ms,len(idx)),random_state=seed,n_jobs=4).fit(z)
                ss=-alt.score_samples(z)
                aa,_=top_set(if_score,.05);bb,_=top_set(ss,.05)
                stability.append(dict(condition=condition,model="IF",fit_id=fit_id,n=len(idx),
                                      setting_id=f"seed{seed}_samples{ms}",anchor_n=len(idx),
                                      spearman=float(spearmanr(if_score,ss).statistic),
                                      jaccard=float((aa&bb).sum()/(aa|bb).sum())))
            for n_neighbor in (10,50):
                alt=LocalOutlierFactor(n_neighbors=n_neighbor,n_jobs=4).fit(z)
                ss=-alt.negative_outlier_factor_
                aa,_=top_set(lof_score,.05);bb,_=top_set(ss,.05)
                stability.append(dict(condition=condition,model="LOF",fit_id=fit_id,n=len(idx),
                                      setting_id=f"neighbors{n_neighbor}",anchor_n=len(idx),
                                      spearman=float(spearmanr(lof_score,ss).statistic),
                                      jaccard=float((aa&bb).sum()/(aa|bb).sum())))
            robust,zr=scale(x,"robust")
            pr,kr,qr,tr=pca_fit(zr)
            gr,chosenr,gmr=gmm_grid(zr)
            arir=adjusted_rand_score(gmm_outputs["X4"][0],gmr.predict(zr)) if gmr is not None and "X4" in gmm_outputs else np.nan
            stability.append(dict(condition=condition,model="scaler",fit_id=fit_id,n=len(idx),
                                  setting_id="robust",anchor_n=len(idx),pca_k=kr,gmm_k=chosenr,ari=arir))
        for rowno,pos in enumerate(idx):
            row=dict(condition=condition,model="row",fit_id=fit_id,n=len(idx),setting_id="standard",
                     excel_row=int(pos+2),q=float(q[rowno]),t2=float(t2[rowno]),
                     if_score=float(if_score[rowno]),lof_score=float(lof_score[rowno]),
                     univariate=float(univariate[rowno]),zero_neighbor=bool(distances[rowno]==0))
            for rep in gmm_outputs:
                lab,prob,ent,density=gmm_outputs[rep]
                row.update({f"{rep}_state":int(lab[rowno]),f"{rep}_max_p":float(prob[rowno]),
                            f"{rep}_entropy":float(ent[rowno]),f"{rep}_log_density":float(density[rowno])})
            diagnostics.append(row)
        fitted[condition]=(idx,scaler,pca,k,gmm_outputs,scores)
    save(summary,"model_summary.csv"); save(profiles,"state_profiles.csv")
    save(diagnostics,"row_diagnostics.csv"); save(pca_details,"pca_components.csv")
    save(gmm_bics,"gmm_bic.csv"); save(stability,"stability.csv")
    return fitted


def sequence(config, values, segments):
    """Use the same exact implementation for direct and CLI execution."""
    from sequence_fast import main as run_sequence

    return run_sequence(config=config, values=values, segments=segments)


def integration(config, values, segments):
    diag=pd.read_csv(HERE/"outputs/tables/row_diagnostics.csv")
    pairs=pd.read_csv(config["previous_pairs"])
    pairs=pairs[(pairs.variable=="voltage")&pairs.length.isin([15,16,17])].copy()
    rows=[]
    for condition in ("all","controlled100"):
        d=diag[diag.condition==condition].set_index("excel_row")
        if "X3_state" not in d:
            continue
        subset=pairs[pairs.condition==condition]
        for r in subset.itertuples():
            a=d.loc[np.arange(r.a_excel,r.a_excel+r.length)]
            b=d.loc[np.arange(r.b_excel,r.b_excel+r.length)]
            def state(frame):
                good=frame.X3_max_p.to_numpy()>=.8
                counts=np.bincount(frame.X3_state.to_numpy(int)[good])
                return int(counts.argmax()) if len(counts) and counts.max()>=.8*len(frame) else -1
            sa,sb=state(a),state(b)
            category="mixed_or_uncertain" if sa<0 or sb<0 else ("same" if sa==sb else "different")
            rows.append(dict(condition=condition,model="X3_GMM_voltage",representation="voltage",
                             fit_id=f"{condition}_X3_standard",n=len(d),setting_id="posterior80_window80",
                             segment_id=next((s.segment for s in segments[segments.condition==condition].itertuples()
                                              if s.excel_start<=r.a_excel<=s.excel_end),"unknown"),
                             phase=r.phase,length=r.length,offset=r.offset,a_excel=r.a_excel,b_excel=r.b_excel,
                             state_a=sa,state_b=sb,category=category,passed=bool(r.passed),
                             corr=float(r.corr) if np.isfinite(r.corr) else np.nan,
                             centered_rmse=float(r.centered_rmse)))
    detail=pd.DataFrame(rows)
    save(detail,"voltage_pair_states.csv")
    grouped=detail.groupby(["condition","phase","length","offset","category"],dropna=False).agg(
        pairs=("passed","size"),passed=("passed","sum"),segments=("segment_id","nunique"),
        median_corr=("corr","median"),median_rmse=("centered_rmse","median")).reset_index()
    grouped["pass_pct"]=100*grouped.passed/grouped.pairs
    grouped["model"]="X3_GMM_voltage"; grouped["representation"]="voltage"
    grouped["fit_id"]=grouped.condition+"_X3_standard";grouped["n"]=grouped.pairs
    grouped["setting_id"]="posterior80_window80"
    save(grouped,"voltage_by_state.csv")
    primary=detail[(detail.length==16)&(detail.phase=="confirmation")]
    for condition,expected in (("all",(716,269)),("controlled100",(134,57))):
        actual=primary[primary.condition==condition]
        if (len(actual),int(actual.passed.sum()))!=expected:
            raise ValueError(f"YSH-003 denominator/pass mismatch {condition}: {len(actual)}, {actual.passed.sum()}")
    boundaries=pd.read_csv(HERE/"outputs/tables/change_points.csv")
    near_rows=[]
    for condition in ("all","controlled100"):
        primary_cp=boundaries[(boundaries.condition==condition)&(boundaries.representation=="X3")&
                              (boundaries.setting_id=="min32_pen3")]
        starts=primary_cp.boundary_excel.to_numpy(int)
        part=primary[primary.condition==condition].copy()
        part["near_boundary"]=part.apply(lambda r:any(abs(r.a_excel-x)<=16 or abs(r.b_excel-x)<=16 for x in starts),axis=1)
        for near,g in part.groupby("near_boundary"):
            near_rows.append(dict(condition=condition,model="X3_PELT_voltage",representation="voltage",
                                  fit_id=f"{condition}_X3_standard",n=len(g),setting_id="boundary_pm16",
                                  near_boundary=bool(near),pairs=len(g),passed=int(g.passed.sum()),
                                  pass_pct=float(g.passed.mean()*100),segments=g.segment_id.nunique()))
    save(near_rows,"voltage_near_boundary.csv")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default=str(HERE/"config.json"))
    parser.add_argument("--stage",choices=["audit","models","sequence","integration","all"],default="all")
    args=parser.parse_args();config=load_config(args.config)
    os.environ.setdefault("OMP_NUM_THREADS",str(config.get("threads",4)))
    start=time.perf_counter()
    raw,values,segments,sheets,raw_hash=load_inputs(config)
    stages=["audit","models","sequence","integration"] if args.stage=="all" else [args.stage]
    for stage in stages:
        print(f"START {stage}",flush=True)
        if stage=="audit": audit(config,raw,values,segments,sheets,raw_hash)
        elif stage=="models": models(config,raw,values,segments)
        elif stage=="sequence": sequence(config,values,segments)
        else: integration(config,values,segments)
        print(f"DONE {stage}: {time.perf_counter()-start:.1f}s",flush=True)
    manifest=dict(stages=stages,seconds=time.perf_counter()-start,python=sys.version,
                  platform=platform.platform(),raw_sha256=raw_hash,
                  files={str(p):digest(p) for p in
                         [HERE/"실험계획.md",HERE/"analyze.py",HERE/"config.json",
                          Path(config["segments"]),Path(config["previous_pairs"])]},
                  packages={name:__import__(name).__version__ for name in
                            ("numpy","pandas","scipy","sklearn","ruptures")})
    (HERE/"outputs").mkdir(exist_ok=True)
    (HERE/"outputs/run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    if digest(config["raw"])!=raw_hash: raise ValueError("raw changed during run")


if __name__=="__main__": main()




