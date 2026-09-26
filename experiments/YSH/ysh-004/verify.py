"""Independent checks on YSH-004 numeric tables and source row mapping."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from analyze import COLS, digest, load_config, load_inputs


def main():
    config = load_config(HERE / "config.json")
    raw, values, segments, sheets, raw_hash = load_inputs(config)
    folder = HERE / "outputs" / "tables"
    diagnostics = pd.read_csv(folder / "row_diagnostics.csv")
    components = pd.read_csv(folder / "pca_components.csv")
    profiles = pd.read_csv(folder / "state_profiles.csv")
    motif = pd.read_csv(folder / "motif_pairs.csv")
    boundaries = pd.read_csv(folder / "change_points.csv")
    detail = pd.read_csv(folder / "voltage_pair_states.csv")
    checks = {}
    all_rows = diagnostics[diagnostics.condition == "all"].sort_values("excel_row")
    assert np.array_equal(all_rows.excel_row.to_numpy(), np.arange(2,11941))
    checks["all_row_mapping"] = len(all_rows)
    z = StandardScaler().fit_transform(values)
    pca = PCA(n_components=4,svd_solver="full").fit(z)
    k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_),.9)+1)
    pick = np.linspace(0,len(values)-1,100,dtype=int)
    transformed = pca.transform(z[pick])
    reconstruction = transformed[:,:k]@pca.components_[:k]+pca.mean_
    direct_q = np.sum((z[pick]-reconstruction)**2,axis=1)
    assert np.allclose(direct_q,all_rows.q.to_numpy()[pick],rtol=1e-8,atol=1e-8)
    checks["pca_q_sample"] = len(pick)
    for condition in ("all","controlled100","unique4"):
        d = diagnostics[diagnostics.condition==condition]
        for rep in ("X4","X3"):
            p=profiles[(profiles.condition==condition)&(profiles.representation==rep)]
            assert int(p.state_n.sum())==len(d)
            assert (d[f"{rep}_max_p"]<=1+1e-10).all()
        assert np.isfinite(d[["q","t2","if_score","lof_score"]].to_numpy()).all()
    checks["model_finite_and_state_denominators"] = len(diagnostics)
    assert np.isclose(components[(components.condition=="all")].explained.sum(),1)
    assert (boundaries.boundary_excel>=boundaries.excel_start).all()
    assert (boundaries.boundary_excel<=boundaries.excel_end).all()
    checks["boundary_rows"] = len(boundaries)
    sample = motif.sample(min(50,len(motif)),random_state=42)
    max_error=0.
    for r in sample.itertuples():
        m=int(r.setting_id.split("_")[0][1:]); j=["force","current","voltage","time"].index(r.representation)
        a=values[r.a_excel-2:r.a_excel-2+m,j]
        b=values[r.b_excel-2:r.b_excel-2+m,j]
        assert abs(r.a_excel-r.b_excel)>=m
        assert not np.array_equal(a,b)
        dist=np.sqrt(np.mean(((a-a.mean())-(b-b.mean()))**2))
        max_error=max(max_error,abs(dist-r.distance))
        assert np.isclose(dist,r.distance,atol=1e-8)
    checks["motif_direct_pairs"] = len(sample)
    checks["motif_max_abs_error"] = max_error
    prev=pd.read_csv(config["previous_pairs"])
    wanted=prev[(prev.variable=="voltage")&prev.length.isin([15,16,17])]
    assert len(detail)==len(wanted)
    assert detail.groupby(["condition","length"]).size().equals(wanted.groupby(["condition","length"]).size())
    primary=detail[(detail.length==16)&(detail.phase=="confirmation")]
    for condition,n,passed in (("all",716,269),("controlled100",134,57)):
        d=primary[primary.condition==condition]
        assert (len(d),int(d.passed.sum()))==(n,passed)
    checks["voltage_pair_rows"] = len(detail)
    labels=all_rows.X3_state.to_numpy()
    remapped=np.max(labels)-labels
    assert adjusted_rand_score(labels,remapped)==1
    checks["ari_label_permutation"] = True
    assert digest(config["raw"])==raw_hash
    from sequence_fast import l2_partition, scan_windows
    import ruptures as rpt
    small = segments[(segments.condition=="controlled100") & (segments.segment=="controlled100_s08")].iloc[0]
    piece = values[small.excel_start-2:small.excel_end-1]
    for m in (15,16,17):
        windows, nearest, where, nonexact, where_nonexact = scan_windows(piece,2,m)
        for i,a in enumerate(windows):
            valid=[j for j,b in enumerate(windows) if abs(i-j)>=m and not np.array_equal(a,b)]
            direct=min(np.sqrt(np.mean(((a-a.mean())-(windows[j]-windows[j].mean()))**2)) for j in valid)
            assert np.isclose(direct,nonexact[i],atol=1e-8)
        matching = pd.read_csv(folder/"motif_summary.csv")
        matching=matching[(matching.condition=="controlled100")&(matching.segment_id==small.segment)&(matching.representation=="voltage")&(matching.setting_id==f"m{m}")]
        assert int(matching.exact_nearest.iloc[0]) == int(np.sum(np.all(windows==windows[where],axis=1)))
    checks["small_segment_all_window_nearest"] = 15+16+17
    for min_size,coef in ((16,3),(32,3),(32,1)):
        z=StandardScaler().fit_transform(piece)
        penalty=coef*z.shape[1]*np.log(len(z))
        direct=l2_partition(z,min_size,penalty)
        library=rpt.Pelt(model="l2",min_size=min_size,jump=1).fit(z).predict(pen=penalty)[:-1]
        assert direct==library
    checks["actual_segment_pelt_comparisons"] = 3
    checks["raw_sha256"] = raw_hash
    (HERE/"outputs/verification.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print(json.dumps(checks,indent=2))


if __name__=="__main__": main()


