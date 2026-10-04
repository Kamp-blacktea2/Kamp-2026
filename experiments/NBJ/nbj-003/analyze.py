"""
Post-hoc reproduction script for NBJ-003.

IMPORTANT
---------
This file was reconstructed from the already-executed notebook stored in
``outputs/``.  It is NOT claimed to be the exact historical script that was
used on the original execution date.  Its purpose is to make the documented
analysis reproducible from the original Excel workbook without changing the
reported interpretation.

The script keeps the notebook's analysis choices and assertions.  Reproduced
figures are written under ``outputs/reproduction/`` so historical notebook
artifacts are not overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import random
import sys
import tempfile
import zipfile
from pathlib import Path
import importlib.metadata as metadata

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

import torch
from torch import nn
from torch.utils.data import DataLoader


EXP_DIR = Path(__file__).resolve().parent
REPRO_OUT = EXP_DIR / "outputs" / "reproduction"
REPRO_OUT.mkdir(parents=True, exist_ok=True)

_TMP_DIR: tempfile.TemporaryDirectory | None = None


def _materialize_xlsx_from_zip(zip_path: Path) -> Path:
    global _TMP_DIR
    with zipfile.ZipFile(zip_path) as zf:
        matches = [n for n in zf.namelist() if n.replace("\\", "/").endswith("Welding Data Set_01.xlsx")]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one Welding Data Set_01.xlsx in {zip_path}, found {len(matches)}"
            )
        payload = zf.read(matches[0])
    _TMP_DIR = tempfile.TemporaryDirectory(prefix="nbj-003_")
    target = Path(_TMP_DIR.name) / "Welding Data Set_01.xlsx"
    target.write_bytes(payload)
    return target


def resolve_data_path(value: str | None) -> Path:
    """Resolve an extracted Excel workbook or the original ZIP.

    Priority:
    1) --data
    2) KAMP_WELDING_DATA environment variable
    3) common repo locations
    4) the historical Downloads ZIP name
    """
    requested = value or os.environ.get("KAMP_WELDING_DATA")
    candidates: list[Path] = []
    if requested:
        candidates.append(Path(requested).expanduser())

    roots = [Path.cwd(), *Path.cwd().parents, EXP_DIR, *EXP_DIR.parents]
    for root in roots:
        for sub in ("", "data", "work"):
            candidates.append(root / sub / "Welding Data Set_01.xlsx")
        candidates.append(root / "2. 용접기 AI 데이터셋 (1).zip")

    candidates.append(Path.home() / "Downloads" / "2. 용접기 AI 데이터셋 (1).zip")

    seen = set()
    for p in candidates:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if not p.is_file():
            continue
        if p.suffix.lower() == ".zip":
            return _materialize_xlsx_from_zip(p)
        if p.suffix.lower() in (".xlsx", ".xls"):
            return p

    raise FileNotFoundError(
        "Original welding workbook was not found. "
        "Run with --data <Welding Data Set_01.xlsx or source ZIP>."
    )


def display(obj) -> None:
    """Console replacement for IPython.display.display."""
    if isinstance(obj, pd.DataFrame):
        print(obj.to_string())
    elif isinstance(obj, pd.Series):
        print(obj.to_string())
    else:
        print(obj)


def Markdown(text: str) -> str:
    """Console replacement for IPython.display.Markdown."""
    return text


FIGURE_NAMES = ['rolling_current_std_vs_defect_rate_reproduced.png', 'window_sensitivity_reproduced.png', 'current_mean_vs_variability_reproduced.png', 'power_resistance_proxy_comparison_reproduced.png', 'ae_vs_rolling_comparison_reproduced.png', 'feature_redundancy_heatmap_reproduced.png', 'april_03_rolling_warning_example_reproduced.png']
_FIGURE_INDEX = 0


def save_show() -> None:
    """Save the current notebook figure instead of opening an interactive window."""
    global _FIGURE_INDEX
    fig = plt.gcf()
    if _FIGURE_INDEX < len(FIGURE_NAMES):
        name = FIGURE_NAMES[_FIGURE_INDEX]
    else:
        name = f"figure_{_FIGURE_INDEX + 1:02d}_reproduced.png"
    path = REPRO_OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("saved figure:", path)
    _FIGURE_INDEX += 1


parser = argparse.ArgumentParser(description="Reproduce NBJ-003 from the original KAMP welding workbook.")
parser.add_argument(
    "--data",
    default=None,
    help="Path to Welding Data Set_01.xlsx or the original welding dataset ZIP. "
         "If omitted, common repo locations and ~/Downloads are searched.",
)
ARGS = parser.parse_args()
DATA_PATH_OVERRIDE = resolve_data_path(ARGS.data)

# --- reproduced from notebook code cell 2 ---
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
if any(f.name=='Malgun Gothic' for f in font_manager.fontManager.ttflist):
    plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams.update({'axes.unicode_minus':False,'figure.dpi':110,'font.size':10})
pd.set_option('display.max_columns',30)
DATA_PATH = Path(DATA_PATH_OVERRIDE)
source_hash=hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
raw=pd.read_excel(DATA_PATH,sheet_name='Raw data')
result=pd.read_excel(DATA_PATH,sheet_name='result')
FEATURES=['weld force(bar)','weld current(kA)','weld Voltage(v)','weld time(ms)']
SHORT=['force','current','voltage','time']
UNITS={'force':'bar','current':'kA','voltage':'V','time':'ms'}
d=raw.rename(columns=dict(zip(FEATURES,SHORT))).copy()
d['date']=pd.to_datetime(d['working time'],errors='raise').dt.normalize()
d['row_id']=np.arange(len(d))
d['excel_row']=d['row_id']+2
result['date']=pd.to_datetime(result['working time'],errors='raise').dt.normalize()
assert d[SHORT+['idx']].notna().all().all()
assert np.isfinite(d[SHORT].to_numpy()).all()
assert len(d)==11939 and len(result)==23
assert (result['defect']>=0).all() and result['defect'].notna().all()
assert set(result['defect type'])=={1,2,3}
assert not result.duplicated(['date','Machine_Name','Item No','defect type']).any()
assert set(map(tuple,d[['Machine_Name','Item No']].drop_duplicates().values))==set(map(tuple,result[['Machine_Name','Item No']].drop_duplicates().values))
assert d[['Machine_Name','Item No']].drop_duplicates().shape[0]==1
dates=sorted(d['date'].unique())
versions={p:metadata.version(p) for p in ['numpy','pandas','openpyxl','matplotlib','torch']}
versions['Python']=platform.python_version()
print('원본:',DATA_PATH,'\nSHA-256:',source_hash,'\n환경:',versions)
print(f'Raw {len(d):,}행 / result {len(result)}행 / {len(dates)}개 날짜 / 삭제 0행')
display(d[['excel_row','idx','date']+SHORT].head())
NBJ_ROOT = EXP_DIR.parent
OUTPUT = REPRO_OUT
SUMMARY_PATH = OUTPUT / '03_rolling_process_instability_summary_reproduced.md'
PROTECTED = [DATA_PATH]
for _p in [
    NBJ_ROOT / 'nbj-001' / 'outputs' / '01_AutoEncoder_guide_baseline.ipynb',
    NBJ_ROOT / 'nbj-002' / 'outputs' / '02_defect_relevance_analysis.ipynb',
]:
    if _p.is_file():
        PROTECTED.append(_p)
protected_hash = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in PROTECTED}
def rho(x,y):
    a=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna().round(12)
    if len(a)<3 or a.x.nunique()<2 or a.y.nunique()<2:return np.nan
    return a.x.rank(method='average').corr(a.y.rank(method='average'))
def mdtable(frame):
    z=frame.copy().reset_index(drop=True)
    def fmt(v):
        if isinstance(v,(float,np.floating)):return '계산 불가' if pd.isna(v) else f'{v:.3f}'
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,z.columns))+' |\n| '+' | '.join(['---']*len(z.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in z.itertuples(index=False,name=None))

# --- reproduced from notebook code cell 4 ---
SUSPECT=[3552,5201]
display(d.loc[d.excel_row.isin(SUSPECT),['excel_row','idx','date']+SHORT])
print('날짜+idx 중복 행:',int(d.duplicated(['date','idx'],keep=False).sum()))
variants={'original':d.copy(),'exclude_suspect':d.loc[~d.excel_row.isin(SUSPECT)].copy()}
daily_tables={}
for variant,frame in variants.items():
    assert frame.excel_row.is_monotonic_increasing
    frame['segment']=frame.date.ne(frame.date.shift()).cumsum()
    frame['position']=frame.groupby('date',sort=False).cumcount()+1
    frame['resistance']=frame.voltage/frame.current.replace(0,np.nan)
    frame['power']=frame.voltage*frame.current
    frame['energy']=frame.voltage*frame.current*frame.time
    frame['abs_delta_current']=frame.groupby('segment',sort=False).current.diff().abs()
    tab=frame.groupby('date').size().rename('production_rows').to_frame()
    tab['recorded_defects']=result.groupby('date').defect.sum(min_count=1)
    tab['defect_rate_pct']=tab.recorded_defects/tab.production_rows*100
    tab['types_recorded']=result.groupby('date')['defect type'].nunique()
    tab['segments']=frame.groupby('date').segment.nunique()
    daily_tables[variant]=tab
    print(variant,'행 수',len(frame),'연속 날짜 구간',frame.segment.nunique())
    display(tab)
daily=daily_tables['original'].copy()
assert daily.loc['2020-03-27','recorded_defects']!=daily.loc['2020-03-27','recorded_defects']
assert daily.loc['2020-03-31','types_recorded']==2
DATES8=daily.index[daily.defect_rate_pct.notna()]
DATES7=DATES8[DATES8!=pd.Timestamp('2020-03-31')]
assert len(DATES8)==8 and len(DATES7)==7

# --- reproduced from notebook code cell 6 ---
WINDOWS=[20,50,100,200]
SCENARIOS={'B':'검사 전 (현재 포함)','A':'용접 전 (직전까지)'}
BASE=['current','power','resistance','force','voltage']
UNIT={'current':'kA','power':'kW (proxy)','resistance':'mΩ (proxy)','force':'bar','voltage':'V','delta_current':'kA','abs_delta_current':'kA'}
def feature_unit(f):
    return UNIT['delta_current'] if 'delta_current' in f else UNIT[f.rsplit('_',1)[0]]
def calculate_roll(frame,n,scenario):
    pieces=[]
    for seg,g in frame.groupby('segment',sort=False):
        z=g[BASE+['abs_delta_current']].copy()
        if scenario=='A':z=z.shift(1)
        out=pd.DataFrame(index=g.index)
        for f in BASE:
            r=z[f].rolling(n,min_periods=n)
            if f in ['current','power','resistance']:
                out[f+'_mean']=r.mean()
                out[f+'_mad']=r.apply(lambda a:np.median(np.abs(a-np.median(a))),raw=True)
            out[f+'_std']=r.std(ddof=1)
            out[f+'_iqr']=r.quantile(.75)-r.quantile(.25)
        rr=z.abs_delta_current.rolling(n,min_periods=n)
        out['delta_current_mean']=rr.mean()
        out['delta_current_p90']=rr.quantile(.9)
        out['abs_delta_current']=z.abs_delta_current.where(z.current.rolling(n,min_periods=n).count().eq(n))
        pieces.append(out)
    return pd.concat(pieces).sort_index()
rolling={}
for variant,frame in variants.items():
    for scenario in SCENARIOS:
        for n in WINDOWS:
            rolling[(variant,scenario,n)]=calculate_roll(frame,n,scenario)
    print('Rolling 생성:',variant,flush=True)
FEATURE_NAMES=list(rolling[('original','B',20)].columns)
check_frame=variants['original'].iloc[:277].copy()
for scenario in SCENARIOS:
    check=calculate_roll(check_frame,20,scenario)
    pd.testing.assert_frame_equal(check,rolling[('original',scenario,20)].loc[check.index])
for (variant,scenario,n),r in rolling.items():
    frame=variants[variant]
    offset=1 if scenario=='A' else 0
    for _,g in frame.groupby('segment',sort=False):
        assert r.loc[g.index[:min(len(g),n-1+offset)],'current_std'].isna().all()
print('미래행 불사용 / 날짜 경계 초기화 확인. 변수:',FEATURE_NAMES)
display(rolling[('original','B',20)].head(23))

# --- reproduced from notebook code cell 8 ---
records=[]
for (variant,scenario,n),r in rolling.items():
    frame=variants[variant]
    for date in sorted(frame.date.unique()):
        mask=frame.date.eq(date)
        for f in FEATURE_NAMES:
            train=r.loc[~mask,f].dropna();test=r.loc[mask,f].dropna()
            q95,q99=train.quantile([.95,.99])
            records.append(dict(variant=variant,scenario=scenario,window=n,feature=f,date=date,
                production_n=int(mask.sum()),valid_n=len(test),reference_n=len(train),coverage_pct=len(test)/mask.sum()*100,
                threshold95=q95,threshold99=q99,feature_mean=test.mean(),feature_p90=test.quantile(.9),feature_p95=test.quantile(.95),
                warning95_count=int((test>q95).sum()),warning99_count=int((test>q99).sum()),
                warning95_rate=(test>q95).mean()*100,warning99_rate=(test>q99).mean()*100))
daily_roll=pd.DataFrame(records)
assert daily_roll.valid_n.gt(0).all()
assert (daily_roll.warning99_count<=daily_roll.warning95_count).all()
display(daily_roll.query("variant=='original' and scenario=='B' and window==50 and feature=='power_iqr'").round(5))
print('날짜×조건×변수 집계:',len(daily_roll),'행. 경고 분모는 valid_n입니다.')

# --- reproduced from notebook code cell 10 ---
METRICS=['feature_mean','feature_p90','feature_p95','warning95_rate','warning99_rate']
def assoc(x,target):
    valid=target.dropna().index.intersection(x.dropna().index)
    full=rho(x.loc[valid],target.loc[valid])
    vals=[rho(x.loc[valid.drop(dt)],target.loc[valid.drop(dt)]) for dt in valid]
    finite=[v for v in vals if np.isfinite(v)]
    seven=valid[valid!=pd.Timestamp('2020-03-31')]
    return dict(rho=full,loo_min=min(finite) if finite else np.nan,loo_max=max(finite) if finite else np.nan,
                loo_defined=len(finite),rho_7=rho(x.loc[seven],target.loc[seven]),n=len(valid))
rows=[];ranking_rows=[]
for keys,g in daily_roll.groupby(['variant','scenario','window','feature'],sort=False):
    variant,scenario,n,f=keys;g=g.set_index('date')
    target=daily_tables[variant].defect_rate_pct
    for metric in METRICS:
        rows.append(dict(variant=variant,scenario=scenario,window=n,feature=f,metric=metric,**assoc(g[metric],target)))
        rank=pd.DataFrame({'value':g[metric],'defect_rate_pct':target}).dropna()
        rank['indicator_rank']=rank.value.rank(ascending=False,method='average')
        rank['defect_rank']=rank.defect_rate_pct.rank(ascending=False,method='average')
        for date,v in rank.iterrows():ranking_rows.append(dict(variant=variant,scenario=scenario,window=n,feature=f,metric=metric,date=date,**v.to_dict()))
association=pd.DataFrame(rows)
rankings=pd.DataFrame(ranking_rows)
table1=association.query("variant=='original'").copy()
display(Markdown('### 표 1 · 전체 window × feature 연관성 (접어서 볼 수 있는 전체 표)'))
display(Markdown('<details><summary>전체 '+str(len(table1))+'개 결과 펼치기</summary>\n\n'+table1.to_html(index=False,float_format=lambda x:f'{x:.4f}')+'\n</details>'))
display(table1.query("scenario=='B' and metric=='feature_mean' and window!=200").pivot_table(index='feature',columns='window',values='rho',dropna=False).round(4))
display(Markdown('<details><summary>모든 지표의 날짜별 순위 펼치기</summary>\n\n'+rankings.query("variant=='original'").to_html(index=False,float_format=lambda x:f'{x:.4f}')+'\n</details>'))

# --- reproduced from notebook code cell 12 ---
eligible=[];stability_rows=[]
for f in FEATURE_NAMES:
    g=association.query("feature==@f and window!=200 and metric=='feature_mean'")
    ok=(len(g)==12 and g.rho.ge(.5).all() and g.loo_min.ge(.15).all() and g.rho_7.ge(.4).all() and g.loo_defined.eq(8).all())
    variability=not (f.endswith('_mean') and f in ['current_mean','power_mean','resistance_mean']) and f!='abs_delta_current'
    if ok and variability:eligible.append(f)
    stability_rows.append(dict(feature=f,rho_min=g.rho.min(),rho_max=g.rho.max(),loo_min=g.loo_min.min(),rho7_min=g.rho_7.min(),passes=ok and variability))
stability=pd.DataFrame(stability_rows)
wide=daily_roll.query("variant=='original' and scenario=='B' and window==50").pivot(index='date',columns='feature',values='feature_mean').reindex(DATES8)
candidate_pool=['current_std','current_iqr','current_mad','delta_current_mean','delta_current_p90','power_iqr','resistance_iqr','voltage_std','voltage_iqr','power_std','resistance_std','power_mad','resistance_mad','force_std','force_iqr']
candidate_corr=pd.DataFrame({f:{h:rho(wide[f],wide[h]) for h in candidate_pool} for f in candidate_pool})
selected=[];duplicates=[]
for f in candidate_pool:
    if f not in eligible:continue
    close=[h for h in selected if abs(candidate_corr.loc[f,h])>=.9]
    if close:duplicates.append({'feature':f,'represented_by':close[0],'rho':candidate_corr.loc[f,close[0]]})
    elif len(selected)<3:selected.append(f)
FOCUS=selected[0] if selected else 'power_iqr'
N_MAIN=50
display(stability.round(4));display(pd.DataFrame(duplicates))
print('필터 통과:',eligible,'중복 축소 후:',selected,'설명용 주 비교:',FOCUS)
table3=[]
for f in selected:
    a=association.query("variant=='original' and scenario=='B' and window==50 and feature==@f and metric=='feature_mean'").iloc[0]
    st=stability.set_index('feature').loc[f]
    table3.append({'feature':f,'window':50,'scenario':'B (A도 확인)','rho':a.rho,'stability':f'전체 조건 ρ {st.rho_min:.3f}~{st.rho_max:.3f}; LOO 최소 {st.loo_min:.3f}',
        '장점':'현재까지 최근 50개로 계산','한계':'8일 탐색·후보 선택 편향·개별 label 없음'})
table3=pd.DataFrame(table3,columns=['feature','window','scenario','rho','stability','장점','한계'])
display(Markdown('### 표 3 · 최종 후보 (최대 3개)'));display(table3)
if table3.empty:print('안정성 기준을 모두 충족한 후보 없음. 후보 확정 및 결합 점수 생성 보류.')
suspect_compare=association.query("scenario=='B' and window==50 and metric=='feature_mean'").pivot(index='feature',columns='variant',values=['rho','rho_7'])
display(suspect_compare.round(4))
display(rankings.query("variant=='original' and scenario=='B' and window==50 and feature==@FOCUS and metric=='feature_mean'"))

# --- reproduced from notebook code cell 14 ---
PRIMARY_SEED=42
SEEDS=[42,7,2026]
EPOCHS,BATCH_SIZE,LR=50,64,0.01
X=d[SHORT].to_numpy(dtype=np.float64)
date_values=d['date'].to_numpy()
class DateAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4,3),nn.RReLU(),nn.Linear(3,2),nn.RReLU(),nn.Linear(2,3),nn.RReLU(),nn.Linear(3,4))
    def forward(self,x):return self.net(x)

def score_eval(model,array):
    model.eval()
    assert not model.training
    with torch.no_grad():
        tensor=torch.tensor(array,dtype=torch.float32)
        return ((model(tensor)-tensor)**2).mean(dim=1).numpy().astype(float)

print('사전 설정:',{'folds':len(dates),'seeds':SEEDS,'epochs':EPOCHS,'batch':BATCH_SIZE,'lr':LR,'threshold':'Train eval-MSE mean + 8 std(ddof=0)','evaluation':'eval/no_grad'})

# --- reproduced from notebook code cell 16 ---
fold_records=[]
scored=[]
for seed in SEEDS:
    for heldout in dates:
        test_mask=d['date'].eq(heldout).to_numpy()
        train_mask=~test_mask
        train,test=X[train_mask],X[test_mask]
        assert len(train)+len(test)==len(X)
        low,high=train.min(axis=0),train.max(axis=0)
        span=high-low
        assert (span>0).all()
        ztrain=(train-low)/span
        ztest=(test-low)/span
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=DateAE()
        optimizer=torch.optim.Adam(model.parameters(),lr=LR)
        tensor=torch.tensor(ztrain,dtype=torch.float32)
        loader=DataLoader(tensor,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
        model.train()
        for epoch in range(EPOCHS):
            for batch in loader:
                optimizer.zero_grad()
                loss=((model(batch)-batch)**2).mean()
                loss.backward();optimizer.step()
        train_error=score_eval(model,ztrain)
        test_error=score_eval(model,ztest)
        np.testing.assert_array_equal(test_error,score_eval(model,ztest))
        threshold=float(train_error.mean()+8*train_error.std(ddof=0))
        train_p95=float(np.quantile(train_error,.95))
        assert train_p95>0 and threshold>0 and np.isfinite(test_error).all()
        train_patterns=set(map(tuple,train))
        overlap=np.array([tuple(row) in train_patterns for row in test])
        range_out=((test<low)|(test>high)).any(axis=1)
        fold_records.append({'seed':seed,'date':heldout,'train_n':len(train),'test_n':len(test),
            'threshold':threshold,'train_error_p95':train_p95,'train_error_mean':train_error.mean(),
            'pattern_overlap_n':int(overlap.sum()),'pattern_overlap_pct':overlap.mean()*100,
            'out_of_train_range_n':int(range_out.sum()),
            **{f'{key}_train_min':low[i] for i,key in enumerate(SHORT)},
            **{f'{key}_train_max':high[i] for i,key in enumerate(SHORT)}})
        scored.append(pd.DataFrame({'row_id':d.loc[test_mask,'row_id'].to_numpy(),'date':heldout,'seed':seed,
            'error':test_error,'relative_error':test_error/train_p95,'threshold':threshold,
            'anomaly':test_error>threshold,'seen_pattern':overlap}))
        print(f'seed={seed}, heldout={pd.Timestamp(heldout).date()}, Train={len(train)}, 평가={len(test)}, 후보={(test_error>threshold).sum()}, pattern overlap={overlap.mean()*100:.1f}%',flush=True)
folds=pd.DataFrame(fold_records)
oos=pd.concat(scored,ignore_index=True)
assert len(oos)==len(d)*len(SEEDS)
assert not oos.duplicated(['seed','row_id']).any()
display(folds.loc[folds.seed.eq(PRIMARY_SEED)].round(6))

# --- reproduced from notebook code cell 18 ---
LABELS={}
ae_rows=[]
for (seed,dt),g in oos.groupby(['seed','date']):
    row={'seed':seed,'date':dt,'ae_mean':g.error.mean(),'ae_median':g.error.median(),
        'ae_p90':g.error.quantile(.9),'ae_p95':g.error.quantile(.95),'ae_p99':g.error.quantile(.99),
        'ae_anomaly_count':int(g.anomaly.sum()),'ae_anomaly_pct':g.anomaly.mean()*100,
        'ae_relative_mean':g.relative_error.mean(),'ae_relative_p95':g.relative_error.quantile(.95)}
    ae_rows.append(row)
ae_by_seed=pd.DataFrame(ae_rows)
ae_primary=ae_by_seed.loc[ae_by_seed.seed.eq(PRIMARY_SEED)].drop(columns='seed').set_index('date')
daily=daily.join(ae_primary)
LABELS.update({'ae_mean':'AE mean MSE','ae_median':'AE median MSE','ae_p90':'AE p90 MSE','ae_p95':'AE p95 MSE','ae_p99':'AE p99 MSE',
 'ae_anomaly_count':'AE 후보 수 (건)','ae_anomaly_pct':'AE 후보 비율 (%)','ae_relative_mean':'AE 상대 mean (배)','ae_relative_p95':'AE 상대 p95 (배)'})
print('오차는 정규화 MSE(무차원), 상대 오차는 Train p95 대비 배수입니다.')
display(daily[['production_rows','recorded_defects','defect_rate_pct']+list(ae_primary.columns)].round(6))
expected_counts=[0,14,0,15,0,14,0,15,0]
assert ae_primary.ae_anomaly_count.tolist()==expected_counts
np.testing.assert_allclose(rho(ae_primary.ae_anomaly_pct,daily.defect_rate_pct),.3546,atol=5e-5)
print('02 주 seed의 9일 후보 수 및 후보 비율 상관 재현 확인')

# --- reproduced from notebook code cell 20 ---
focus_daily=daily_roll.query("variant=='original' and scenario=='B' and window==50 and feature==@FOCUS").set_index('date')
comparison=[]
for f in ['ae_mean','ae_p95','ae_anomaly_pct','ae_relative_p95']:
    comparison.append(dict(indicator=f,**assoc(ae_primary[f],daily.defect_rate_pct)))
for metric in ['feature_mean','warning95_rate','warning99_rate']:
    comparison.append(dict(indicator=f'{FOCUS} / {metric} / B50',**assoc(focus_daily[metric],daily.defect_rate_pct)))
table2=pd.DataFrame(comparison)
display(Markdown('### 표 2 · AutoEncoder와 rolling 직접 비교'));display(table2.round(4))
ae_roll_corr=pd.DataFrame({f:{m:rho(ae_primary[f],focus_daily[m]) for m in ['feature_mean','warning95_rate','warning99_rate']} for f in ['ae_mean','ae_p95','ae_anomaly_pct']})
display(ae_roll_corr.round(4))
primary_rows=oos.query('seed==42').set_index('row_id').sort_index()
pattern_frequency=d.groupby(SHORT,dropna=False).current.transform('size')
pattern_rho=rho(pd.Series(primary_rows.relative_error.to_numpy(),index=d.index),pattern_frequency)
force_share=d.assign(high=d.force.gt(3)).groupby('date').high.mean()*100
pattern_daily_rho=rho(ae_primary.ae_anomaly_pct.reindex(DATES8),force_share.reindex(DATES8))
print(f'다른 날짜 패턴 중복률 범위: {folds.pattern_overlap_pct.min():.1f}~{folds.pattern_overlap_pct.max():.1f}%')
print(f'반복 빈도 vs 행별 AE 상대오차 ρ={pattern_rho:.4f}; Force>3 비율 vs AE 후보 비율 (8일) ρ={pattern_daily_rho:.4f}')
valid_mask=rolling[('original','B',50)][FOCUS].notna()
ae_aligned=primary_rows.loc[d.loc[valid_mask,'row_id']].groupby('date').anomaly.mean()*100
print('AE 후보 비율(rolling과 같은 유효 행)의 불량률 상관:',rho(ae_aligned,daily.defect_rate_pct))
display(pd.DataFrame({'AE_full_pct':ae_primary.ae_anomaly_pct,'AE_rolling_valid_pct':ae_aligned,'rolling_warning95_pct':focus_daily.warning95_rate,'rolling_valid_n':focus_daily.valid_n}))
seed_compare=[]
for seed,g in ae_by_seed.groupby('seed'):
    g=g.set_index('date')
    seed_compare.append({'seed':seed,**{f:rho(g[f],daily.defect_rate_pct) for f in ['ae_mean','ae_p95','ae_anomaly_pct']}})
display(pd.DataFrame(seed_compare).round(4))
# 02 전체 날짜 지표 재현: 숫자가 다른 경우 계산 방식의 차이를 설명할 수 있게 남깁니다.
prior=[]
for label,f,stat,expected in [('power IQR','power','iqr',.7380952381),('resistance IQR','resistance','iqr',.6904761905),('current mean','current','mean',.0238095238),('current std','current','std',.4285714286)]:
    g=variants['original'].groupby('date')[f]
    x=(g.quantile(.75)-g.quantile(.25)) if stat=='iqr' else getattr(g,stat)()
    value=rho(x,daily.defect_rate_pct);assert abs(value-expected)<1e-6
    prior.append({'02 지표':label,'재현 rho':value,'기존 rho':expected})
display(pd.DataFrame(prior))
print('02의 Δ는 고유 연속 idx만 사용; 03의 Δ는 연속 날짜 구간 내 원본 Excel 인접 행을 사용하므로 동일 정의가 아닙니다.')

# --- reproduced from notebook code cell 22 ---
score_rows=[];score_row_values={};score_status='결합 점수 생성 보류: 중복 축소 후 안정 후보가 2개 미만'
if len(selected)>=2:
    score_status='동일 가중치 탐색 점수 생성 (독립 성능 검증 아님)'
    for variant,frame in variants.items():
        for scenario in SCENARIOS:
            r=rolling[(variant,scenario,50)][selected]
            for date in sorted(frame.date.unique()):
                mask=frame.date.eq(date);train=r.loc[~mask].dropna();test=r.loc[mask].dropna()
                med=train.median();spread=train.quantile(.75)-train.quantile(.25)
                if (spread<=0).any():
                    score_rows.append(dict(variant=variant,scenario=scenario,date=date,status='Train IQR=0'))
                    continue
                ztrain=((train-med)/spread).mean(axis=1);ztest=((test-med)/spread).mean(axis=1)
                q95,q99=ztrain.quantile([.95,.99])
                score_row_values[(variant,scenario,pd.Timestamp(date))]=ztest
                score_rows.append(dict(variant=variant,scenario=scenario,date=date,status='ok',valid_n=len(ztest),mean_score=ztest.mean(),p95_score=ztest.quantile(.95),threshold95=q95,threshold99=q99,
                    warning95_rate=ztest.ge(q95).mean()*100,warning99_rate=ztest.ge(q99).mean()*100))
score_daily=pd.DataFrame(score_rows)
score_assoc=[]
if not score_daily.empty:
    for (variant,scenario),g in score_daily.groupby(['variant','scenario']):
        g=g.set_index('date')
        for f in ['mean_score','p95_score','warning95_rate','warning99_rate']:
            if f in g:score_assoc.append(dict(variant=variant,scenario=scenario,metric=f,**assoc(g[f],daily_tables[variant].defect_rate_pct)))
print(score_status)
display(score_daily.round(4));display(pd.DataFrame(score_assoc).round(4))

# --- reproduced from notebook code cell 24 ---
def scatter_dates(ax,x,y,title,xlabel):
    g=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
    ax.scatter(g.x,g.y,c='#267c99',s=50)
    for dt,row in g.iterrows():ax.annotate(dt.strftime('%m/%d'),(row.x,row.y),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(title=title,xlabel=xlabel,ylabel='날짜별 기록 불량률 (%)');ax.grid(alpha=.2)
    ax.margins(.16)
fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
rr=assoc(focus_daily.feature_mean,daily.defect_rate_pct)
scatter_dates(ax,focus_daily.feature_mean,daily.defect_rate_pct,f'B · N=50 · {FOCUS}\n8일 ρ={rr["rho"]:.3f} / 7일 ρ={rr["rho_7"]:.3f}',f'Rolling {FOCUS}의 날짜 평균 ({feature_unit(FOCUS)})')
save_show()

# --- reproduced from notebook code cell 26 ---
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for ax,scenario in zip(axes,['B','A']):
    g=association.query("variant=='original' and scenario==@scenario and feature==@FOCUS and metric=='feature_mean'").sort_values('window')
    ax.plot(g.window,g.rho,'o-',label='8일 ρ');ax.plot(g.window,g.rho_7,'s--',label='3/31 제외 7일 ρ')
    ax.fill_between(g.window,g.loo_min,g.loo_max,alpha=.18,label='날짜 하나 제외 범위')
    ax.axhline(0,color='grey',lw=.7);ax.set(xticks=WINDOWS,ylim=(-1,1),xlabel='최근 창 크기 N (용접 기록 개수)',ylabel='Spearman ρ (무차원)',title=f'{SCENARIOS[scenario]} · {FOCUS}');ax.legend(fontsize=8)
save_show()

# --- reproduced from notebook code cell 28 ---
fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
features=['current_mean','current_std','current_iqr','current_mad','delta_current_mean','delta_current_p90']
x=np.arange(len(features))
for i,n in enumerate([20,50,100]):
    vals=association.query("variant=='original' and scenario=='B' and window==@n and metric=='feature_mean'").set_index('feature').reindex(features).rho
    ax.bar(x+(i-1)*.25,vals,width=.25,label=f'N={n}')
ax.set(xticks=x,xticklabels=['평균','표준편차','IQR','MAD','|Δ| 평균','|Δ| p90'],ylabel='기록 불량률과 Spearman ρ (무차원)',xlabel='Current rolling 변수 (날짜 평균으로 요약)',title='Current: 평균 vs 변동성 · B · 8일');ax.axhline(0,color='grey',lw=.7);ax.legend();save_show()

# --- reproduced from notebook code cell 30 ---
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for ax,base in zip(axes,['power','resistance']):
    for stat in ['mean','std','iqr','mad']:
        f=base+'_'+stat
        g=association.query("variant=='original' and scenario=='B' and window!=200 and feature==@f and metric=='feature_mean'").sort_values('window')
        ax.plot(g.window,g.rho,'o-',label=stat)
    ax.set(xticks=[20,50,100],ylim=(-1,1),xlabel='창 N (용접 기록 개수)',ylabel='기록 불량률과 Spearman ρ (무차원)',title=f'{base} proxy · B · 8일');ax.axhline(0,color='grey',lw=.7);ax.legend()
save_show()

# --- reproduced from notebook code cell 32 ---
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
g=pd.DataFrame({'ae':ae_primary.ae_anomaly_pct,'roll':focus_daily.warning95_rate}).reindex(DATES8)
axes[0].scatter(g.ae,g.roll)
for dt,row in g.iterrows():axes[0].annotate(dt.strftime('%m/%d'),(row.ae,row.roll),xytext=(3,3),textcoords='offset points',fontsize=8)
axes[0].set(xlabel='AE 후보 비율 (%)',ylabel='Rolling 95백분위 초과 비율 (%)',title=f'두 지표 간 ρ={rho(g.ae,g.roll):.3f}');axes[0].margins(.18)
scatter_dates(axes[1],ae_primary.ae_anomaly_pct,daily.defect_rate_pct,'AE 후보 비율 · seed 42','AE 후보 비율 (%)')
scatter_dates(axes[2],focus_daily.warning95_rate,daily.defect_rate_pct,f'B N=50 · {FOCUS}','Rolling 경고율 (%)')
save_show()

# --- reproduced from notebook code cell 34 ---
heat_features=['current_std','delta_current_mean','delta_current_p90','power_iqr','resistance_iqr','voltage_std','voltage_iqr']
h=candidate_corr.loc[heat_features,heat_features]
fig,ax=plt.subplots(figsize=(10,7),layout='constrained');im=ax.imshow(h,vmin=-1,vmax=1,cmap='RdBu_r')
ax.set(xticks=np.arange(len(h)),yticks=np.arange(len(h)),xticklabels=h.columns,yticklabels=h.index,title='후보 중복성 · B N=50 날짜 평균 · 8일',xlabel='공정 변동성 지표',ylabel='공정 변동성 지표')
plt.setp(ax.get_xticklabels(),rotation=35,ha='right')
for i in range(len(h)):
    for j in range(len(h)):ax.text(j,i,f'{h.iloc[i,j]:.2f}',ha='center',va='center',color='white' if abs(h.iloc[i,j])>.65 else 'black')
fig.colorbar(im,ax=ax,label='Spearman ρ (무차원)');save_show();display(candidate_corr.round(3))

# --- reproduced from notebook code cell 36 ---
EXAMPLE=pd.Timestamp('2020-04-03')
fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained',sharex=True)
for ax,scenario in zip(axes,['B','A']):
    frame=variants['original'];mask=frame.date.eq(EXAMPLE)
    vals=rolling[('original',scenario,50)].loc[mask,FOCUS]
    rr=daily_roll.query("variant=='original' and scenario==@scenario and window==50 and feature==@FOCUS and date==@EXAMPLE").iloc[0]
    positions=frame.loc[mask,'position']
    ax.plot(positions,vals,lw=1,label=FOCUS)
    ax.axhline(rr.threshold95,color='#d3942b',ls='--',label='Train 95백분위');ax.axhline(rr.threshold99,color='#b44144',ls=':',label='Train 99백분위')
    warning=vals.gt(rr.threshold95);ax.scatter(positions[warning],vals[warning],s=8,c='#b44144',label='경고 후보')
    ax.set(title=f'04/03 · {SCENARIOS[scenario]} · N=50',ylabel=f'{FOCUS}\n({feature_unit(FOCUS)})');ax.legend(fontsize=8,loc='upper right')
axes[-1].set_xlabel('날짜 내부 원본 Excel 기록 순번 (행)');save_show()

# --- reproduced from notebook code cell 38 ---
def av(f,metric='feature_mean',n=50,scenario='B',variant='original'):
    return association.query('variant==@variant and scenario==@scenario and window==@n and feature==@f and metric==@metric').iloc[0]
focus_stat=av(FOCUS);mean_stat=av('current_mean');current_stat=av('current_std')
power_stat=av('power_iqr');res_stat=av('resistance_iqr');warn_stat=av(FOCUS,'warning95_rate')
ae_stat=assoc(ae_primary.ae_anomaly_pct,daily.defect_rate_pct)
focus_ae_rho=rho(focus_daily.feature_mean,ae_primary.ae_anomaly_pct)
window_stability=[]
for n in [20,50,100]:
    g=association.query("variant=='original' and feature==@FOCUS and metric=='feature_mean' and window==@n")
    window_stability.append({'N':n,'A/B 최소 rho':g.rho.min(),'LOO 최소 rho':g.loo_min.min(),'7일 최소 rho':g.rho_7.min()})
window_table=pd.DataFrame(window_stability)
best_robust_n=int(window_table.sort_values(['LOO 최소 rho','A/B 최소 rho'],ascending=False).iloc[0].N)
recommend='N=50 (사전 고정 비교용; 배포 권고는 보류)' if selected else '확정 보류 (N=20/50/100 추가 날짜 검증 필요)'
warning_ae_rho=rho(focus_daily.warning95_rate.reindex(DATES8),ae_primary.ae_anomaly_pct.reindex(DATES8))
additional=f'AE 후보 비율과 {FOCUS} 날짜 평균 간 ρ={focus_ae_rho:.3f}, rolling 경고율과는 ρ={warning_ae_rho:.3f}; 연속 값의 움직임과 경고 순위 중복을 구분해야 하며 결합 효용은 미검증'
score_result_text=score_status
if score_assoc:
    score_main=pd.DataFrame(score_assoc).query("variant=='original' and scenario=='B'").set_index('metric')
    score_result_text=f"결합 점수 날짜 평균 ρ={score_main.loc['mean_score','rho']:.3f} (LOO {score_main.loc['mean_score','loo_min']:.3f}~{score_main.loc['mean_score','loo_max']:.3f}, 7일 {score_main.loc['mean_score','rho_7']:.3f}), 95 경고율 ρ={score_main.loc['warning95_rate','rho']:.3f}, 99 경고율 ρ={score_main.loc['warning99_rate','rho']:.3f}. 평균의 연관성이 경고율 개선으로 이어진 것은 아니다."
qanswers=[
f'Q1. 최근 창에서도 관계가 유지됐는가? B N=50에서 power IQR ρ={power_stat.rho:.3f}, resistance IQR ρ={res_stat.rho:.3f}; 안정성 필터 통과 후 중복 축소 후보 {len(selected)}개이며 개별 조기검출 근거는 아니다.',
f'Q2. 어떤 창이 안정적인가? {FOCUS}의 날짜 제거 최소 ρ 기준 관측상 N={best_robust_n}이 가장 높았으나 같은 8일을 이용한 비교이다. 추천: {recommend}.',
f'Q3. 유망한 실시간 feature는? '+(', '.join(selected) if selected else '엄격한 안정성 기준을 모두 충족한 최종 후보 없음')+f'; 설명용 {FOCUS} B50 ρ={focus_stat.rho:.3f}, LOO {focus_stat.loo_min:.3f}~{focus_stat.loo_max:.3f}, 7일 {focus_stat.rho_7:.3f}.',
f'Q4. 평균보다 변동성이 유용했는가? Current 평균 ρ={mean_stat.rho:.3f}, std ρ={current_stat.rho:.3f}; power/resistance IQR은 각각 {power_stat.rho:.3f}/{res_stat.rho:.3f}. 일부 관계의 차이이며 모든 변동성 변수의 우월성을 뜻하지 않는다.',
f'Q5. AE보다 추가 정보가 있는가? {additional}. 불량률과 AE 비율 ρ={ae_stat["rho"]:.3f}, rolling 경고율 ρ={warn_stat.rho:.3f}; 예측 성능 우열은 판정할 수 없다.',
'Q6. 실제 개별 불량 사전 판정이 가능한가? 불가능하다. 제품별 품질·검사시점 연결 정보가 없고 날짜별 집계만 있어 사전경고 후보의 탐색까지만 가능하다.'
]
display(Markdown('\n\n'.join(qanswers)))
display(window_table.round(4))
core=association.query("variant=='original' and scenario=='B' and window==50 and metric=='feature_mean'").set_index('feature').loc[['current_mean','current_std','delta_current_mean','delta_current_p90','power_iqr','resistance_iqr'],['rho','loo_min','loo_max','rho_7']].reset_index()
summary=f"""# Rolling 공정 변동성 기반 용접 불량 위험 사전경고 가능성 검증

## 분석 목적
현재까지 수집된 최근 용접 데이터로 계산 가능한 변동성이 날짜별 기록 불량률과 연결되는지 탐색한다. 개별 불량 분류는 하지 않는다.

## 기존 분석에서 출발한 가설
02의 하루 전체 power proxy IQR ρ=0.738, resistance proxy IQR ρ=0.690, current mean ρ=0.024에서 출발했다. 하루 전체 집계 대신 최근 창에서도 관계가 유지되는지 확인한다. 02 수치를 재계산하여 일치함을 확인했다.

## 분석 방법
- Raw {len(d):,}행·9일, result {len(result)}행. 기록 불량률 분석 8일; 3/31 제외 민감도 7일. 3/27 미기록은 0으로 채우지 않는다.
- 원본 Excel 행 순서, 날짜 변경마다 창 초기화. 의심 날짜 행 3552·5201은 원본 유지 분석과 제외 분석을 병행했다. 제외 시 Raw {len(variants['exclude_suspect']):,}행으로 생산 분모도 다시 계산했다.
- B(현재 포함) 주 분석, A(직전까지) 보조. N=20/50/100, 참고 200; min_periods=N, 초기 NaN 유지.
- Train 날짜 rolling 분포의 95/99백분위 초과로 경고; 분모는 유효 행 수. 날짜 제외는 회고적 참조 방식이며 미래 배포 검증은 아니다.
- 8일 ρ, 한 날짜 제거 범위, 7일 ρ, 두 시나리오와 창·의심행 민감도를 함께 검토했다. 후보 선택은 일별 feature 평균, N=50 고정으로 비교하고 중복 |ρ|≥0.9를 축소했다.

## 핵심 결과
### 표 1 · B N=50 대표 지표 (모든 창×변수 표는 Notebook에 수록)
{mdtable(core)}

### 창 크기 민감도 — {FOCUS}
{mdtable(window_table)}

### 표 3 · 최종 사전경고 후보
{mdtable(table3) if not table3.empty else '모든 안정성 기준을 통과한 중복 제거 후보 없음. 후보 확정 보류.'}

{chr(10).join('- '+q for q in qanswers[:4])}

의심행 제외 민감도: {FOCUS} B50 원본 ρ={focus_stat.rho:.3f}, 제외 ρ={av(FOCUS,variant='exclude_suspect').rho:.3f}. 두 행을 다른 날짜로 정정한 결과는 아니다.

**Process Instability Score:** {score_result_text}

## AutoEncoder와 비교
### 표 2 · 같은 날짜 기준 비교
{mdtable(table2)}

AE는 02의 9일 제외 × 3 seeds를 재현했다. 다른 날짜의 동일 센서 조합 중복률은 {folds.pattern_overlap_pct.min():.1f}~{folds.pattern_overlap_pct.max():.1f}%다. AE 후보 비율과 반복 Force>3 집단 날짜 비율의 ρ={pattern_daily_rho:.3f}, 불량률과는 ρ={ae_stat['rho']:.3f}다. 행별 반복 빈도와 AE 상대오차 ρ={pattern_rho:.3f}이며 서로 다른 분석 단위이므로 직접 성능 비교하지 않는다. 반복 집단과 함께 움직이는 경향과 불량 판별 능력을 구분한다.

{qanswers[4]}

- AE는 반복 공정패턴에 더 민감한가? 반복 Force 집단 비율과 AE 후보 비율의 날짜 순위는 ρ={pattern_daily_rho:.3f}로, 불량률과의 ρ={ae_stat['rho']:.3f}보다 가깝다. 다만 반복 패턴 일반에 대한 민감도나 원인까지 입증한 것은 아니다.
- Rolling은 더 일관적인가? {FOCUS} 평균의 LOO 최소 ρ={focus_stat.loo_min:.3f}, AE 비율의 최소 ρ={ae_stat['loo_min']:.3f}이다. 경고율에서는 연관성이 더 좋아졌다는 근거가 없다.
- 다른 정보인가? 날짜 평균과 AE 비율의 ρ={focus_ae_rho:.3f}는 완전한 중복은 아니지만, 경고율 간 ρ={warning_ae_rho:.3f}로 중복을 별도로 고려해야 한다.
- 함께 쓸 가치가 있는가? 연속 지표를 병렬로 관찰하는 후속 검증 가치는 있으나, 개별 품질 정보 없이 결합의 효과를 입증할 수 없다.

4/3의 01 가이드 후보 14건, 02 날짜 제외 후보 13~15건, 기록 불량 4건은 동일 제품으로 연결하지 않았다. 02 Δ는 고유·연속 idx, 03 Δ는 원본 Excel 인접 기록이므로 값이 달라도 계산 오류를 뜻하지 않는다. 전체일 IQR과 rolling IQR은 집계 방식·초기 유효 행이 달라 같은 수치를 기대하지 않는다.

## 실시간 적용 가능성
센서 수집 → 최근 N개 변동성 계산 → 참조 임계값과 비교 → 공정 불안정성 알림을 검토할 수 있다. 초기 창 미충족은 `WARM-UP/평가 불가`, 유효한 값이 95 경계 이하이면 `NORMAL`, 95 초과이면 `WATCH`, 99 초과이면 `HIGH RISK`라는 UI 예시를 사용할 수 있으나 NORMAL도 품질 보증이 아니다. 배포 시에는 과거에 확보한 참조 날짜만 사용하고 센서 결측/날짜 경계를 처리해야 한다.

추천 window: {recommend}. {score_status}. 점수의 가중치/참조 scaling은 불량률로 최적화하지 않았지만 후보 선택이 전체 8일에 의존하므로 선택 편향이 남는다.

## 현재 한계
- 날짜 7~8개, 서로 겹치는 창과 반복 센서 조합, 다수 지표 탐색·후보 선택 편향으로 높은 상관이 우연일 수 있다.
- 제품 label, 정확한 검사시점·발생시점이 없어 lead time·개별 불량 검출 성능을 판단할 수 없다. 하루 평균/경고율은 사후 날짜 집계다.
- 3/27 미기록·3/31 유형 누락, 날짜 오류 의심과 idx 중복, 행 순서의 실제 생산 순서 여부가 남아 있다.
- 기록 불량 합계는 고유 불량 제품 수인지 확인되지 않았으며 Raw 행 수는 생산량의 대리량이다.
- LODO 참조는 미래 날짜를 포함하고 참조일 생산량에 따라 가중된다. 별도 시간 순서 검증이 필요하다.
- Proxy는 실제 접촉저항/정확한 열에너지 측정치가 아니다. 센서 변동 자체가 원인인지도 알 수 없다.

## 다음 단계
원본 날짜와 생산 순서를 확인하고, 새로운 날짜의 센서·제품 ID·검사시점·품질 결과를 연결한다. 후보와 창을 고정한 뒤 과거 날짜만 참조하는 시간 순서 검증으로 경고 빈도·선행시간·오경고 부담을 확인한다. 결과를 보기 전 프로토콜을 고정하고 기존 8일은 탐색 자료로 남긴다.

## 팀 공유용 결론
1. 최근 창 지표와 기록 불량률 관계는 위 수치로 관찰됐지만 개별 제품의 결과를 설명하지 않는다.
2. 중복 축소 안정 후보는 {len(selected)}개이며 {', '.join(selected) if selected else '후보 확정을 보류했다'}.
3. AE 후보 비율 ρ={ae_stat['rho']:.3f}, {FOCUS} 경고율 ρ={warn_stat.rho:.3f}이고 결합 효용은 아직 검증되지 않았다.
4. 날짜·창·시나리오·의심행 민감도를 함께 제시했으며 최고 ρ 하나로 최적 설정을 정하지 않았다.
5. 현재 제안은 품질검사 전 공정 불안정성 사전경고 후보이고 실제 개별 불량 판정은 불가능하다.
"""
if not score_daily.empty:summary+='\n\n### 결합 점수의 날짜별 연관성 (탐색)\n'+mdtable(pd.DataFrame(score_assoc))+'\n'
display(Markdown(summary))
SUMMARY_PATH.write_text(summary,encoding='utf-8')

# --- reproduced from notebook code cell 40 ---
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in protected_hash.items())
assert len(DATES8)==8 and len(DATES7)==7
assert association.n.eq(8).all()
assert (daily_roll.warning95_count<=daily_roll.valid_n).all()
assert len(selected)<=3
assert SUMMARY_PATH.read_text(encoding='utf-8')==summary
for forbidden in ['불량 예측 성공','정확도 향상','불량 원인을 규명','최적 공정조건 발견']:
    assert forbidden not in summary
print('03 분석 완료')
print('가장 안정적인 후보 feature:',', '.join(selected) if selected else '확정 보류; 필터를 모두 통과한 후보 없음')
print('추천 window:',recommend)
print('AE 대비 추가 정보 여부:',additional)
print('개별 불량 판정 가능 여부: 불가능 — 제품별 label 및 검사시점 없음')
print('원본/기존 Notebook 변경 0개; 요약과 계산 수치 연결 확인')

print(f'Completed {__file__}')
print(f'Reproduced figures: {_FIGURE_INDEX}/{len(FIGURE_NAMES)} in {REPRO_OUT}')