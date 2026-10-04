"""
Post-hoc reproduction script for NBJ-002.

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
    _TMP_DIR = tempfile.TemporaryDirectory(prefix="nbj-002_")
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


FIGURE_NAMES = ['daily_defect_and_ae_summary_reproduced.png', 'sensor_defect_rate_scatter_reproduced.png', 'sensor_metric_correlation_heatmap_reproduced.png', 'defect_type_correlation_heatmap_reproduced.png', 'ae_seed_sensitivity_reproduced.png']
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


parser = argparse.ArgumentParser(description="Reproduce NBJ-002 from the original KAMP welding workbook.")
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

# --- reproduced from notebook code cell 4 ---
daily=d.groupby('date').size().rename('production_rows').to_frame()
daily['recorded_defects']=result.groupby('date')['defect'].sum(min_count=1)
daily['defect_rate_pct']=daily['recorded_defects']/daily['production_rows']*100
types={1:('indent','파임불량'),2:('insufficient','용접부족'),3:('crack','크랙발생')}
type_wide=result.pivot(index='date',columns='defect type',values='defect')
for k,(name,label) in types.items():
    daily[name+'_count']=type_wide[k]
    daily[name+'_rate_pct']=daily[name+'_count']/daily['production_rows']*100
daily['recorded_type_count']=result.groupby('date')['defect type'].nunique().reindex(daily.index,fill_value=0)
quality_dates=daily.index[daily['defect_rate_pct'].notna()]
assert len(quality_dates)==8
assert pd.isna(daily.loc['2020-03-27','defect_rate_pct'])
assert pd.isna(daily.loc['2020-03-31','crack_count'])
assert daily.loc['2020-04-07','indent_count']==0
assert daily['recorded_defects'].sum()==39
display(daily.rename(columns={'production_rows':'생산 기록 수','recorded_defects':'기록 불량 합계','defect_rate_pct':'기록 불량률 (%)','indent_count':'파임불량 수','insufficient_count':'용접부족 수','crack_count':'크랙발생 수'}))
print('불량률 분석: 기록 있는 8일 / 유형별 유효 날짜: 파임 8일, 용접부족 8일, 크랙 7일')

# --- reproduced from notebook code cell 6 ---
LABELS={}
for key in SHORT:
    for stat in ['mean','std','iqr']:
        LABELS[f'{key}_{stat}']=f'{key} {stat} ({UNITS[key]})'
    g=d.groupby('date')[key]
    daily[key+'_mean']=g.mean()
    daily[key+'_std']=g.std(ddof=1)
    daily[key+'_iqr']=g.quantile(.75)-g.quantile(.25)
daily['force_outside_pct']=d.assign(flag=~d['force'].between(2,2.6,inclusive='both')).groupby('date')['flag'].mean()*100
daily['force_gt3_pct']=d.assign(flag=d['force']>3).groupby('date')['flag'].mean()*100
LABELS.update({'force_outside_pct':'Force [2.0,2.6] 밖 (%)','force_gt3_pct':'Force >3 bar (%)'})
assert (d['current']>0).all()
d['resistance_proxy']=d['voltage']/d['current']
d['power_proxy']=d['voltage']*d['current']
d['energy_proxy']=d['voltage']*d['current']*d['time']
proxy_units={'resistance_proxy':'mΩ 상당','power_proxy':'kW 상당','energy_proxy':'J 상당'}
for key,unit in proxy_units.items():
    g=d.groupby('date')[key]
    daily[key+'_mean']=g.mean()
    daily[key+'_std']=g.std(ddof=1)
    daily[key+'_iqr']=g.quantile(.75)-g.quantile(.25)
    for stat in ['mean','std','iqr']: LABELS[key+'_'+stat]=f'{key.replace("_proxy", " proxy")} {stat} ({unit})'
display(daily[list(LABELS)].round(6))

# --- reproduced from notebook code cell 8 ---
delta_rows=[]
for dt,group in d.groupby('date'):
    g=group.sort_values(['idx','row_id'],kind='stable').copy()
    unique_key=~g['idx'].duplicated(keep=False)
    valid=g['idx'].diff().eq(1)&unique_key&unique_key.shift(1,fill_value=False)
    row={'date':dt,'delta_valid_pairs':int(valid.sum()),'delta_excluded_pairs':len(g)-1-int(valid.sum()),'duplicate_idx_rows':int((~unique_key).sum())}
    for key in SHORT:
        values=g[key].diff().abs().loc[valid]
        row[f'delta_{key}_mean']=values.mean()
        row[f'delta_{key}_p90']=values.quantile(.90)
        row[f'delta_{key}_p95']=values.quantile(.95)
        for stat in ['mean','p90','p95']: LABELS[f'delta_{key}_{stat}']=f'|Δ{key}| {stat} ({UNITS[key]})'
    delta_rows.append(row)
daily=daily.join(pd.DataFrame(delta_rows).set_index('date'))
display(daily[['production_rows','delta_valid_pairs','delta_excluded_pairs','duplicate_idx_rows']+[k for k in LABELS if k.startswith('delta_')]].round(6))

# --- reproduced from notebook code cell 10 ---
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

# --- reproduced from notebook code cell 12 ---
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

# --- reproduced from notebook code cell 14 ---
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

# --- reproduced from notebook code cell 16 ---
def spearman(a,b):
    z=pd.concat([pd.Series(a).rename('a'),pd.Series(b).rename('b')],axis=1).dropna()
    z=z.round(12)
    if len(z)<4 or (z.nunique()<2).any():return np.nan
    return float(z.rank(method='average').corr(method='pearson').iloc[0,1])

TARGETS=['defect_rate_pct','indent_rate_pct','insufficient_rate_pct','crack_rate_pct']
TARGET_LABELS={'defect_rate_pct':'전체 기록 불량률','indent_rate_pct':'파임불량 기록률','insufficient_rate_pct':'용접부족 기록률','crack_rate_pct':'크랙 기록률'}
features=list(LABELS)
assoc=[]
for target in TARGETS:
    for feature in features:
        z=daily[[feature,target]].dropna()
        rho=spearman(z[feature],z[target])
        loo=[spearman(z.drop(dt)[feature],z.drop(dt)[target]) for dt in z.index]
        finite=[v for v in loo if np.isfinite(v)]
        assoc.append({'target':target,'feature':feature,'n_dates':len(z),'rho':rho,
          'loo_min':min(finite) if finite else np.nan,'loo_max':max(finite) if finite else np.nan,
          'loo_defined':len(finite)})
association=pd.DataFrame(assoc)
seed_corr=[]
for seed in SEEDS:
    table=ae_by_seed.loc[ae_by_seed.seed.eq(seed)].set_index('date')
    for f in ae_primary.columns:
        seed_corr.append({'seed':seed,'feature':f,'rho':spearman(table[f],daily.defect_rate_pct)})
seed_correlations=pd.DataFrame(seed_corr)
seed_stability=seed_correlations.groupby('feature').rho.agg(['min','max','count'])
summary=association.loc[association.target.eq('defect_rate_pct')].copy().set_index('feature')
def classify(feature,row):
    r=row.rho
    if not np.isfinite(r) or row.n_dates<6 or row.loo_defined!=row.n_dates:return 'C'
    lo,hi=row.loo_min,row.loo_max
    if lo<-.1 and hi>.1:return 'C'
    if feature.startswith('ae_'):
        st=seed_stability.loc[feature]
        if st['count']<len(SEEDS) or not ((st['min']>0 and r>0) or (st['max']<0 and r<0)):return 'C'
    direction_floor=lo if r>0 else -hi
    return 'A' if abs(r)>=.6 and direction_floor>=.3 else 'B'
summary['category']=[classify(f,row) for f,row in summary.iterrows()]
summary['direction']=np.where(summary.rho>=0,'지표↑ / 기록 불량률↑','지표↑ / 기록 불량률↓')
summary.loc[summary.rho.isna(),'direction']='미정의'
with pd.option_context('display.max_rows',None):
    display(summary[['n_dates','rho','loo_min','loo_max','category','direction']].round(4))
display(seed_correlations.pivot(index='feature',columns='seed',values='rho').round(4))
count_associations=[]
for target in ['recorded_defects','indent_count','insufficient_count','crack_count']:
    for f in features+['production_rows']:
        z=daily[[f,target]].dropna()
        count_associations.append({'target_count':target,'feature':f,'n_dates':len(z),'rho':spearman(z[f],z[target])})
count_associations=pd.DataFrame(count_associations)
print('아래 원시 건수 상관은 생산량 차이에 영향을 받으므로 주 해석에 쓰지 않습니다.')
display(count_associations.loc[count_associations.feature.isin(['production_rows','force_outside_pct','ae_anomaly_count','ae_anomaly_pct'])].round(4))

# --- reproduced from notebook code cell 18 ---
comparison=[]
for key in SHORT:
    fs=[key+'_mean',key+'_std',key+'_iqr',f'delta_{key}_mean',f'delta_{key}_p90',f'delta_{key}_p95']
    for f in fs:
        row=summary.loc[f]
        comparison.append({'sensor':key,'metric':f,'rho':row.rho,'abs_rho':abs(row.rho),'category':row.category})
variability_comparison=pd.DataFrame(comparison)
display(variability_comparison.round(4))
type_table=association.pivot(index='feature',columns='target',values='rho')
type_n=association.pivot(index='feature',columns='target',values='n_dates')
with pd.option_context('display.max_rows',None):display(type_table.round(4))
common_type_dates=daily.dropna(subset=['indent_rate_pct','insufficient_rate_pct','crack_rate_pct']).index
common_type_assoc=pd.DataFrame([{'feature':f,**{t:spearman(daily.loc[common_type_dates,f],daily.loc[common_type_dates,t]) for t in TARGETS},'n_dates':len(common_type_dates)} for f in features]).set_index('feature')
print('유형 간 비교용 공통 날짜 수:',len(common_type_dates))
display(common_type_assoc.round(4))
RANK_FEATURES=['defect_rate_pct','force_outside_pct','force_gt3_pct','ae_p95','ae_relative_p95','ae_anomaly_pct','current_std','current_iqr','delta_current_p95']
ranking=daily.loc[quality_dates,RANK_FEATURES].round(12).rank(ascending=False,method='average')
display(ranking)

# --- reproduced from notebook code cell 20 ---
CORE=['force_outside_pct','force_gt3_pct','force_mean','force_std','force_iqr',
      'ae_mean','ae_median','ae_p90','ae_p95','ae_p99','ae_relative_p95','ae_anomaly_count','ae_anomaly_pct',
      'current_mean','current_std','current_iqr','delta_current_mean','delta_current_p95',
      'voltage_std','time_std','resistance_proxy_mean','power_proxy_mean','energy_proxy_mean']
selected=list(dict.fromkeys(CORE+summary.index[summary.category.eq('A')].tolist()))
core_table=summary.loc[selected,['n_dates','rho','loo_min','loo_max','category','direction']].copy()
without_331=daily.drop(pd.Timestamp('2020-03-31'))
core_table['rho_without_0331']=[spearman(without_331[f],without_331.defect_rate_pct) for f in selected]
core_table.insert(0,'지표',[LABELS[f] for f in selected])
with pd.option_context('display.max_rows',None):display(core_table.round(4))
print('모든 지표의 유형별 개수/비율 연관은 count_associations, association, common_type_assoc에 보존되어 있습니다.')

# --- reproduced from notebook code cell 22 ---
fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
x=np.arange(len(daily));labels=daily.index.strftime('%m/%d')
axes[0].bar(x,daily.production_rows,color='#6789ad')
axes[0].set(title='날짜별 생산 기록 수',ylabel='Raw 행 수 (건)')
axes[1].plot(x,daily.defect_rate_pct,'o-',color='#b74549',label='전체 기록 불량률 (8일)')
for dt in quality_dates:
    j=daily.index.get_loc(dt);r=daily.loc[dt]
    axes[1].annotate(f'{int(r.recorded_defects)}/{int(r.production_rows)}',(j,r.defect_rate_pct),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
axes[1].text(daily.index.get_loc(pd.Timestamp('2020-03-27')),.04,'기록 없음',ha='center')
axes[1].set(ylabel='기록 불량률 (%)',ylim=(0,.85));axes[1].legend()
axes[2].bar(x-.18,daily.recorded_defects,width=.36,label='기록된 불량 수',color='#b74549')
axes[2].bar(x+.18,daily.ae_anomaly_count,width=.36,label='OOS AE 후보 수 (seed 42)',color='#2f80ad')
axes[2].set(ylabel='건수 (건)',xlabel='날짜 (2020년)',title='기록 불량과 AE 경보: 같은 개별 제품으로 연결되지 않음')
axes[2].set_xticks(x,labels);axes[2].legend()
for ax in axes:ax.grid(axis='y',alpha=.2)
fig.suptitle('날짜별 불량 기록과 out-of-sample AE 요약 | Raw 11,939행',fontsize=15)
fig.tight_layout();save_show();plt.close(fig)

# --- reproduced from notebook code cell 24 ---
SCATTER_FEATURES=['force_outside_pct','force_gt3_pct','ae_relative_p95','ae_anomaly_pct','current_std','delta_current_p95']
fig,axes=plt.subplots(2,3,figsize=(16,9))
for ax,f in zip(axes.flat,SCATTER_FEATURES):
    z=daily[[f,'defect_rate_pct']].dropna()
    ax.scatter(z[f],z.defect_rate_pct,color='#3579ad',s=48)
    for j,(dt,row) in enumerate(z.iterrows()):
        ax.annotate(dt.strftime('%m/%d'),(row[f],row.defect_rate_pct),xytext=(5,5 if j%2 else -12),textcoords='offset points',fontsize=8)
    ax.set(title=f'ρ={spearman(z[f],z.defect_rate_pct):.3f} | n={len(z)}일',xlabel=LABELS[f],ylabel='기록 불량률 (%)')
    ax.grid(alpha=.2);ax.margins(.15)
fig.suptitle('센서 지표와 실제 기록 불량률의 날짜별 순위 관계',fontsize=15)
fig.tight_layout();save_show();plt.close(fig)

# --- reproduced from notebook code cell 26 ---
stats=['mean','std','iqr','delta_mean','delta_p90','delta_p95']
mat=np.array([[summary.loc[f'{k}_{s}' if not s.startswith('delta_') else f'delta_{k}_{s[6:]}','rho'] for s in stats] for k in SHORT])
fig,ax=plt.subplots(figsize=(11,4.5))
cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#dddddd')
im=ax.imshow(np.ma.masked_invalid(mat),vmin=-1,vmax=1,cmap=cmap,aspect='auto')
ax.set_xticks(range(6),['평균','표준편차','IQR','|Δ| 평균','|Δ| p90','|Δ| p95'])
ax.set_yticks(range(4),['Force','Current','Voltage','Time'])
for i in range(4):
    for j in range(6):ax.text(j,i,'—' if not np.isfinite(mat[i,j]) else f'{mat[i,j]:.2f}',ha='center',va='center',color='white' if abs(mat[i,j])>.7 else 'black')
ax.set(title='평균·변동성 지표와 기록 불량률 | 동일한 8일',xlabel='날짜별 지표',ylabel='센서')
fig.colorbar(im,ax=ax,label='Spearman ρ (무차원)')
fig.tight_layout();save_show();plt.close(fig)

# --- reproduced from notebook code cell 28 ---
TYPE_FEATURES=['force_outside_pct','ae_relative_p95','ae_anomaly_pct','current_mean','current_std','current_iqr',
 'delta_current_p95','voltage_std','time_std','resistance_proxy_mean','power_proxy_mean','energy_proxy_mean']
vals=common_type_assoc.loc[TYPE_FEATURES,TARGETS].to_numpy()
fig,ax=plt.subplots(figsize=(11,9))
im=ax.imshow(np.ma.masked_invalid(vals),vmin=-1,vmax=1,cmap=cmap,aspect='auto')
ax.set_xticks(range(4),['전체 기록률','파임불량 기록률','용접부족 기록률','크랙 기록률'])
ax.set_yticks(range(len(TYPE_FEATURES)),[LABELS[f] for f in TYPE_FEATURES])
for i in range(len(TYPE_FEATURES)):
    for j in range(4):ax.text(j,i,'—' if not np.isfinite(vals[i,j]) else f'{vals[i,j]:.2f}',ha='center',va='center',color='white' if abs(vals[i,j])>.7 else 'black')
ax.set(title='유형별 기록률과 센서 지표 | 공통 7일의 탐색적 상관',xlabel='날짜별 유형 수 / 생산 기록 수',ylabel='날짜별 센서·AE 지표')
fig.colorbar(im,ax=ax,label='Spearman ρ (무차원)')
fig.tight_layout();save_show();plt.close(fig)

# --- reproduced from notebook code cell 30 ---
fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
for seed,color in zip(SEEDS,['#276d9b','#d68138','#7c5b9d']):
    table=ae_by_seed.loc[ae_by_seed.seed.eq(seed)].set_index('date').reindex(daily.index)
    for ax,f in zip(axes,['ae_anomaly_pct','ae_relative_p95']):
        rho=seed_correlations.query('seed==@seed and feature==@f').rho.iloc[0]
        ax.plot(x,table[f],marker='o',color=color,label=f'seed {seed}, ρ={rho:.3f} (8일)')
axes[0].set(title='날짜별 threshold 초과 비율',ylabel='AE 후보 비율 (%)')
axes[1].set(title='날짜별 상대 reconstruction error p95',ylabel='Train p95 대비 (배)',xlabel='날짜 (2020년)')
axes[1].set_xticks(x,labels)
for ax in axes:ax.legend();ax.grid(alpha=.2)
fig.suptitle('AE seed 민감도 | 9일 평가, 불량 연관성은 기록 있는 8일만',fontsize=15)
fig.tight_layout();save_show();plt.close(fig)

# --- reproduced from notebook code cell 32 ---
baseline_paths=[EXP_DIR.parent/'nbj-001'/'outputs'/'01_AutoEncoder_guide_baseline.ipynb']
baseline_path=next((p for p in baseline_paths if p.exists()),None)
baseline_verified=False
if baseline_path:
    prior=json.loads(baseline_path.read_text(encoding='utf-8'))
    texts=[]
    for cell in prior['cells']:
        for out in cell.get('outputs',[]):
            for value in [out.get('text',''),out.get('data',{}).get('text/markdown',''),out.get('data',{}).get('text/plain','')]:
                texts.append(''.join(value) if isinstance(value,list) else value)
    prior_text='\n'.join(texts)
    baseline_verified=('14/3,469' in prior_text and '2020-04-03' in prior_text)
print('기존 01의 저장된 출력에서 14건/4월3일 확인:',baseline_verified)
special=[{'구분':'기존 01 baseline (별도 실험)','04/03 후보 또는 기록 수':14,'설명':'전체 scaling / 원래 앞뒤 분할 / train-mode RReLU'},
         {'구분':'result 기록 불량 합계','04/03 후보 또는 기록 수':float(daily.loc['2020-04-03','recorded_defects']),'설명':'파임 2 + 용접부족 1 + 크랙 1; 제품 연결 없음'}]
for seed in SEEDS:
    row=ae_by_seed.loc[ae_by_seed.seed.eq(seed)&ae_by_seed.date.eq(pd.Timestamp('2020-04-03'))].iloc[0]
    special.append({'구분':f'현재 날짜 제외 AE, seed {seed}','04/03 후보 또는 기록 수':int(row.ae_anomaly_count),'설명':'Train-only scaling / eval+no_grad / fold threshold'})
display(pd.DataFrame(special))
display(Markdown('**기존 AE 14건과 실제 기록 4건은 서로 다른 목록입니다. 어느 후보가 실제 불량인지 현재 자료로 확인할 수 없습니다.**'))

# --- reproduced from notebook code cell 34 ---
category_names={'A':'기록 불량률과 연관이 보이는 탐색 후보','B':'공정 변화/이상도 지표지만 기록 불량률 연관이 약함','C':'현재 데이터로 판단 제한'}
for category in ['A','B','C']:
    group=summary.loc[summary.category.eq(category),['rho','n_dates','loo_min','loo_max','direction']].copy()
    group.insert(0,'지표',[LABELS[f] for f in group.index])
    display(Markdown(f'### {category}. {category_names[category]} ({len(group)}개)'))
    with pd.option_context('display.max_rows',None): display(group.round(4))
answers=[]
for label,fs in [('Force 특이집단이 많은 날 불량도 높은가?',['force_outside_pct','force_gt3_pct']),
                 ('AE가 높은 날 불량도 높은가?',['ae_mean','ae_p95','ae_relative_p95','ae_anomaly_pct'])]:
    detail='; '.join(f'{LABELS[f]}: ρ={summary.loc[f,"rho"]:.3f}, {summary.loc[f,"category"]}' for f in fs)
    answers.append({'질문':label,'관측 결과':detail})
for key in SHORT:
    g=variability_comparison.loc[variability_comparison.sensor.eq(key)].dropna(subset=['rho'])
    means=g.loc[g.metric.eq(key+'_mean')].iloc[0]
    varies=g.loc[~g.metric.eq(key+'_mean')]
    strongest=varies.loc[varies.abs_rho.idxmax()]
    answers.append({'질문':f'{key}: 평균보다 변동성이 더 관련 있는가?',
      '관측 결과':f'평균 ρ={means.rho:.3f}; 변동성 중 최대 |ρ|는 {strongest.metric}, ρ={strongest.rho:.3f}; 사후 최대값이며 우월성 검증 아님'})
display(pd.DataFrame(answers))
type_candidate_records=[]
for target in TARGETS[1:]:
    sub=association.loc[association.target.eq(target)].dropna(subset=['rho']).copy()
    sub['abs_rho']=sub.rho.abs()
    for _,r in sub.nlargest(3,'abs_rho').iterrows():
        type_candidate_records.append({'유형':TARGET_LABELS[target],'지표':LABELS[r.feature],'전체 유효 날짜 ρ':r.rho,'n_dates':r.n_dates,
         '공통 7일 ρ':common_type_assoc.loc[r.feature,target],'날짜 제외 최소':r.loo_min,'날짜 제외 최대':r.loo_max})
print('유형별 |ρ| 상위 3개: 다중 탐색의 사후 요약이며 확증적 증거가 아닙니다.')
display(pd.DataFrame(type_candidate_records).round(4))
display(Markdown('**공통 C 항목:** 개별 제품 불량 여부, 실제 불량 원인, 센서 허용범위, 실제 접촉저항/열량, 3/27 불량률, 3/31 크랙 수, 독립적인 미래 예측 성능은 현재 자료로 판단할 수 없습니다.'))

# --- reproduced from notebook code cell 36 ---
def rho_text(f):
    value=summary.loc[f,'rho']
    return '미정의' if not np.isfinite(value) else f'{value:+.3f}'
a_group=summary.loc[summary.category.eq('A')].copy()
a_group['abs_rho']=a_group.rho.abs()
if len(a_group):
    top=a_group.sort_values('abs_rho',ascending=False).head(3)
    candidate_text=' / '.join(f'{LABELS[f]}(ρ={r.rho:+.3f})' for f,r in top.iterrows())
else:candidate_text='사전 정리 규칙을 만족한 A 후보 없음'
conclusion=[
 '분석은 불량 기록이 있는 8일의 탐색적 비교이며 3/27은 제외했고, 크랙 유형은 3/31도 미기록이라 7일만 사용했습니다.',
 f'Force 임시 범위 밖 비율의 기록 불량률 상관은 ρ={rho_text("force_outside_pct")}, Force >3 bar 비율은 ρ={rho_text("force_gt3_pct")}로 관찰됐습니다.',
 f'날짜 제외 AE의 상대 p95는 ρ={rho_text("ae_relative_p95")}, 후보 비율은 ρ={rho_text("ae_anomaly_pct")}이며 seed별 상관 범위와 반복 패턴 중첩을 함께 고려해야 합니다.',
 f'후속 검증 후보의 사후 요약은 {candidate_text}이며 다중 탐색·작은 날짜 수 때문에 원인이나 예측 성능으로 확정할 수 없습니다.',
 '기존 4/3 AE 후보 14건과 기록 불량 4건은 같은 제품으로 연결되지 않으며, 실제 불량 여부·최적 공정조건은 이 결과만으로 판단할 수 없습니다.'
]
display(Markdown('\n\n'.join(f'{i+1}. {s}' for i,s in enumerate(conclusion))))

# --- reproduced from notebook code cell 38 ---
assert len(daily)==9 and daily.production_rows.sum()==11939
assert daily.defect_rate_pct.notna().sum()==8
assert daily.crack_rate_pct.notna().sum()==7
assert not oos.duplicated(['seed','row_id']).any()
assert all(oos.loc[oos.seed.eq(s),'row_id'].nunique()==11939 for s in SEEDS)
assert folds.groupby('seed').size().eq(9).all()
assert pd.isna(daily.loc['2020-03-27','recorded_defects'])
assert hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()==source_hash
assert np.isfinite(oos[['error','relative_error','threshold']].to_numpy()).all()
print('PASS: 11,939행 × 3 seeds의 날짜 제외 평가, 9 fold/seed, 원본 해시 불변, 3/27 미기록 보존')
print('실행 환경:',versions)

print(f'Completed {__file__}')
print(f'Reproduced figures: {_FIGURE_INDEX}/{len(FIGURE_NAMES)} in {REPRO_OUT}')