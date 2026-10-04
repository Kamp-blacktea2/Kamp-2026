"""
Post-hoc reproduction script for NBJ-001.

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
    _TMP_DIR = tempfile.TemporaryDirectory(prefix="nbj-001_")
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


FIGURE_NAMES = ['input_sensor_distribution_reproduced.png', 'train_loss_curve_reproduced.png', 'reconstruction_error_distribution_reproduced.png']
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


parser = argparse.ArgumentParser(description="Reproduce NBJ-001 from the original KAMP welding workbook.")
parser.add_argument(
    "--data",
    default=None,
    help="Path to Welding Data Set_01.xlsx or the original welding dataset ZIP. "
         "If omitted, common repo locations and ~/Downloads are searched.",
)
ARGS = parser.parse_args()
DATA_PATH_OVERRIDE = resolve_data_path(ARGS.data)

# --- reproduced from notebook code cell 2 ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
device = torch.device('cpu')

DATA_PATH = Path(DATA_PATH_OVERRIDE)
source_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
raw = pd.read_excel(DATA_PATH, sheet_name='Raw data')
FEATURES = ['weld force(bar)', 'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']
LABELS = ['가압력 (bar)', '전류 (kA)', '전압 (V)', '통전시간 (ms)']
versions = {name: metadata.version(name) for name in ['numpy','pandas','openpyxl','matplotlib','torch']}
versions['Python'] = platform.python_version()
print('원본:', DATA_PATH)
print('SHA-256:', source_hash)
print('실행 환경:', versions)
print('seed:', SEED, '| device:', device)
if any(f.name == 'Malgun Gothic' for f in font_manager.fontManager.ttflist):
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 110
pd.set_option('display.max_columns', 20)
display(raw.head())

# --- reproduced from notebook code cell 4 ---
assert len(raw) == 11939, f'가이드와 다른 행 수: {len(raw)}'
assert set(FEATURES).issubset(raw.columns)
X = raw[FEATURES].apply(pd.to_numeric, errors='raise').copy()
assert np.isfinite(X.to_numpy()).all(), '결측/무한대가 있습니다. 임의 처리 없이 확인해야 합니다.'
row_info = raw[['idx', 'working time']].copy()
row_info.insert(0, 'Excel 행 번호', np.arange(len(raw)) + 2)
print(f'입력 {len(X):,}행 × {X.shape[1]}변수 / 삭제·대체한 행 0개')
print('날짜 수:', raw['working time'].nunique())
display(pd.DataFrame({'변수': FEATURES, '단위': ['bar','kA','V','ms'], '결측 수': X.isna().sum().values}))
display(X.describe().T)

# --- reproduced from notebook code cell 6 ---
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, col, label in zip(axes.flat, FEATURES, LABELS):
    ax.hist(X[col], bins=40, color='#3579ad', edgecolor='white', linewidth=.4)
    ax.set(title=f'{label} 분포 (n={len(X):,})', xlabel=label, ylabel='행 수 (건)')
    ax.grid(axis='y', alpha=.2)
fig.suptitle('Guide baseline 입력 센서 분포', fontsize=15)
fig.tight_layout()
save_show()
plt.close(fig)

# --- reproduced from notebook code cell 8 ---
data_min = X.min().to_numpy()
data_max = X.max().to_numpy()
data_range = data_max - data_min
assert (data_range > 0).all(), '상수 열의 처리 방식을 별도 확인해야 합니다.'
scale = 1.0 / data_range
offset = -data_min * scale
scaled_values = X.to_numpy() * scale + offset
scaled_X = pd.DataFrame(scaled_values, columns=FEATURES, index=X.index)
scaling_reference = pd.DataFrame({'변수': FEATURES, '전체 최소': data_min, '전체 최대': data_max})
np.testing.assert_allclose(scaled_values, (X.to_numpy()-data_min)/data_range, atol=1e-12, rtol=0)
display(scaling_reference)
display(scaled_X.head())
assert np.isfinite(scaled_values).all()
assert np.allclose(scaled_values.min(axis=0), 0)
assert np.allclose(scaled_values.max(axis=0), 1)

# --- reproduced from notebook code cell 10 ---
class GuideAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 3), nn.RReLU(),
            nn.Linear(3, 2), nn.RReLU(),
            nn.Linear(2, 3), nn.RReLU(),
            nn.Linear(3, 4)
        )
    def forward(self, inputs):
        return self.network(inputs)

model = GuideAutoEncoder().to(device)
print(model)
print('학습 파라미터 수:', sum(p.numel() for p in model.parameters()))

# --- reproduced from notebook code cell 12 ---
SPLIT = 8470
train_data = torch.tensor(scaled_values[:SPLIT], dtype=torch.float32, device=device)
test_data = torch.tensor(scaled_values[SPLIT:], dtype=torch.float32, device=device)
assert train_data.shape == (8470, 4) and test_data.shape == (3469, 4)
print('Train:', tuple(train_data.shape), '| Test:', tuple(test_data.shape))
split_info = row_info.copy()
split_info['구간'] = np.where(np.arange(len(raw)) < SPLIT, 'Train', 'Test')
display(split_info.groupby(['구간', 'working time'], sort=False).size().rename('행 수').to_frame())

# --- reproduced from notebook code cell 14 ---
EPOCHS, BATCH_SIZE, LR = 50, 64, 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
history = []
model.train()
for epoch in range(1, EPOCHS + 1):
    batch_loss_sum, weighted_loss_sum, seen = 0.0, 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        batch_loss_sum += loss.item()
        weighted_loss_sum += loss.item() * len(batch)
        seen += len(batch)
    history.append({'epoch': epoch, 'guide_batch_loss_sum': batch_loss_sum,
                    'mean_train_MSE': weighted_loss_sum / seen})
    if epoch == 1 or epoch % 10 == 0:
        print(f'epoch {epoch:2d}: 평균 Train MSE={weighted_loss_sum / seen:.8f}, 가이드식 합={batch_loss_sum:.6f}')
history = pd.DataFrame(history)
display(history.tail())
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(history['epoch'], history['mean_train_MSE'], color='#3579ad')
ax.set(title=f'Train 학습 곡선 (n={len(train_data):,})', xlabel='학습 반복 (epoch)', ylabel='정규화 MSE (무차원)')
ax.grid(alpha=.2)
save_show()
plt.close(fig)

# --- reproduced from notebook code cell 16 ---
torch.manual_seed(SEED)
assert model.training
def guide_row_errors(data):
    errors = []
    with torch.no_grad():
        for row in data:
            errors.append(criterion(model(row), row).item())
    return np.asarray(errors, dtype=np.float64)

train_errors = guide_row_errors(train_data)
test_errors = guide_row_errors(test_data)
assert len(train_errors) == SPLIT and len(test_errors) == len(raw) - SPLIT
assert np.isfinite(train_errors).all() and np.isfinite(test_errors).all()
error_stats = pd.DataFrame({
    'Train': pd.Series(train_errors).describe(percentiles=[.25,.5,.75,.95,.99]),
    'Test': pd.Series(test_errors).describe(percentiles=[.25,.5,.75,.95,.99])
}).T
display(error_stats)
print('표의 std는 표본 표준편차이며, 다음 threshold는 가이드 np.std 기본값(ddof=0)을 사용합니다.')

# --- reproduced from notebook code cell 18 ---
threshold = float(np.mean(train_errors) + 8 * np.std(train_errors, ddof=0))
train_flags = train_errors >= threshold
test_flags = test_errors >= threshold
counts = pd.DataFrame([
    {'구간': 'Train', '행 수': len(train_errors), '이상치 후보 수': int(train_flags.sum()), '후보 비율 (%)': train_flags.mean()*100},
    {'구간': 'Test', '행 수': len(test_errors), '이상치 후보 수': int(test_flags.sum()), '후보 비율 (%)': test_flags.mean()*100}
])
print(f'Threshold = {threshold:.10f}')
display(counts)

# --- reproduced from notebook code cell 20 ---
combined_errors = np.concatenate([train_errors, test_errors])
plot_max = max(float(combined_errors.max()), threshold) * 1.05
bins = np.linspace(0, max(plot_max, 1e-12), 65)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
for values, label, color in [(train_errors, 'Train', '#3579ad'), (test_errors, 'Test', '#d68138')]:
    axes[0].hist(values, bins=bins, weights=np.full(len(values), 100/len(values)),
                 histtype='step', linewidth=1.6, color=color, label=f'{label} (n={len(values):,})')
    ordered = np.sort(values)
    axes[1].step(np.maximum(ordered, 1e-12), np.arange(1,len(values)+1)/len(values)*100,
                 where='post', color=color, label=f'{label} (n={len(values):,})')
for ax in axes:
    ax.axvline(threshold, color='#b4414b', linestyle='--', label='Train 평균 + 8×표준편차')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
axes[0].set(title='복원오차의 구간별 비율', xlabel='정규화 MSE (무차원)', ylabel='구간 내 비율 (%)')
axes[1].set(title='복원오차 누적분포', xlabel='정규화 MSE (무차원, 로그 눈금)', ylabel='누적 비율 (%)', xscale='log')
axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
fig.suptitle('Guide baseline: Train / Test reconstruction error')
fig.tight_layout()
save_show()
plt.close(fig)

# --- reproduced from notebook code cell 22 ---
test_rows = pd.concat([row_info.iloc[SPLIT:], X.iloc[SPLIT:]], axis=1).copy()
test_rows['reconstruction_error'] = test_errors
test_rows['이상치 후보'] = test_flags
candidate_rows = test_rows.loc[test_rows['이상치 후보']].sort_values('reconstruction_error', ascending=False)
print(f'Test 이상치 후보: {len(candidate_rows):,}건 / {len(test_rows):,}건')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(candidate_rows)
daily_candidates = test_rows.groupby('working time').agg(행수=('이상치 후보','size'), 후보수=('이상치 후보','sum'))
daily_candidates['후보 비율 (%)'] = daily_candidates['후보수']/daily_candidates['행수']*100
display(daily_candidates)
if len(candidate_rows):
    display(candidate_rows[FEATURES].agg(['min','median','max']).T.rename(columns={'min':'후보 최소','median':'후보 중앙값','max':'후보 최대'}))
    pattern_counts = candidate_rows.groupby(FEATURES).size().sort_values(ascending=False).rename('후보 행 수').reset_index()
    print('후보 내 빈도가 높은 센서 조합 (최대 10개):')
    display(pattern_counts.head(10))
else:
    print('이 기준에서 후보가 없어 후보 공정조건을 요약할 수 없습니다. 불량이 없다는 뜻은 아닙니다.')
assert hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() == source_hash
print('원본 Excel 해시 일치: 변경 없음')

# --- reproduced from notebook code cell 24 ---
summary = [
    '**1. 무엇을 학습했나요?** 전체 원본을 Min-Max 변환한 센서 4개 중 앞 8,470행의 값을 복원하도록 학습했습니다. 정상/불량 정답은 학습하지 않았습니다.',
    f'**2. 후보는 몇 개인가요?** threshold {threshold:.8f}에서 Train {int(train_flags.sum()):,}/{len(train_flags):,}건, Test {int(test_flags.sum()):,}/{len(test_flags):,}건({test_flags.mean()*100:.3f}%)입니다.'
]
if len(candidate_rows):
    max_count = int(daily_candidates['후보수'].max())
    top_days = daily_candidates.index[daily_candidates['후보수'].eq(max_count)]
    dates_text = ', '.join(pd.Timestamp(day).strftime('%Y-%m-%d') for day in top_days)
    conditions = '; '.join(f'{label}: {candidate_rows[col].min():.4g}~{candidate_rows[col].max():.4g}' for col,label in zip(FEATURES,LABELS))
    summary.append(f'**3. 어떤 조건에서 나왔나요?** Test 후보 수가 가장 많은 날짜는 {dates_text}(각 {max_count}건)입니다. 후보들의 관측 범위는 {conditions}입니다. 이 범위의 모든 조합이 후보라는 뜻은 아니며, 표본 수로 나눈 날짜별 후보 비율과 실제 조합 표를 함께 봐야 합니다.')
else:
    summary.append('**3. 어떤 조건에서 나왔나요?** Test 후보가 0건이므로 집중 조건을 제시하지 않습니다. 이 threshold가 잡지 못하는 조건이 있거나 경계가 높을 가능성도 있어 정상 품질의 증거가 아닙니다.')
summary.append('**4. 왜 실제 불량이라고 할 수 없나요?** 제품별 검사 정답이 없습니다. 큰 오차는 이 모델의 복원이 어렵다는 뜻이며, 드문 조건·학습 상태·스케일 등의 영향을 받을 수 있습니다. 작은 오차도 실제 품질을 보증하지 않습니다.')
summary_text = '\n\n'.join(summary)
display(Markdown(summary_text))

print(f'Completed {__file__}')
print(f'Reproduced figures: {_FIGURE_INDEX}/{len(FIGURE_NAMES)} in {REPRO_OUT}')