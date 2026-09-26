"""Text-first numerical report for YSH-003; reviewed narrative is a run snapshot."""

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TABLES = OUT / "tables"


def table(frame):
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return str(int(value)) if value == int(value) else f"{value:.4f}"
        return str(value).replace("|", "/")

    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines.extend(
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def main():
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    checks = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    scan = pd.read_csv(TABLES / "motif_scan.csv")
    candidates = pd.read_csv(TABLES / "selected_candidates.csv")
    details = pd.read_csv(TABLES / "selected_pair_details.csv")
    templates = pd.read_csv(TABLES / "template_boundary_checks.csv")
    segments = pd.read_csv(TABLES / "segments.csv")
    raw = pd.read_excel(config["raw"], sheet_name="Raw data")
    parts = []

    def add(text):
        parts.append(text.strip())

    add("""# 실험결과 — YSH-003 유사한 작업 신호와 경계 후보

**결론: 전압에는 값이 정확히 같지 않아도 16건 안팎의 모양이 국소적으로 반복되는 근거가 있다. 그러나 고정된 위상 템플릿과 네 변수 전체의 동시 반복은 약해, 제품 한 개의 경계나 제품당 16개 spot을 복원했다고 판단할 수 없다.**

계획서를 먼저 저장한 뒤 실험을 실행했다. 사용자 선호에 따라 그림·HTML 대신 상세 수치·표·해석으로 정리한다.

## Metadata

- 실험 ID: YSH-003, 담당자 YSH, 세 번째 실험.
- 상태: 완료. 유형 EDA / VALIDATION / SIGNAL_PATTERN.
- 실행일: 2026-09-23. 장치 CPU. 모델 학습 없음.
- 계획: [실험계획.md](실험계획.md). 작업 브랜치 ysh-001-plan. commit/push 없음.
- 작업 가정: 같은 품번의 부품을 컨베이어 방식으로 반복 생산. 실제 설비 배치·한 행의 물리 단위는 미확정.

## 실제 수행 내용

원본 순서의 연속 구간과 YSH-002의 controlled100 구간을 사용했다. 각 구간을 앞/뒤 절반으로 나누고 길이 N=2..100, 시작점 0과 floor(N/2)에서 인접한 비중첩 블록을 비교했다. 인접 비교 쌍은 블록을 공유할 수 있으므로 독립 표본이 아니다.

원시값 RMSE, 평균 제거 RMSE, 모양 상관, 대칭 모양 오차, 진폭비, 변화 방향 일치율을 계산했다. 4개 값 배열이 전부 동일한 쌍은 근사 모양 통계에서 제외하고 별도로 집계했다. 앞 절반에서 후보를 선택하고 뒤 절반에서는 길이를 다시 고르지 않았다.

## 계획에서 변경된 점

- 핵심 계획과 수치 기준은 변경하지 않았다. 반구간 길이와 N 때문에 비교 쌍이 5개 미만인 결과는 산출해 보존하되 후보 선정에서 제외하고 확인 표에 적격 집단 수를 함께 표시했다.
- 반복 가족을 새로 구성하지 않고, 원본과 기존 controlled100 조건을 재검산해 사용했다. 이 통제는 진짜 독립 표본을 확보했다는 뜻이 아니다.
- 고정 위상 템플릿의 뒤 절반 시작점은 절반 경계에서 위상을 재시작하지 않고, 원래 offset과 같은 나머지 위치가 되도록 조정했다. 국소 인접 블록 비교는 계획대로 각 반구간에서 시작한다.
- 시간 늘이기·DTW·신경망·외부 데이터는 사용하지 않았다. 가변 길이 작업 경계를 완전히 분할한 실험이 아니다.

## 핵심 결과

### 1. 비교 규칙과 분모

| 항목 | 사전 고정 기준 |
|---|---|
| 후보 탐색 길이 | N=2..100 |
| 후보 선정 가능 길이 | N>=8. 짧은 벡터의 우연 상관을 피하기 위한 내부 기준 |
| 탐색 집단 적격 | 정확 일치가 아닌 쌍 >=5, 양쪽 변동이 있는 쌍 >=5 |
| 모양 일치 기준 | 상관>=0.7, 대칭 모양 오차<=0.75, 진폭비 0.5..2 모두 만족 |
| 상수 블록 | 상관 미산출, 모양 일치 판정 실패. 결측을 0 상관으로 바꾸지 않음 |
| 원시값 비교 | 단위를 보존한 RMSE |
| 모양 비교 | 각 블록 평균만 제거. 블록별 표준편차로 나누지 않음 |
| 후보 선정 | 앞 절반에서 변수별 일치율 상위 3개. 동률은 상관, 작은 N 순 |

대칭 모양 오차는 `sqrt(sum((Ac-Bc)^2)/(sum(Ac^2)+sum(Bc^2)))`다. 양쪽 모양이 같으면 0이고, 비슷한 에너지의 비상관 모양이면 약 1이다. 품질 점수가 아니며 상관·진폭비와 함께 사용했다. 표의 일치율은 불량 예측 정확도가 아니다.

all=원본 전체, controlled100=100건 이상 동일 블록의 후행 출현 제외 조건. discovery=앞 절반, confirmation=뒤 절반. 전체 자료는 이전 EDA에서 이미 보았으므로 뒤 절반도 엄밀한 독립 미관측 시험셋이 아니다.
""")
    stats = (
        segments.groupby("condition")
        .agg(records=("n", "sum"), segments=("segment", "size"), dates=("date", "nunique"))
        .reset_index()
    )
    add(
        table(
            stats.rename(
                columns={
                    "condition": "조건",
                    "records": "기록 수",
                    "segments": "연속 구간",
                    "dates": "날짜 수",
                }
            )
        )
    )

    add("### 2. 앞 절반에서 선택한 후보 — 높은 순위가 곧 강한 반복은 아님")
    cols = [
        "variable",
        "length",
        "rank",
        "mean_pass_rate",
        "median_corr",
        "median_raw_rmse",
        "median_centered_rmse",
        "segments",
        "compared_pairs",
    ]
    selection = candidates[cols].copy()
    selection["mean_pass_rate"] *= 100
    add(
        table(
            selection.rename(
                columns={
                    "variable": "변수",
                    "length": "N",
                    "rank": "순위",
                    "mean_pass_rate": "집단 평균 일치 %",
                    "median_corr": "집단 중앙 상관",
                    "median_raw_rmse": "원시 RMSE",
                    "median_centered_rmse": "평균 제거 RMSE",
                    "segments": "구간 수",
                    "compared_pairs": "비교 쌍 합",
                }
            )
        )
    )
    add(
        """후보 선정 일치율은 적격한 구간×offset 집단의 비가중 평균이다. 아래 확인 표의 합산 비율과 가중 방식이 다르다. 원시 RMSE 단위는 force=bar, current=kA, voltage=V, time=ms다.

전압에서는 16·15·17건이 상위다. 다른 변수는 상위 길이라도 대부분 집단 평균 일치율이 낮고 중앙 상관도 약하다. 반드시 상위 3개를 뽑는 절차이므로, 순위만으로 모든 변수에 작업 주기가 있다고 해석하지 않는다."""
    )

    add("### 3. 후보의 뒤 절반 확인")
    matched = scan.merge(candidates[["variable", "length"]], on=["variable", "length"])
    rows = []
    for keys, group in matched[matched.phase == "confirmation"].groupby(
        ["condition", "variable", "length"]
    ):
        n = group.nonexact_pairs.sum()
        rows.append(
            [
                *keys,
                int(n),
                int(group.passed_nonexact_pairs.sum()),
                100 * group.passed_nonexact_pairs.sum() / n,
                int(((group.nonexact_pairs >= 5) & (group.valid_nonexact_pairs >= 5)).sum()),
                len(group),
                group["corr"].median(),
                group.raw_rmse.median(),
                group.centered_rmse.median(),
            ]
        )
    add(
        table(
            pd.DataFrame(
                rows,
                columns=[
                    "조건",
                    "변수",
                    "N",
                    "근사 비교 쌍",
                    "일치 쌍",
                    "합산 일치 %",
                    "적격 집단",
                    "전체 집단",
                    "집단 중앙 상관",
                    "원시 RMSE",
                    "평균 제거 RMSE",
                ],
            )
        )
    )
    add(
        """합산 일치율은 통과 쌍 합/정확 일치 제외 쌍 합이다. 짧은 확인 구간도 표에는 포함하지만, 적격 집단이 적거나 없는 경우 안정적인 재현으로 보지 않는다. 예를 들어 controlled100의 큰 N은 두 개의 긴 구간에 크게 의존한다."""
    )

    add("### 4. 전압: 16건과 이웃 길이 비교")
    rows = []
    voltage = scan[(scan.variable == "voltage") & scan.length.between(12, 20)]
    for keys, group in voltage.groupby(["condition", "phase", "length"]):
        n = group.nonexact_pairs.sum()
        rows.append(
            [
                *keys,
                int(n),
                int(group.passed_nonexact_pairs.sum()),
                100 * group.passed_nonexact_pairs.sum() / n,
                group["corr"].median(),
                group.shape_error.median(),
                group.direction_agreement.median(),
            ]
        )
    add(
        table(
            pd.DataFrame(
                rows,
                columns=[
                    "조건",
                    "반구간",
                    "N",
                    "비교 쌍",
                    "일치 쌍",
                    "일치 %",
                    "집단 중앙 상관",
                    "모양 오차",
                    "방향 일치율",
                ],
            )
        )
    )
    add(
        """16건 부근의 국소적인 모양 유사성이 15·17건 등과 비교해 어떻게 달라지는지 보여준다. 넓은 15~17건 범위의 유사성은 일정한 제품당 spot 수와 동일한 주장이 아니다. 변화 방향 일치율에는 양쪽 모두 변화가 없는 위치도 포함된다."""
    )

    add("### 5. 전압 16건: 시작점과 완전 동일 여부")
    rows = []
    for keys, g in scan[(scan.variable == "voltage") & (scan.length == 16)].groupby(
        ["condition", "phase", "offset"]
    ):
        n = g.nonexact_pairs.sum()
        rows.append(
            [
                *keys,
                int(g.pairs.sum()),
                int(g.exact_all_pairs.sum()),
                int(g.identical_variable_pairs.sum()),
                int(n),
                int(g.passed_nonexact_pairs.sum()),
                100 * g.passed_nonexact_pairs.sum() / n,
            ]
        )
    add(
        table(
            pd.DataFrame(
                rows,
                columns=[
                    "조건",
                    "반구간",
                    "시작점",
                    "전체 쌍",
                    "4변수 완전 동일",
                    "전압만 완전 동일",
                    "근사 비교 쌍",
                    "전압 모양 일치",
                    "일치 %",
                ],
            )
        )
    )
    add(
        """전압 16건 비교에서는 4변수 전체가 같은 쌍도, 전압 배열 자체가 완전히 같은 쌍도 없었다. 따라서 여기서의 모양 일치는 완전 동일 배열을 다시 센 결과가 아니다. 다만 서로 다른 날짜의 긴 배열 재출현 때문에 개별 비교 쌍들 사이의 독립성이 확보된 것은 아니다.

0과 8 시작점에서 비슷한 일치율이 나타나면, 특정 한 시작점이 제품 경계라고 선택할 근거는 약하다. 이번에는 계획대로 두 시작점을 검사했으며 가능한 16개 위상 전부를 검사한 것은 아니다."""
    )

    add("### 6. 같은 16건 묶음에서 다른 변수도 함께 반복되는가?")
    unique_pairs = details[
        (details.variable == "voltage") & (details.length == 16) & (~details.exact_all)
    ]
    rows = []
    for keys, g in unique_pairs.groupby(["condition", "phase"]):
        rows.append(
            [
                *keys,
                len(g),
                int((g.variables_passed >= 1).sum()),
                int((g.variables_passed >= 2).sum()),
                int((g.variables_passed >= 3).sum()),
                int((g.variables_passed == 4).sum()),
                100 * (g.variables_passed >= 2).mean(),
            ]
        )
    add(
        table(
            pd.DataFrame(
                rows,
                columns=[
                    "조건",
                    "반구간",
                    "비교 쌍",
                    "1변수 이상",
                    "2변수 이상",
                    "3변수 이상",
                    "4변수 모두",
                    "2변수 이상 %",
                ],
            )
        )
    )
    add(
        """네 변수 모두 같은 모양 기준을 만족하는 16건 비교는 없었다. 전압의 국소 반복을 작업 전체의 반복이나 제품 한 개의 생산 단위로 확장하기 어렵다. 단, 실제로 같은 작업을 하더라도 모든 변수가 동일한 형태로 움직여야 한다는 물리 법칙을 가정한 것은 아니다. 여기서는 증거의 범위를 제한한다."""
    )

    add("### 7. 고정 위상 템플릿은 뒤 절반에 유지되는가?")
    selected_templates = templates.merge(
        candidates[["variable", "length"]], on=["variable", "length"]
    )
    rows = []
    for keys, g in selected_templates.groupby(["condition", "variable", "length"]):
        rows.append(
            [
                *keys,
                len(g),
                int(g.confirmation_blocks.sum()),
                g.template_corr.median(),
                g.template_raw_rmse.median(),
                g.template_centered_rmse.median(),
                100 * g.template_pass_rate.mean(),
            ]
        )
    add(
        table(
            pd.DataFrame(
                rows,
                columns=[
                    "조건",
                    "변수",
                    "N",
                    "집단 수",
                    "확인 블록 합",
                    "중앙 템플릿 상관",
                    "원시 RMSE",
                    "평균 제거 RMSE",
                    "집단 평균 일치 %",
                ],
            )
        )
    )
    add(
        """국소 인접 쌍끼리 비슷한 것과 앞 절반 평균 템플릿이 뒤 절반의 같은 위상에 유지되는 것은 다르다. 전압의 N=16도 고정 템플릿 상관이 전반적으로 약하다. 평균 템플릿은 위상이 흔들리는 패턴을 평균내면서 진폭이 작아질 수 있어 진폭 조건도 함께 불리해진다. 따라서 템플릿 실패를 반복 자체가 없다는 증거로 쓰지 않지만, 고정 제품 경계의 근거로도 쓰지 않는다."""
    )
    vtemp = templates[(templates.variable == "voltage") & (templates.length == 16)]
    add(
        table(
            vtemp[
                [
                    "condition",
                    "segment",
                    "offset",
                    "training_blocks",
                    "confirmation_blocks",
                    "template_corr",
                    "boundary_mean_change",
                    "internal_mean_change",
                ]
            ].rename(
                columns={
                    "condition": "조건",
                    "segment": "구간",
                    "offset": "시작점",
                    "training_blocks": "앞 블록",
                    "confirmation_blocks": "뒤 블록",
                    "template_corr": "템플릿 상관",
                    "boundary_mean_change": "경계 평균 |ΔV|",
                    "internal_mean_change": "내부 평균 |ΔV|",
                }
            )
        )
    )
    add(
        """경계의 전압 변화가 항상 내부보다 크지 않고, 날짜·시작점에 따라 달라진다. 이것으로 공통 reset이나 제품 교체 경계를 특정하지 못했다. 실제 제품 교체가 센서 요약값에 큰 변화로 남지 않을 가능성도 있으므로 경계가 없다는 뜻은 아니다. 일부 짧은 구간의 높은 템플릿 상관은 확인 블록이 3개뿐인 소표본이라는 점을 함께 봐야 한다."""
    )

    add("### 8. 실제로 비슷한 전압 두 묶음 — 숫자는 다르지만 모양이 닮은 예")
    pool = details[
        (details.variable == "voltage")
        & (details.length == 16)
        & (details.phase == "confirmation")
        & (details.condition == "controlled100")
        & details.passed
        & (~details.exact_variable)
    ]
    example = pool.iloc[(pool["corr"] - pool["corr"].median()).abs().argmin()]
    a = (
        raw["weld Voltage(v)"]
        .iloc[int(example.a_excel) - 2 : int(example.a_excel) - 2 + 16]
        .to_numpy()
    )
    b = (
        raw["weld Voltage(v)"]
        .iloc[int(example.b_excel) - 2 : int(example.b_excel) - 2 + 16]
        .to_numpy()
    )
    add(
        f"controlled100 뒤 절반의 전압 16건 **통과 사례 중 상관 중앙값에 가장 가까운 쌍**을 선택했다. 최고의 사례만 고른 것이 아니다. {example.segment}, Excel {int(example.a_excel)}~{int(example.a_excel)+15} ↔ {int(example.b_excel)}~{int(example.b_excel)+15}."
    )
    add(
        table(
            pd.DataFrame(
                {
                    "위치": np.arange(1, 17),
                    "A 전압(V)": a,
                    "B 전압(V)": b,
                    "A 평균 제거": a - a.mean(),
                    "B 평균 제거": b - b.mean(),
                }
            )
        )
    )
    add(
        table(
            pd.DataFrame(
                [
                    [
                        example["corr"],
                        example.raw_rmse,
                        example.centered_rmse,
                        example.shape_error,
                        example.amplitude,
                        example.direction_agreement,
                    ]
                ],
                columns=[
                    "상관",
                    "원시 RMSE(V)",
                    "평균 제거 RMSE(V)",
                    "대칭 모양 오차",
                    "진폭비 B/A",
                    "방향 일치율",
                ],
            )
        )
    )
    add(
        """이 사례는 실제 생산품 2개의 대응으로 확인된 것이 아니라, 기록을 16건씩 잘랐을 때 모양 기준을 만족한 사례다. 앞뒤 품목·spot ID가 없으므로 제품 경계를 부여하지 않는다."""
    )
    all_examples = details[
        (details.variable == "voltage")
        & (details.length == 16)
        & (details.phase == "confirmation")
        & (details.condition == "controlled100")
        & details["corr"].notna()
    ]
    picks = []
    for label, q in [("낮은 상관", 0.1), ("전체 중간", 0.5), ("높은 상관", 0.9)]:
        target = all_examples["corr"].quantile(q)
        row = all_examples.iloc[(all_examples["corr"] - target).abs().argmin()]
        picks.append(
            [
                label,
                row.segment,
                int(row.a_excel),
                int(row.b_excel),
                row["corr"],
                row.centered_rmse,
                row.amplitude,
                bool(row.passed),
            ]
        )
    add(
        table(
            pd.DataFrame(
                picks,
                columns=[
                    "선택 기준",
                    "구간",
                    "A 시작행",
                    "B 시작행",
                    "상관",
                    "평균 제거 RMSE",
                    "진폭비",
                    "기준 통과",
                ],
            )
        )
    )
    add("낮은·중간·높은 사례를 함께 보면 모든 16건 묶음이 반복되는 것이 아님을 알 수 있다.")

    add("### 9. 길이 2~7건은 왜 후보에서 제외했는가?")
    short = scan[(scan.condition == "all") & (scan.phase == "discovery") & (scan.length < 8)]
    rows = []
    for (var, n), g in short.groupby(["variable", "length"]):
        rows.append(
            [
                var,
                n,
                int(g.nonexact_pairs.sum()),
                100 * g.passed_nonexact_pairs.sum() / g.nonexact_pairs.sum(),
                g["corr"].median(),
            ]
        )
    add(table(pd.DataFrame(rows, columns=["변수", "N", "비교 쌍", "일치 %", "집단 중앙 상관"])))
    add(
        """짧은 벡터는 적은 점만으로 높은 상관이 쉽게 생긴다. 특히 N=2의 비상수 상관은 ±1이 된다. 작은 길이에서의 높은 통과율만으로 공정 묶음을 정하지 않았고, 사전에 N>=8로 후보 선정을 제한했다. 8은 물리적 최소 spot 수가 아니라 이번 탐색의 내부 기준이다."""
    )

    add("""## Baseline 대비 변화

YSH-002의 lag16 후보를 인접한 16건짜리 **모양 비교**로 직접 확인했다. 전압에서는 실제 숫자가 다른 묶음 사이의 유사성이 남았다. 반면 고정 템플릿과 다변량 공동 패턴은 약해, 국소 반복과 제품 경계를 구분해야 한다는 근거가 강화됐다.

## 해석

같은 부품을 반복 생산하는 가설은 여전히 가능한 설명이다. 하지만 국소적 전압 반복은 주기·위상 변동, 측정·제어 신호 구조, 반복 작업의 일부 등 여러 설명과 양립한다. 데이터만으로 컨베이어 구조나 한 제품의 spot 수를 식별한 것은 아니다.

따라서 “비슷하면 반복 후보로 본다”는 접근은 유용했다. 다만 16건의 모든 블록이 같다는 뜻도, 높은 상관이 곧 같은 제품 단계라는 뜻도 아니다. 단일 상관 대신 실제 값·모양 오차·진폭비·방향·다변량 동시성·확인 구간을 함께 봐야 한다.

## 가설과 일치 여부

| 가설 | 판정 | 이유 |
|---|---|---|
| 완전 동일하지 않은 반복 모양이 존재 | 전압에서 부분 지지 | 16건의 비동일 전압 배열에서도 기준 통과 사례가 존재 |
| 특정 제품 길이 N으로 전체를 고정 분할 | 미확정 | 고정 템플릿 재현 약함, 경계 변화 불일관 |
| 네 변수 전체의 반복 작업 묶음 | 이번 기준에서 지지 약함 | 16건에서 4변수 공동 통과 없음 |
| 같은 품번의 반복 생산 | 가설 유지 | 신호와 양립하지만 개별 제품 ID·spot 위치가 없음 |

## 오류/실패

분석과 검산은 정상 종료했다. 상수 블록의 상관 미산출, 짧은 구간에서 큰 N의 비교 부족은 의도한 처리다. 산출되지 않은 조합은 근거가 없으며 실패율 0으로 채우지 않았다. 독립 날짜 검증을 확보하지 못한 점은 계속 남는 한계다.

## 재현 확인
""")
    add(
        table(
            pd.DataFrame(
                [
                    ["Python", summary["python"]],
                    ["pandas / numpy", summary["pandas"] + " / " + summary["numpy"]],
                    ["주 분석 시간(초)", summary["seconds"]],
                    ["독립 비교 쌍 검산", checks["independent_pair_checks"]],
                    ["요약·상세 분모 대조", checks["summary_detail_reconciliation"]],
                    ["앞뒤 경계·후보 순위", checks["candidate_ranking_verified"]],
                    ["입력 해시 보존", checks["input_unchanged"]],
                    ["입력 SHA-256", summary["raw_sha256"]],
                ],
                columns=["항목", "값"],
            )
        )
    )
    add(
        """주 분석 1회 후 별도 코드에서 960개 표본 비교의 원시/평균 제거 오차·상관·진폭·분모·경계를 확인하고, 선택된 길이의 모든 요약/상세 개수를 대조했다. 추가로 verify_templates.py에서 템플릿 972행의 상관·오차·절대 위상·경계 변화량을 원본으로 재계산해 검산했다(outputs/final_checks.json). 별도 환경에서의 재현이나 통계적 유의성 검정을 수행한 것은 아니다. 코드·설정·계획 실행 시점 해시는 outputs/summary.json에 보존한다.

```powershell
& 'F:/Kamp/.venv/Scripts/python.exe' 'F:/Kamp/experiments/YSH/ysh-003/analyze.py' --config 'F:/Kamp/experiments/YSH/ysh-003/config.json'
& 'F:/Kamp/.venv/Scripts/python.exe' 'F:/Kamp/experiments/YSH/ysh-003/verify.py'
& 'F:/Kamp/.venv/Scripts/python.exe' 'F:/Kamp/experiments/YSH/ysh-003/verify_templates.py'
& 'F:/Kamp/.venv/Scripts/python.exe' 'F:/Kamp/experiments/YSH/ysh-003/report.py'
```

재실행은 같은 outputs와 보고서를 갱신한다. 입력·설정 변경 후에는 표뿐 아니라 이번 실행에 맞춘 서술 해석도 다시 검토해야 한다.

## 결론

**전압의 16건 안팎 국소 유사 반복은 후보로 남긴다. 고정 제품 경계·제품당 spot 수 복원은 보류한다.** 시간 변형 없이도 일부 유사성이 나타났지만, 네 변수의 공동 반복과 고정 위상 지속이 충분하지 않다.

## 채택 / 보류 / 폐기

- 채택: 원시값과 평균 제거 모양을 함께 비교하고, 전압의 국소 유사성을 제한된 탐색 결과로 기록.
- 보류: 16건 단위 product_id 생성, 제품당 spot 수·컨베이어 cycle 확정, 생산량 환산.
- 비채택: 모든 블록을 정규화·시간 변형해 억지로 맞추거나, 일부 좋은 쌍만으로 전체 공정 반복을 주장하는 접근.

## 다음 실험 또는 다음 행동

고정 경계가 필요하다면 시작 위상 전수·국소 주기 길이 변동을 별도 질문으로 검토하고, 가능하면 실제 제품 ID·spot 위치·프로그램 단계와 대조해야 한다. 이것을 바로 불량 예측용 제품 ID로 사용하지 않는다. 현재 자료만으로 모델을 비교한다면 반복 누수와 평가 질문을 먼저 정하고 정적 4변수 기준선을 별도 계획한다.

## 상세 표 안내

| 파일 | 내용 |
|---|---|
| outputs/tables/motif_scan.csv | 전체 길이·offset·반구간·조건별 수치 |
| outputs/tables/candidate_ranking.csv | 앞 절반 후보 순위 전체 |
| outputs/tables/selected_candidates.csv | 고정한 변수별 상위 3개 |
| outputs/tables/selected_pair_details.csv | 선택 길이의 실제 행 대응과 수치 |
| outputs/tables/template_boundary_checks.csv | 고정 위상 템플릿과 경계 변화 |
| outputs/tables/segments.csv / partitions.csv | 원본 구간 및 앞뒤 분리 위치 |
"""
    )
    report = "\n\n".join(parts) + "\n"
    (ROOT / "실험결과.md").write_text(report, encoding="utf-8")
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ROOT.iterdir()
        if p.suffix in [".py", ".json"]
    }
    (OUT / "code_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Report: {len(report.splitlines())} lines")


if __name__ == "__main__":
    main()
