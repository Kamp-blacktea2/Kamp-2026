"""Render a detailed Korean Markdown report from audited numerical tables."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TABLES = OUT / "tables"


def read(name):
    return pd.read_csv(TABLES / name)


def table(frame, decimals=4):
    frame = frame.rename(
        columns={
            "condition": "조건",
            "segment": "구간",
            "date": "날짜",
            "n": "기록 수",
            "max_lag": "최대 lag",
            "pairs": "비교 쌍",
            "exact_count": "일치 쌍",
            "exact_rate": "정확 일치율",
            "longest_exact": "최장 연속 일치",
            "unordered_reference": "무순서 참고율",
            "partition": "구간 분할",
            "multiple": "배수",
            "value": "상관",
            "three_cycles": "3주기 이상",
            "metric": "지표",
            "candidate_lag": "후보 lag",
            "variable": "변수",
            "cycles": "완전 주기 수",
            "phase_mean_variance_fraction": "위상 평균 분산 비율",
            "threshold": "기준/문턱",
            "policy": "집단 제약",
            "components": "성분 수",
            "singletons": "단일행 성분",
            "largest_rows": "최대 성분 행",
            "largest_fraction": "최대 성분 비율",
            "excluded_rows": "제외 행",
            "covered_rows": "반복 포함 행",
            "repeat_pairs": "반복 쌍",
            "date_a": "날짜 A",
            "date_b": "날짜 B",
            "max_length": "최장 길이",
            "window": "창 길이",
            "eligible_positions": "계산 위치 수",
            "single_candidates": "1변수 이상 후보",
            "joint_candidates": "공동 후보",
            "retained": "대표점",
            "excel_row_after": "변화 직후 Excel 행",
            "delta_force": "ΔF(bar)",
            "delta_current": "ΔI(kA)",
            "delta_voltage": "ΔV(V)",
            "delta_time": "ΔT(ms)",
            "unique": "고유값 수",
            "min_positive_gap": "최소 양수 간격",
            "typical_gap": "대표 간격",
            "longest_run": "최장 유지(건)",
            "top_value": "최빈값",
            "top_count": "최빈값 행 수",
            "top_share": "최빈값 비율",
        }
    )

    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            if value == int(value):
                return str(int(value))
            if 0 < abs(value) < 1e-5:
                return f"{value:.3e}"
            return f"{value:.{decimals}f}"
        return str(value).replace("|", "/")

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "|" + "---|" * len(frame.columns)
    body = [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider] + body)


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    checks = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    lags = read("lag_scan.csv")
    segments = read("segments.csv")
    graph = read("group_summary.csv")
    variables = read("variable_summary.csv")
    changes = read("change_summary.csv")
    durations = read("state_durations.csv")
    peaks = read("peak_candidates.csv")
    repeat = read("repeat_alignment.csv")
    raw = pd.read_excel(config["raw"], sheet_name="Raw data")
    names = ["force", "current", "voltage", "time"]
    cols = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
    parts = []

    def add(text):
        parts.append(text.strip())

    add("""# 실험결과 — YSH-002 반복 구조의 근거 분리

**결론: 동일 블록의 영향과 수준 지속·전환이 함께 관찰되며, 전압에는 국소적인 16건 반복 후보도 남는다. 하나의 공통 설비 cycle로 확정할 수 없다. 가장 강한 검증 제약은 긴 동일 블록을 통해 9개 날짜가 모두 연결된다는 점이다.**

사용자 요청에 따라 그림·HTML 없이 수치와 표 중심으로 작성했다. 숫자는 이번 CPU 실행에서 계산했으며, 관측 사실과 해석·한계를 구분했다.

## Metadata

- 실험 ID: YSH-002, 담당자 YSH, 두 번째 실험.
- 유형: EDA / VALIDATION / ERROR_ANALYSIS.
- 상태: 완료. 실제 설비·데이터 제작 원인의 식별은 미확정.
- 실행일: 2026-09-23. 작업 브랜치 ysh-001-plan. commit/push 없음.
- 계획: [실험계획.md](실험계획.md).
- 장치: CPU. 모델·학습·외부 데이터 사용 없음.

## 실제 수행 내용

원본/후행 반복 제외 50·100·200건 기준에 대해 구간별 유효 lag를 전수 계산했다. 원시값과 1차 차분 상관, 4변수 정확 일치율, 연속 정확 일치 길이, 이웃 대비 피크와 배수·앞뒤 절반 재검토를 수행했다. 정확 대응의 행 집단과 입력창/날짜 단위 제약, 전후 중앙값의 공동 변화점, 상태 길이와 값의 양자화·유지길이를 계산했다.

표의 공통 정의:

- all=원본 전체. controlled50/100/200=해당 길이 이상 정확 반복의 후행 출현을 합집합으로 제외한 민감도 조건. 원본 파일은 삭제하지 않았다.
- F=가압력(bar), I=전류(kA), V=전압(V), T=통전시간(ms).
- 위치는 1부터 센 Excel 행 번호 또는 날짜 내부 위치이며, idx와 구분한다. lag는 기록 간격이다.
- 상관은 구간별 Pearson 상관이다. 집계 표는 명시한 경우를 제외하고 **구간별 상관의 비가중 중앙값**이다.
- —는 계산 불가/관측 없음이다. 0과 다르다. 표시는 소수 4자리이며 상세 CSV는 더 많은 정밀도를 보존한다.

## 계획에서 변경된 점

1. 최신 사용자 요청을 반영해 그림 최대 4장 계획을 상세 Markdown 표로 대체했다.
2. 기존 245개 반복 쌍은 원본 해시와 모든 값의 일치를 다시 검사한 뒤 재사용했다. 정확 반복의 전체 오프셋 탐색을 처음부터 재실행한 것은 아니다.
3. 50/100/200 기준의 행 수와 집단 구조에 차이가 있어 세 통제 조건 모두 lag 스캔을 수행했다. 변화점은 계획의 주 비교인 all/controlled100만 수행했다.
4. 길이 통제는 retained 구간과 같은 날짜·길이의 원본 창에서 처음/가운데/마지막 시작점을 결정적으로 선택했다. 원본과 같은 행인 창도 표시했으며, 임의로 독립 대조군이라 부르지 않았다.
5. 반복 가족 대표는 복잡한 겹침을 전체 동일 시퀀스로 합치지 않고, 가장 긴 10개 쌍에서 정확히 공유하는 구간의 앞선 출현만 요약했다. 서로 중첩되므로 10개 독립 가족이 아니다.
6. 입력창 집단은 모든 유효 20건 겹침창을 보존하고 그 행들을 함께 배치한다는 강한 조건을 검사했다. 경계창을 제거하는 purged split을 설계·실행한 것은 아니다.

## 핵심 결과

### 1. 입력과 관측 단위
""")
    add(
        table(
            pd.DataFrame(
                [
                    ["원본", "11,939행 × 10열, 결측 0"],
                    ["날짜 / 경계 분리 구간", "9일 / 12구간"],
                    ["기존 정확 반복 쌍 재검증", str(summary["verified_repeat_pairs"])],
                    [
                        "A/B 내부 구간 순서",
                        "모두 일치. B의 동일 내부 구간 계산은 중복 실행하지 않음",
                    ],
                    ["전체 자료 SHA-256", summary["input_sha256"]],
                    ["관측 단위", "파일상 생산순번. 제품/용접점/수집 주기는 미확정"],
                ],
                columns=["항목", "값"],
            )
        )
    )
    add(
        table(
            segments[segments.condition == "all"][
                ["segment", "date", "n", "excel_start", "excel_end", "max_lag"]
            ].rename(
                columns={
                    "segment": "구간",
                    "date": "날짜",
                    "n": "기록 수",
                    "excel_start": "Excel 시작",
                    "excel_end": "Excel 끝",
                    "max_lag": "최대 유효 lag",
                }
            )
        )
    )
    add(
        """날짜+idx 충돌 및 날짜 모순 때문에 1행짜리 구간 3개를 분리했다. 이 구간은 시계열 계산에서 제외되지만 원본 행은 보존한다. 최소 비교 쌍 100개가 필요하므로 유효 lag는 `1..min(1000,N-100)`이다. 주기 후보 검토에는 추가로 N>=3L을 요구했다."""
    )

    add("### 2. 반복 통제에 따른 표본 변화")
    rows = []
    for condition, meta in summary["conditions"].items():
        ss = segments[segments.condition == condition]
        rows.append(
            [
                condition,
                meta["rows"],
                meta["dates"],
                meta["segments"],
                int((ss.max_lag > 0).sum()),
                int(ss.max_lag.sum()),
            ]
        )
    add(
        table(
            pd.DataFrame(
                rows,
                columns=[
                    "조건",
                    "기록 수",
                    "날짜 수",
                    "구간 수",
                    "lag 계산 가능 구간",
                    "구간×lag 수",
                ],
            )
        )
    )
    add(table(segments[segments.condition == "controlled100"][["segment", "date", "n", "max_lag"]]))
    add(
        """100건 기준에서는 2,416행이 남지만 lag 계산 가능한 구간은 3개뿐이다. 이 중 127행 구간은 lag 27까지만 사용할 수 있다. 따라서 lag 27까지와 28 이후의 집계 중앙값은 구성 구간이 다르다. 2,416행을 독립 표본 수라고 해석하지 않는다.

YSH-001의 보조 자기상관은 길이 200 이상 구간을 대상으로 했고, 이번 계획은 비교 쌍 100개를 기준으로 하므로 짧은 127행 구간이 일부 lag에 새로 포함된다. 통제 후 중앙값이 이전 보고서와 달라질 수 있으며 계산 오류를 뜻하지 않는다."""
    )

    add("### 3. lag별 상관과 4변수 정확 반복")
    selection = [1, 2, 9, 16, 18, 20, 27, 32, 36, 48, 64, 100, 200, 399, 500, 1000]
    for condition in ["all", "controlled100"]:
        rows = []
        for lag, group in lags[(lags.condition == condition) & lags.lag.isin(selection)].groupby(
            "lag"
        ):
            rows.append(
                [
                    lag,
                    len(group),
                    int(group.pairs.sum()),
                    *[group[f"raw_{name}"].median() for name in names],
                    100 * group.exact_count.sum() / group.pairs.sum(),
                    int(group.longest_exact.max()),
                ]
            )
        add(
            f"**{condition}** — 상관은 구간별 중앙값, 정확 일치율은 일치 쌍 합/비교 쌍 합이다. 서로 다른 집계 방식을 구분한다."
        )
        add(
            table(
                pd.DataFrame(
                    rows,
                    columns=[
                        "lag",
                        "구간 수",
                        "비교 쌍 합",
                        "F 상관",
                        "I 상관",
                        "V 상관",
                        "T 상관",
                        "정확 일치 %",
                        "최장 연속 일치",
                    ],
                )
            )
        )
    add(
        """가압력·전류는 낮은 lag에서 높은 상관이 유지되고 점차 감소하는 경향이 있다. 이것은 수준 지속과 일관되지만 장비의 실제 동역학을 증명하지 않는다. lag 1의 높은 정확 일치율도 1건 생산 cycle이라는 뜻이 아니다. 인접 기록이 같은 값으로 저장되는 특성을 먼저 고려해야 한다.

20건에서 4변수 전체의 공통 정확 반복을 지지하는 결과는 약하다. 하지만 이를 근거로 모든 변수의 모든 주기를 부정해서는 안 된다."""
    )

    add("### 4. 정확 일치 피크는 어디서 나오는가?")
    top = lags[(lags.condition == "all") & (lags.lag > 1) & lags.three_cycles].nlargest(
        10, "exact_rate"
    )
    add(
        table(
            top[
                [
                    "segment",
                    "n",
                    "lag",
                    "pairs",
                    "exact_count",
                    "exact_rate",
                    "longest_exact",
                    "unordered_reference",
                ]
            ]
        )
    )
    add(
        """이 표의 exact_rate는 0~1 비율이다. unordered_reference는 같은 구간의 값 빈도로 계산한 `Σf(f-1)/(N(N-1))`이다. 독립성·유의성 검정이 아니라 값 빈도만으로 생기는 무순서 일치의 참고값이다.

3월 30일 all_s08의 lag 399에서 1,071쌍 중 399쌍이 일치하고, 그 399쌍이 연속된다. 이는 긴 동일 블록이 정확 일치 피크를 만들 수 있음을 보여준다. 일정 공정 cycle의 증거로 곧바로 해석할 수 없다. 해당 날짜가 controlled100에서 전부 제외되므로 “통제 후 피크 소멸”에는 날짜 자체의 소실도 포함된다."""
    )

    add("### 5. 변수별 국소 주기 후보: 전압 16건")
    voltage_rows = []
    for (condition, segment), group in lags[
        lags.condition.isin(["all", "controlled100"]) & lags.lag.isin([16, 32, 48])
    ].groupby(["condition", "segment"], sort=False):
        values = group.set_index("lag")
        voltage_rows.append(
            [
                condition,
                segment,
                int(group.n.iloc[0]),
                *[
                    values.loc[k, "raw_voltage"] if k in values.index else np.nan
                    for k in [16, 32, 48]
                ],
                values.loc[16, "diff_voltage"] if 16 in values.index else np.nan,
            ]
        )
    add(
        table(
            pd.DataFrame(
                voltage_rows,
                columns=["조건", "구간", "N", "V lag16", "V lag32", "V lag48", "차분 V lag16"],
            )
        )
    )
    rechecks = read("peak_rechecks.csv")
    selected = rechecks[
        (rechecks.segment == "controlled100_s01")
        & (rechecks.metric == "raw_voltage")
        & (rechecks.candidate_lag == 16)
    ]
    add("**3월 24일 구간의 앞·뒤 절반 검토**")
    add(table(selected[["partition", "n", "multiple", "pairs", "value", "three_cycles"]]))
    add(
        """3월 24일 전압에서는 16·32·48건과 앞/뒤 절반에 양의 상관이 남고 차분에서도 16건 상관이 남는다. 따라서 **전압의 국소 기록 주기 후보**로 남길 근거가 있다. 반면 3월 25일 계열 구간의 차분 lag16은 음수이고, 4월 7일의 짧은 보존 구간에서는 32·48건을 같은 표본 기준으로 계산할 수 없다.

그러므로 “전압에 국소 반복 후보가 있다”와 “모든 용접 기록이 16건 주기로 생성된다”는 다른 주장이다. 특히 날짜들이 긴 동일 블록으로 연결되어 있으므로 날짜별 재현을 독립적인 재현 횟수로 세지 않는다."""
    )

    add("### 6. 통전시간 피크의 배수 일관성")
    selected = rechecks[
        (rechecks.segment == "controlled100_s01")
        & rechecks.metric.isin(["raw_time", "diff_time"])
        & rechecks.candidate_lag.isin([18, 27, 64])
        & (rechecks.partition == "full")
    ]
    add(table(selected[["metric", "candidate_lag", "multiple", "pairs", "value"]]))
    add(
        """상위 피크 탐색은 지표별 ±5 lag 이웃 대비 국소 최대의 차이가 큰 5개를 선택한 사후 탐색이다. 18건·64건의 단일 상관 피크가 있어도 모든 배수에서 계속 지지되지는 않는다. 따라서 한 피크만 보고 cycle_position을 확정하지 않는다. 전체 후보와 배수·반구간 결과는 peak_candidates.csv와 peak_rechecks.csv에 보존했다.

후보별 완전 주기를 위상 위치로 접은 평균·표준편차와 위상 평균 분산 비율도 계산했다. 이 비율은 같은 자료에 맞춘 기술통계이며, 주기 길이가 커질수록 과적합될 수 있어 판정 점수나 p값으로 쓰지 않는다."""
    )
    phase = read("phase_summary.csv")
    add(
        table(
            phase[
                (phase.condition == "controlled100")
                & phase.lag.isin([16, 18, 27, 64])
                & phase.variable.isin(["voltage", "time"])
            ][["segment", "variable", "lag", "cycles", "phase_mean_variance_fraction"]]
        )
    )

    add(
        "3월 24일 전압을 16건으로 접은 위상 평균 분산 비율은 0.0184, 약 1.84%다. 가까운 주기끼리 상관이 있어도 구간 전체가 같은 위상에 고정됐다는 근거는 약하다. 고정 cycle_position 특징의 효과를 입증한 것이 아니다. 짧은 127행 구간의 더 큰 비율은 완전 주기가 7개뿐인 과적합 가능성과 함께 봐야 한다."
    )

    add("### 7. 같은 길이 대조: 길이 변화와 값 변화의 구분")
    matched = read("length_controls.csv")
    subset = matched[matched.lag.isin([1, 16, 20])]
    rows = []
    for keys, group in subset.groupby(["retained_segment", "n", "excel_start", "same_rows"]):
        by_lag = group.set_index("lag")
        rows.append(
            [
                *keys,
                by_lag.loc[1, "raw_force"],
                by_lag.loc[16, "raw_voltage"],
                by_lag.loc[20, "raw_current"],
            ]
        )
    add(
        table(
            pd.DataFrame(
                rows,
                columns=[
                    "대상 통제 구간",
                    "N",
                    "원본 창 시작",
                    "동일 행 여부",
                    "F lag1",
                    "V lag16",
                    "I lag20",
                ],
            )
        )
    )
    add(
        """시작점이 바뀌면 같은 날짜·길이에서도 상관이 달라질 수 있다. true 행은 retained와 동일한 구간이므로 독립 대조군이 아니다. 3월 24일 전체 1,200건에는 같은 날짜의 다른 1,200건 창이 없어 비자명한 길이 대조가 불가능하다. 이 비교는 위치·길이 효과를 드러내며, 동일 블록 제거의 인과 효과를 식별하지 않는다."""
    )

    add("### 8. 날짜 내부 위치까지 복원한 긴 동일 블록")
    block_columns = [
        "a_date",
        "a_start",
        "a_end",
        "a_start_date_position",
        "a_end_date_position",
        "b_date",
        "b_start",
        "b_end",
        "b_start_date_position",
        "b_end_date_position",
        "length",
    ]
    add(table(repeat.nlargest(10, "length")[block_columns]))
    add(
        """표에서 a_start/a_end와 b_start/b_end는 Excel 행 번호, *_date_position은 해당 날짜에 속하는 원본 행을 1부터 센 위치다. idx는 별도 상세 표에 있다. 모든 쌍은 공정변수 4개가 원본 정밀도에서 정확히 같았다. 부분적으로 겹치는 쌍이 있으므로 길이를 더해 독립 표본 수나 독립 가족 수를 만들지 않는다."""
    )

    add("### 9. 검증 집단을 어떻게 정의하느냐에 따른 차이")
    add(table(graph))
    add(
        """- row_correspondence: 길이 기준을 만족하는 반복 블록에서 대응하는 행끼리만 연결.
- window20: 위 제약에 더해 모든 겹치는 20건 입력창의 행을 함께 묶음. 20건 이상인 연속 구간 전체가 연결되는 강한 조건.
- whole_date: 정확 대응에 더해 같은 날짜의 모든 행을 함께 묶음.
- covered_rows는 반복 양쪽 출현의 합집합 크기다. excluded_rows는 후행 출현만의 합집합 크기다.

**100건 기준 행 대응만 묶으면 2,361성분이지만, 날짜 전체를 함께 묶으면 11,939행 전부가 하나의 성분이다.** 따라서 같은 날짜를 분리하지 않으면서 긴 동일 블록의 양쪽도 분리하지 않는 검증 집단은 둘 이상 만들 수 없다. 50/200건 기준에서도 동일한 결론이다.

window20의 최대 성분은 11,936행이고 나머지는 모순 경계의 1행짜리 구간 3개다. 이 결과는 **모든 겹침창을 보존하는 조건**의 결과다. 경계창을 제거하고 반복 대응도 purge하는 검증까지 불가능하다는 증명은 아니다. 그 방법의 표본 손실과 남는 집단은 후속 별도 설계가 필요하다.

행 대응 성분 수 역시 독립 표본 수가 아니다. 짧은 동일값 반복과 인접 의존성은 긴 블록 기준만으로 모두 통제되지 않는다."""
    )
    comp = read("component_sizes.csv")
    histogram = (
        comp[(comp.threshold == 100) & (comp.policy == "row_correspondence")]
        .groupby("rows")
        .size()
        .reset_index(name="성분 수")
        .rename(columns={"rows": "성분당 행 수"})
    )
    add(table(histogram))
    add("**날짜 연결의 근거 — 100건 기준, 각 날짜 쌍의 최장 동일 구간**")
    add(table(read("date_connections.csv").query("threshold == 100").drop(columns="threshold")))

    add("### 10. 수준 전환: 창 길이·문턱 민감도")
    aggregated = (
        changes.groupby(["condition", "window", "threshold"])[
            ["eligible_positions", "single_candidates", "joint_candidates", "retained"]
        ]
        .sum()
        .reset_index()
    )
    add(table(aggregated))
    add(
        """직전 w건과 직후 w건의 중앙값 차이를 해당 구간 IQR/1.349로 나눴다. F/I/V 중 2개 이상이 문턱을 넘으면 공동 후보로 표시하고, w건 이내로 이어지는 후보 집단에서 최대 점수 위치 하나를 택했다. 동률이면 가장 앞 위치다. 원시 후보는 서로 겹치는 위치이므로 독립 사건 수가 아니다.

all의 w=20에서 문턱을 3→4로 높여도 대표점 수가 8→12로 늘었다. 강한 문턱이 연결된 후보 덩어리를 나누면서 대표점이 늘어날 수 있기 때문이다. 따라서 대표점 수는 단조적인 이상 강도 척도가 아니며, 상태 길이에도 알고리즘 설정의 영향이 크다.

controlled100에서는 w=20/문턱3의 대표점이 2개이며, w=50/문턱3에서는 0개다. 이것은 상태 전환이 물리적으로 사라졌다는 뜻이 아니다. 표본·구간 IQR·창 크기가 함께 달라진다."""
    )
    rows = []
    for (condition, w, threshold), group in durations.groupby(["condition", "window", "threshold"]):
        complete = group.loc[~group.censored, "length"]
        rows.append(
            [
                condition,
                w,
                threshold,
                len(group),
                int(group.censored.sum()),
                len(complete),
                complete.min(),
                complete.median(),
                complete.max(),
            ]
        )
    add(
        table(
            pd.DataFrame(
                rows,
                columns=[
                    "조건",
                    "창",
                    "문턱",
                    "상태 전체",
                    "잘린 상태",
                    "완전 상태",
                    "최소 길이",
                    "중앙 길이",
                    "최대 길이",
                ],
            )
        )
    )
    add(
        """첫·마지막 상태는 연속 구간 경계 때문에 지속시간이 잘린 관측이다. 후보가 없으면 구간 전체를 한 개의 잘린 상태로 세었다. 주 설정에서 완전 상태는 원본 5개, 통제 1개뿐이므로 상태 지속시간의 일반적인 분포를 추정하기에는 부족하다. 약 200건 cycle을 뒷받침한다고 볼 수 없다."""
    )

    add("### 11. 주 설정의 실제 전환 위치와 4변수 변화량")
    cp = read("change_candidates.csv")
    selected = cp[(cp.window == 20) & (cp.threshold == 3) & cp.retained]
    add(
        table(
            selected[
                [
                    "condition",
                    "segment",
                    "excel_row_after",
                    "delta_force",
                    "delta_current",
                    "delta_voltage",
                    "delta_time",
                ]
            ]
        )
    )
    add(
        """변화량은 직후 중앙값−직전 중앙값이며 원 단위다. 통전시간 변화량 0은 원시 신호 전체가 불변이라는 뜻이 아니라 두 창의 중앙값이 같다는 뜻이다. 원본 여러 날짜에서 같은 변화량이 반복되는 현상 역시 동일 블록의 영향과 분리해 해석해야 한다. 레시피 변경이나 고장 원인으로 확정하지 않는다."""
    )

    add("### 12. 값의 양자화·유지길이: 설정값인가 측정값인가?")
    keep = [
        "condition",
        "variable",
        "unique",
        "min_positive_gap",
        "typical_gap",
        "longest_run",
        "top_value",
        "top_count",
        "top_share",
    ]
    add(table(variables[variables.condition.isin(["all", "controlled100"])][keep]))
    add(
        """min_positive_gap은 원본 고유값 간 최솟값이다. 가압력의 약 4.44e-16은 실질적인 측정 해상도로 해석할 수 없는 부동소수점 수준의 차이다. typical_gap은 1e-10보다 큰 간격을 소수 9자리로 표현한 최빈 간격이며, 계측기 사양을 뜻하지 않는다. 정확 반복 검사는 반올림하지 않았다.

최장 동일값 유지길이는 F 12건, I 8건, V 8건, T 9건이다. 통전시간이 소수의 값에 집중돼 있지만 긴 설정값 plateau라고 단정할 근거는 부족하다."""
    )
    counts = raw["weld time(ms)"].value_counts().sort_index().reset_index()
    counts.columns = ["통전시간(ms)", "기록 수"]
    counts["비율 %"] = counts["기록 수"] / len(raw) * 100
    add(table(counts))
    add(
        """통전시간의 고유값은 8개다. 비정수는 72.16·72.18·72.20·72.24이며 총 297건(약 2.49%)이다. controlled100에는 비정수 27건이 남는다. 따라서 70/71/72/73의 네 범주라는 가정은 맞지 않는다. 설정값·센서 양자화·후처리 중 무엇인지는 데이터 설명을 추가 확인해야 한다."""
    )
    shared = []
    dates = pd.to_datetime(raw["working time"]).dt.strftime("%Y-%m-%d")
    for column, name in zip(cols, names):
        by = pd.DataFrame({"value": raw[column], "date": dates}).groupby("value").date.nunique()
        many = set(by[by >= 2].index)
        shared.append(
            [
                name,
                int((by >= 2).sum()),
                int(raw[column].isin(many).sum()),
                int(raw[column].nunique()),
                int(raw[column].round(9).nunique()),
            ]
        )
    add(
        table(
            pd.DataFrame(
                shared,
                columns=[
                    "변수",
                    "2일 이상에 출현한 값 수",
                    "그 값에 속한 행 수",
                    "원본 고유값 수",
                    "소수9자리 표현 고유값 수",
                ],
            )
        )
    )
    add(
        "반복 날짜의 값 공유도 관찰 사실이다. 실제 독립 생산일에서 재현된 설정·측정값이라는 뜻은 아니다."
    )

    add("""## Baseline 대비 변화

YSH-001에서는 20건 공통 반복이 지지되지 않았고 최대 1,639건 동일 궤적을 확인했다. 이번에는 유효 lag 전수 비교를 통해 399건 등 정확 반복 피크가 긴 블록에 대응함을 수치화했고, 전압의 국소 16건 후보를 별도로 발견했다. 더 중요한 추가 결과는 날짜 전체 보존과 정확 대응 보존을 동시에 요구하면 모든 날짜가 하나의 검증 성분이 된다는 점이다.

## 해석

세 가지가 공존한다. 가압력·전류의 수준 지속, 설정에 민감한 다변량 전환, 날짜 간 긴 정확 반복이 있다. 전압에는 일부 구간에서 배수·차분·반구간에 남는 국소 반복 후보가 있으므로 “주기가 전혀 없다”는 결론도 과하다. 반대로 이를 공통 설비 cycle로 확정해 시간축이나 제품 생산 단위를 재구성하는 것도 근거가 부족하다.

반복 제거의 결과만으로 복제 원인을 증명하지 못한다. 원본의 생성·수집·정리 과정이 확인되어야 실제 주기와 재사용을 물리적으로 구분할 수 있다. EDA 전체를 본 자료이므로 피크 선택과 검증 결과는 탐색 근거다.

## 가설과 일치 여부

| 가설 | 판정 | 근거와 한계 |
|---|---|---|
| 하나의 공통 설비 cycle | 미확정 | 변수·구간별 차이, 독립 재현 집단 부족 |
| 국소 기록 주기 후보 | 부분 지지 | 전압 16/32/48건이 일부 구간에서 지속, 다른 구간의 차분에서는 불일치 |
| 상태 지속·전환 | 부분 지지 | F/I 낮은 lag 상관, 공동 변화량. 상태 길이는 설정 민감·소표본 |
| 동일 블록 영향 | 강한 관찰 근거 | 최대 1,639건 완전 동일, 날짜 전체 연결 |
| 인위적 복제라는 제작 원인 | 판단 불가 | 생성 코드·공식 계보가 없음 |

## 오류/실패

검토 중 차분 피크 표의 비교 쌍 수가 원시값 기준으로 1개 크게 표시된 문제를 수정하고 재실행했다. 차분 상관 자체는 올바르게 계산됐으며 최종 표에는 차분의 쌍 수와 적격 여부를 사용한다. 핵심 분석과 독립 검산은 정상 종료했다. 원본 데이터·기존 ysh-001·scaled 파일은 수정하지 않았다. 상수 벡터의 상관이나 짧은 구간의 큰 lag는 의도적으로 계산하지 않았고, 빈 상태 길이는 0으로 채우지 않았다. 그림·HTML은 사용자 요청에 따라 생성하지 않았다.

## 재현 확인
""")
    add(
        table(
            pd.DataFrame(
                [
                    ["Python", summary["python"]],
                    ["pandas / numpy", f'{summary["pandas"]} / {summary["numpy"]}'],
                    ["주 분석 실행 시간(초)", round(summary["seconds"], 2)],
                    ["상관·일치율·연속 길이 독립 표본 검산", checks["lag_spot_checks"]],
                    [
                        "모든 설정의 대표 변화점 원 단위 중앙값 재검산",
                        checks["retained_change_checks"],
                    ],
                    ["원본과 정확 반복 쌍 재확인", checks["exact_pairs_verified"]],
                    ["날짜 그래프 독립 BFS", "50/100/200 기준 모두 9일 한 성분"],
                    ["전체 유효 lag 누락 여부", "없음"],
                    ["입력 해시 전후 비교", "동일"],
                    ["학습 / 난수 / seed", "없음"],
                ],
                columns=["검증 항목", "결과"],
            )
        )
    )
    add(
        """표시 오류를 수정한 최종 코드로 재실행한 뒤 독립 계산으로 표본 상관과 모든 대표 변화점·정확 쌍·날짜 그래프를 검산했다. 실행 코드·설정 해시는 outputs/summary.json에 있고, 보고서 생성 코드까지 포함한 현재 파일 해시는 outputs/code_manifest.json에 기록한다.

```powershell
& 'F:/Kamp/.venv/Scripts/python.exe' 'F:/Kamp/experiments/YSH/ysh-002/analyze.py' --config 'F:/Kamp/experiments/YSH/ysh-002/config.json'
& 'F:/Kamp/.venv/Scripts/python.exe' 'F:/Kamp/experiments/YSH/ysh-002/verify.py'
& 'F:/Kamp/.venv/Scripts/python.exe' 'F:/Kamp/experiments/YSH/ysh-002/report.py'
```

재실행하면 같은 실험 outputs와 보고서를 갱신한다. 입력이나 설정을 바꾸면 보고서의 서술형 해석을 다시 검토해야 한다. 표는 파일에서 계산하지만 해석 문장은 이번 실행을 검토한 기록이다.

## 결론

**동일 블록 영향 + 수준 지속·전환 + 일부 변수의 국소 반복 후보가 섞인 결과다.** 다음 단계는 모든 날짜를 독립 관측으로 보는 모델 점수 경쟁이 아니다. 정확 대응과 경계창을 어떻게 처리할지 결정한 후, 남은 자료로 어떤 검증 질문에 답할 수 있는지 먼저 명시해야 한다.

## 채택 / 보류 / 폐기

- 채택: 반복 대응 제약을 보존한 분할 설계, 전압 국소 주기 후보의 추가 검토, 과거 정보만 사용하는 상태 특징의 별도 계획.
- 보류: 공통 cycle_position, 날짜 단독 holdout의 독립성 주장, 자동 반복행 제거를 최종 정제로 채택, 물리적 정상·불량·안정영역 판정.
- 비채택: lag 피크 하나로 실제 설비 주기를 확정하거나 2,416행/2,361성분을 독립 표본 수로 간주하는 해석.

## 다음 실험 또는 다음 행동

1. 정적 4변수 모델과 순서창 모델의 평가 질문을 구분한다. 정적 모델도 긴 블록 집단만으로 짧은 동일값 누수를 모두 통제한 것은 아니다.
2. 날짜 보존을 완화하거나 경계창·반복 대응을 제거하는 분할을 설계할 경우, 남는 행·날짜·분포와 검증 의미를 먼저 감사한다. 이를 새 날짜 일반화와 동일시하지 않는다.
3. 충분한 검증 범위가 정해지면 원본 4변수 Isolation Forest를 별도 실험한다. 지금 바로 YSH-003 학습을 실행하지 않는다. 수집 계보·새 기간 자료가 필요하다는 결과도 허용한다.

## 상세 수치 파일

이 보고서만 먼저 읽으면 된다. 세부 확인이 필요할 때 outputs/tables의 파일을 사용한다.

| 파일 | 용도 |
|---|---|
| lag_scan.csv / segments.csv | 모든 구간·lag의 상관·정확 일치·분모 |
| peak_candidates.csv / peak_rechecks.csv | 피크 순위와 배수·앞뒤 절반 |
| phase_summary.csv / phase_profiles.csv | 후보 lag로 접은 위상 통계 |
| repeat_alignment.csv / date_connections.csv | 정확 반복의 원본·날짜 위치와 날짜 연결 |
| group_summary.csv / component_sizes.csv | 분리 제약별 집단 수·크기 |
| length_controls.csv / shared_interval_summary.csv | 길이 대조와 명시적 공유 구간 요약 |
| change_summary.csv / change_candidates.csv / state_durations.csv | 변화점·상태 길이 전체 설정 |
| variable_summary.csv | 값 간격·유지·고유값 통계 |
"""
    )

    report = "\n\n".join(parts) + "\n"
    (ROOT / "실험결과.md").write_text(report, encoding="utf-8")
    manifest = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in ROOT.iterdir()
        if path.suffix in [".py", ".json"]
    }
    (OUT / "code_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Detailed report written: {len(report.splitlines())} lines, {len(report)} characters")


if __name__ == "__main__":
    main()
