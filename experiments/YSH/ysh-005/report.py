"""Generate a self-contained, numerical Markdown report from verified outputs."""

from common import *


def table(frame, digits=5):
    frame = frame.copy()

    def fmt(value):
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value).replace("|", "/").replace("\n", " ")

    header = "| " + " | ".join(map(str, frame.columns)) + " |\n"
    header += "| " + " | ".join(["---"] * len(frame.columns)) + " |\n"
    return (
        header
        + "\n".join(
            "| " + " | ".join(fmt(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)
        )
        + "\n"
    )


def main():
    verification = json.loads((HERE / "outputs" / "verification.json").read_text(encoding="utf-8"))
    assert verification["passed"]
    raw, segments = load()
    metric = read("phase_metrics")
    metric = metric[metric.origin == "idx"]
    daily = read("daily_quality_comparison")
    parts = []

    def add(text):
        parts.append(text.strip() + "\n")

    def tab(frame, digits=5):
        parts.append(table(frame, digits))

    add("""# 실험결과 — YSH-005 네 지점의 행 대응과 작업 묶음·집계 품질

## Metadata

- 실험 ID: YSH-005, 다섯 번째 실험. 담당자 YSH.
- 상태: CPU 분석·수치 검산·보고서 작성 완료. 물리적 제품 경계는 미확정.
- 실행일: 2026-09-23.
- 계획: [실험계획.md](실험계획.md). 결과 표: [outputs/tables](outputs/tables).
- 유형: EDA / VALIDATION / FEATURE / ML / ERROR_ANALYSIS.
- 입력: 현재 Raw 11,939행, 네 변수 결측 0. 원본 및 수정 데이터 사전 해시 일치.
- 단위: force=bar, current=kA, voltage=V, time=ms. 행 간 시간 간격은 미상.

## 결론

**이번 데이터에서 연속 네 행을 네 물리적 용접점이나 제품 한 개로 확정할 근거는 확보하지 못했다.**

1. 네 위치를 추가한 예측은 44개 구간·변수 조건 모두 사전 기준인 MAE 5% 개선에 미달했다. 이 44개는 all 9구간×4변수와 controlled100 2구간×4변수이며 독립 반복 횟수가 아니다.
2. 시작점에 따라 2,974~2,981개의 완전 네 행 후보를 만들 수 있지만, 시작점을 특정하는 일관된 전이 증거가 없다. 실제 제품 수를 뜻하지 않는다.
3. 전압 lag16은 일부 구간에서 차분 후에도 남는다. 그러나 lag4가 일관되게 강한 것이 아니며, 네 위치의 고정 패턴 증거와 별개다.
4. 가압력>3bar는 해당 네 날짜마다 291행으로, 123행과 168행의 두 run이다. ‘300행=75제품’은 지지되지 않는다.
5. GMM 군집은 스케일에 민감하고 확인 기간의 모든 네 변수 조합이 학습에 존재한다. 군집을 제품 종류로, AE/IF flag를 실제 불량으로 해석할 수 없다.

## 실제 수행 내용

### 공식 설명, 가설, 확인 결과를 구분

| 항목 | 근거 수준 | 이번 결과 |
|---|---|---|
| 약8초·4지점·각72ms 수집 | 데이터 사전에 인용된 가이드 설명 | 보존. 원 가이드 PDF는 프로젝트에서 찾지 못했음 |
| 한 행=한 지점, 동일 순서 반복 | 이번 검증 가설 | 통계적 지지 부족. 부재를 증명한 것은 아님 |
| 연속4행=한 작업/제품 | 추가 경계·관측단위 가설 | 시작점과 제품 대응 미확정 |
| lag4/8/12/16 | 위 가정 아래 같은 지점의 1/2/3/4묶음 간격 | 함께 비교. lag16은 보조 재검증 |
| 네 변수는 가압·통전·유지 단계 | 근거 없음 | 단계 라벨 생성 안 함. waveform·이벤트 로그 필요 |
| Result 개수 | 날짜/유형별 집계 | 날짜로만 연결. 행 라벨·F1 생성 안 함 |

### 계산 범위와 검증 설계

- 원본 날짜/idx 연속성을 지키는 12구간. 반복 통제본 2,416행·9구간·4일. 삭제 뒤 행을 다시 붙이지 않았다.
- 구간 길이≥200인 곳은 앞60%로 Ridge/scaler 학습, 뒤40%로 확인. 직전 관측값을 사용하는 1단계 예측이며, 여러 미래 값을 재귀적으로 예측하는 모형은 아니다.
- M0=학습 중앙값, M1=직전값의 선형 모델, M2=M1+위상 one-hot. Ridge alpha=1, 전처리는 학습 구간에서만 fit.
- p=2/3/4/5/8/16, 원신호·차분 lag1..64, 학습 네 위치 중앙값 제거 잔차, 64행 블록별 phase 회전 100회(seed42).
- 네 행 후보 r=0/1/2/3을 모두 유지. r은 `(첫 idx-1) mod4`이며 실제 Spot 번호가 아니다.
- 묶음 S8=변수별 평균·범위, P16=4행×4변수. GMM K1..4(full, reg1e-3, n_init5), Standard/Robust, seed42/7/2026.
- 묶음/행 IF 300 trees, 최대256 학습 표본, 학습 점수95% 문턱. 묶음99% 문턱도 저장.
- AE 4→3→2→3→4, 각 hidden RReLU, linear output, Adam .01, batch64, 50epochs, seed42/7/2026. 학습 MinMax, clipping 없음, 평가 mode의 행 MSE 평균+8표준편차 문턱.
- 모델 학습=3월24~31일 8,470행, 확인=4월2·3·7일 3,469행. 이 기간은 이미 EDA한 자료이며 새로운 숨김 테스트셋이 아니다.
- 구조·모델 코드에서는 Result 값을 읽지 않았다. 공정 지표를 저장한 뒤 compare_result.py에서 집계 품질을 연결했다.

## 핵심 결과

### 1. 네 위치를 추가하면 예측이 좋아지는가?

아래 MAE는 구간별 확인 행 수로 가중한 평균이다. 단위는 변수별 원시 단위다. 서로 다른 변수의 MAE 크기를 직접 비교하지 않는다. M1 대비 개선율은 양수일 때 개선이다.
""")
    weighted = []
    for (condition, name), part in metric.groupby(["condition", "variable"]):
        p4 = part[part.p == 4]
        m1 = np.average(p4.m1_mae, weights=p4.test_n)
        row = dict(condition=condition, variable=name, M1=m1)
        for p in CONFIG["periods"]:
            subset = part[part.p == p]
            row[f"p{p}"] = np.average(subset.m2_mae, weights=subset.test_n)
        row["p4_개선_%"] = (m1 - row["p4"]) / m1 * 100
        row["p4_vs_p2_%"] = (row["p2"] - row["p4"]) / row["p2"] * 100
        weighted.append(row)
    tab(pd.DataFrame(weighted), 7)
    add(
        "구간별 p4의 M1 대비 개선율(%). 5% 이상인 셀은 없다. controlled100_s01은 all_s01과 같은 자료이며 독립 재현이 아니다."
    )
    improvement = (
        metric[metric.p == 4].pivot(
            index=["condition", "segment"], columns="variable", values="improvement"
        )
        * 100
    )
    tab(improvement.reset_index(), 3)
    add(
        """따라서 H1은 이번 사전 기준에서 **지원 부족**이다. 네 지점의 값 분포가 비슷하거나 실제 행 집계 방식이 다르면 이 검정이 검출하지 못할 수 있다. 공식4지점 설명 자체를 부정하는 결과는 아니다.

64행마다 phase를 무작위 회전한 100개 대조와 비교한 비율은 아래와 같다. 수치는 '실제 p4 MAE가 대조보다 작았던 비율'을 구간 평균한 기술통계다. 유의확률이 아니며 큰 값도 효과 크기를 대체하지 못한다.
"""
    )
    control = read("phase_controls")
    tab(
        control.groupby(["condition", "variable"])
        .actual_better.mean()
        .rename("대조보다_낮은_MAE_비율")
        .reset_index(),
        4,
    )
    add(
        """idx위상과 구간 상대위상은 일정 회전이므로 Ridge 예측 오차가 수치 오차 범위에서 동일했다. phase 라벨 회전만으로 제품 시작점을 고를 수 없다.

### 2. 네 행 후보의 개수와 경계

각 r은 같은 자료를 다른 시작점으로 묶은 별도 가정이다. 네 r의 개수를 합산하지 않는다. `4×묶음+미할당=기록 수`를 각 r에서 확인했다.
"""
    )
    counts = read("candidate_counts")
    tab(
        counts.groupby(["condition", "r"])[["groups", "unassigned", "records"]].sum().reset_index(),
        0,
    )
    add("날짜별 완전 후보 수. 상세 미할당 행은 unassigned_rows.csv에 원본 Excel 행으로 보존했다.")
    tab(
        counts[counts.condition == "all"]
        .pivot(index="date", columns="r", values="groups")
        .reset_index(),
        0,
    )
    add(
        """경계 변화량은 '다음 행 phase가 r인 전이'와 나머지 세 내부 전이의 절대변화 평균을 비교했다. 아래는 전압 경계 평균−내부 평균을 학습 전압 표준편차로 나눈 값이다. 양수는 해당 경계 변화가 더 크다는 뜻이며, 물리적 시작점 정답은 아니다.
"""
    )
    boundary = read("boundary_candidates")
    tab(
        boundary[(boundary.condition == "all") & (boundary.variable == "voltage")]
        .pivot(index="date", columns="r", values="standardized_difference")
        .reset_index(),
        4,
    )
    add(
        """어떤 날짜는 r0/r2, 다른 날짜는 r1/r3의 변화가 더 크며 동일 r이 일관되게 유지되지 않는다. 특히 두 위상이 함께 커지는 경우는 2행 값 유지와도 구별해야 한다. 변수별 exact run과 길이2/8/16 경계 대조를 별도 표로 저장했다. H2는 **경계 식별 불가**다.

### 3. lag4·8·12·16과 전압의 국소 반복

아래는 전압의 **원시 행 1차 차분** 상관이다. lag16의 일부 양의 관계는 남지만 lag4가 보편적으로 높지 않다. lag8의 음의 상관은 반대 방향 변화가 나타나는 간격이라는 뜻이며 '반복 없음'과 같지 않다.
"""
    )
    lag = read("lag_metrics")
    subset = lag[
        (lag.condition == "all")
        & (lag.variable == "voltage")
        & (lag.signal == "difference")
        & lag.lag.isin([4, 8, 12, 16])
    ]
    tab(subset.pivot(index="date", columns="lag", values="correlation").reset_index(), 4)
    add(
        "원신호와 학습 phase 중앙값 제거 후의 lag16 비교. 중앙값 제거로 사라지지 않는 관계도 존재한다."
    )
    subset = lag[(lag.variable == "voltage") & (lag.lag == 16)]
    tab(
        subset.pivot(
            index=["condition", "segment"], columns="signal", values="correlation"
        ).reset_index(),
        4,
    )
    add(
        """네 위치별로 뽑은 전압에서 4묶음 간격의 차분 상관은 다음과 같다. 이 차분은 위치별 `X_t−X_(t−4)`이며 위 표의 `X_t−X_(t−1)`과 다르다. 두 표의 차분 상관값을 동일 지표로 비교하지 않는다.
"""
    )
    phase_v = read("voltage_phase")
    tab(
        phase_v[(phase_v.group_lag == 4) & (phase_v.signal == "difference")]
        .pivot(index=["condition", "segment"], columns="phase", values="correlation")
        .reset_index(),
        4,
    )
    add(
        """3월24일 등에서는 네 위치 모두 비슷한 양의 관계가 있고, 높은 가압력 구간이 있는 날짜 전체에서는 약하거나 음수다. 따라서 '특정 한 spot이 lag16을 만든다'는 증거도 확보하지 못했다. 날짜 전체의 상태 변화가 반복 신호를 가릴 가능성은 후속 가설이며 이번 표만으로 원인을 확정하지 않는다.

**H3 판정: lag16 자체는 국소 후보로 보존하되 4지점×4작업이라는 물리적 연결은 미확정.** 동일 데이터에 대한 재해석이며 독립 재현이 아니다. window_profiles.csv의 길이4/8/16 인접 창 상관은 YSH-003의 복합 유사성 통과율과 정의가 다르다.

### 4. 높은 가압력은 실제 연속300행인가?

아래 합계는 떨어진 run을 더한 날짜별 초과 기록 수다. 합계를 하나의 연속 block으로 부르지 않는다.
"""
    )
    force = read("force_runs")
    force_main = force[(force.condition == "all") & (force.r == 0)]
    tab(
        force_main.groupby(["threshold", "date"])
        .length.agg(run_count="count", total_records="sum", longest_run="max")
        .reset_index(),
        0,
    )
    add(
        "주 기준 >3bar의 실제 run 위치 및 네 위치 정렬. 길이168은 4로 나누어지지만 시작 위상이 날짜별로 다르고, 길이123은 나누어지지 않는다."
    )
    tab(
        force_main[force_main.threshold == 3][
            [
                "date",
                "excel_start",
                "excel_end",
                "length",
                "start_phase",
                "end_phase",
                "length_mod4",
            ]
        ],
        0,
    )
    add(
        """길이168에서 시작 phase에 맞춘 r만 42개 완전 후보가 된다. 네 r를 순환시키면 길이가 4의 배수인 run은 원래 4가지 중 1가지가 정렬된다. 이는 독립적인 공정 증거가 아니다. >2.8/3.0/3.2bar에서도 연속300행은 없다. 7.8~8.0bar 값은 해당 네 날짜 각각108행이다.

H4는 **75작업 정렬 주장에 대한 지원 부족**이다. '300행'이 더 넓은 공정 이벤트를 지칭했을 가능성까지 부정하지 않으며, 그 이벤트의 시작·끝 정의는 확보되지 않았다.

### 5. 묶음을 몇 조건 유형으로 나눌 수 있는가?

GMM은 실제 품번이나 제품 종류를 학습하지 않는다. 선택 K와 학습/확인 후보 수를 아래에 모두 제시한다. BIC는 같은 표현·스케일 안에서만 K를 비교했다. 스케일이 다른 BIC 절댓값은 비교하지 않는다.
"""
    )
    model_summary = read("model_summary")
    tab(
        model_summary[
            ["condition", "r", "representation", "scale", "k", "train_n", "test_n", "unique_train"]
        ],
        0,
    )
    add(
        """all의 모든 설정이 검색 상한 K=4를 선택했다. 따라서 '정확히 네 종류'라는 뜻이 아니다. controlled100의 확인 후보는 44~45개이며 4월7일에만 남는다. 확인 분포가 좁아져 모델 비교의 의미도 달라진다.

Standard/Robust의 군집 대응 ARI. 1이면 동일 분할이고 낮을수록 분할이 다르다. 아래 모든 설정이 사전 주의 기준0.6 미만이다.
"""
    )
    sensitivity = read("candidate_sensitivity")
    tab(sensitivity[sensitivity.comparison == "gmm_scale"][["key", "ari"]], 4)
    add(
        "seed·반복 처리별 ARI 범위. test seed 비교의 최솟값은 44~45개 통제 확인 후보 등 작은 표본의 영향도 함께 받는다."
    )
    tab(
        sensitivity[
            sensitivity.comparison.str.startswith("gmm_seed")
            | (sensitivity.comparison == "all_vs_controlled100")
        ]
        .groupby("comparison")
        .ari.agg(["min", "median", "max"])
        .reset_index(),
        4,
    )
    add(
        "예시: 사전 기본 설정 all/r0/S8/Standard의 조건 프로파일. r0는 대표로 보여주는 기본 정렬이며 최적 경계로 선택한 것이 아니다. 범위 평균은 각 네 행 안의 max−min을 평균했다."
    )
    profile = read("candidate_profiles")
    tab(profile[profile.key == "all_r0_S8_standard"].drop(columns="key"), 4)
    add(
        """이 설정은 통전시간의 묶음 내 범위와 높은 가압력·급격한 변화가 있는 묶음을 주로 구분한다. 네 위치 정답을 복원했다는 뜻이 아니다. **조건 유형은 설정 의존**, 제품 종류 수는 미확정이다.

### 6. AE/IF와 스케일링 결과

현재 raw의 학습 MinMax 범위. 과거 scaled_data.xlsx의 극단값 범위를 사용하지 않았다.
"""
    )
    ae_fit = json.loads((HERE / "outputs" / "ae_fit.json").read_text(encoding="utf-8"))
    tab(
        pd.DataFrame({"variable": NAMES, "train_min": ae_fit["min"], "train_max": ae_fit["max"]}), 6
    )
    add(
        f"확인 자료 중 학습 MinMax [0,1]을 벗어나는 행은 {ae_fit['test_outside_any_count']}개다. 모든 확인 조합이 이미 학습에 있으므로 자연스러운 결과다."
    )
    row_models = read("row_models")
    tab(
        row_models[
            [
                "model",
                "threshold",
                "train_flags",
                "test_flags",
                "train_score_mean",
                "train_score_std",
            ]
        ],
        7,
    )
    add(
        """AE 세 seed 모두 학습42행·확인14행을 표시했고 flag 집합은 동일했다. 다만 점수 전체의 seed 간 Spearman은 약0.444~0.639로, 순위 전체까지 안정적이라고 볼 수 없다. 행 IF의 Standard/Robust 결과는 이번 자료에서 동일했다.

확인 AE14행은 모두 4월3일 아래 위치다. 학습 자료의 정확 일치 원본행을 함께 표시했다. 재구성 오차가 크다는 것은 불량 정답이 아니다.
"""
    )
    flagged = read("flagged_row_details")
    tab(
        flagged[(flagged.model == "ae_42") & (flagged.split == "test")][
            ["excel_row", "first_train_exact_excel", "force", "current", "voltage", "time", "score"]
        ],
        6,
    )
    add(
        "행 점수의 묶음 평균/최대값과 묶음 IF 점수를 모두 저장했다. 아래는 날짜별 묶음 IF95% 문턱 초과 수이며, r에 따라 개수가 달라진다. 불량 개수와 비슷한 r를 선택하지 않았다."
    )
    cand_daily = read("daily_candidates")
    sub = cand_daily[cand_daily.key.str.fullmatch("all_r[0-3]_S8_standard")]
    tab(sub.pivot(index="date", columns="r", values="if_flag95_count").reset_index(), 0)
    add("""### 7. 날짜 분할에서 정확 반복은 얼마나 남는가?

width1은 네 변수 한 행, width4/16/100은 연속된 네 변수 sequence다. covered_test_rows는 학습에 존재하는 확인 창들에 한 번이라도 포함된 확인 행의 합집합이다.
""")
    tab(read("overlap_audit"), 0)
    add(
        """확인3,469행의 네 변수 조합은 **전부 학습에 존재**한다. 길이4·16에서도 모든 확인 행이 어떤 일치 창에는 포함된다. 그러나 '모든 확인 창이 일치'하는 것은 아니다. 예를 들어 길이16은 3,424창 중3,274창이 일치한다.

width1/4/16의 겹친 행을 제거하면 확인 자료가0행이므로 새로운 조합에 대한 일반화를 검증할 수 없다. width100 제거 후407행이 남지만, 그407행 역시 개별 네 변수 조합은 모두 학습에 있다. 엄격한 의미의 새로운 공정 상태 검증으로 부르지 않는다.

purged_scores.csv와 purged_candidate_models.csv에 제거 기준별 남은 행/후보 및 flag 수를 보존했다. 제거된 부분이 많다고 무작위 row split으로 대체하지 않았다.

### 8. Result와는 어떻게 대응되는가?

유형1=파임, 유형2=용접부족, 유형3=크랙. NA는 미기록이다. 기록된 유형 개수 합계39는 중복 없는 불량제품 수로 확정할 수 없다.
"""
    )
    tab(
        daily[
            [
                "date",
                "split",
                "records",
                "r0_groups",
                "type_1",
                "type_2",
                "type_3",
                "known_types",
                "if_standard_42_flag_count",
                "ae_42_flag_count",
            ]
        ],
        0,
    )
    add(
        """3월27일의141 IF flag/14 AE flag는 품질기록이 없어 실제 불량 개수와 비교할 수 없다. 3월31일은 유형3 누락이므로 합계5는 부분합이다. 4월3일은 AE14행이지만 기재된 불량 합은4이며, 두 수의 관측 단위도 아직 다르다. 확인3일 전체 AE14와 Result11이 가까워 보이더라도 성능 증거가 아니다.

아래는 **세 유형 모두 기록된7일**의 기재합과 사전 지정 행 지표의 Spearman이다. 날짜 하나를 뺐을 때의 범위도 병기한다. 전체 날짜를 본 사후 기술통계이며 독립 검증·신뢰구간·인과 효과가 아니다.
"""
    )
    quality = read("quality_correlations")
    q = quality[
        (quality.target == "complete_sum")
        & (quality.target_form == "count")
        & ~quality.metric.str.contains(":")
    ]
    tab(q[["metric", "n", "spearman", "leave_one_min", "leave_one_max"]], 4)
    add(
        "유형별 대응은 별도로 보았다. 아래 기본 지표를 결과에 따라 교체하지 않았으며, 분모 보정과 날짜 제외 범위의 전체 값은 quality_correlations.csv에 있다."
    )
    names = [
        "records",
        "force_above3_share",
        "current_median",
        "current_iqr",
        "time_71_share",
        "time_72_share",
        "time_73_share",
        "if_standard_42_flag_count",
        "ae_42_flag_count",
    ]
    sub = quality[
        quality.metric.isin(names)
        & quality.target.isin(["type_1", "type_2", "type_3"])
        & (quality.target_form == "count")
    ]
    tab(sub.pivot(index="metric", columns="target", values="spearman").reset_index(), 4)
    add(
        """유형1/2는8일, 유형3은7일만 있다. 확인 기간만은3일이므로 계획의 최소5일 조건에 따라 상관을 계산하지 않았다. count/records 및 count/candidate는 후보 단위의 노출량 지표이며 실제 불량률이라는 이름을 붙이지 않았다. 표본이 적고 동일 공정 블록이 날짜 간 반복되어 양의 상관이 나와도 독립적인 현장 재현으로 세지 않는다.

## Baseline 대비 변화

| 비교 | 이번 변화 | 해석 |
|---|---|---|
| YSH-003 전압16 모양 후보 | 네 위치 가설 아래 lag4/8/12/16, phase 제거, 위치별 차분 추가 | 국소 후보는 남음. 네 지점·제품 경계 증거로 승격되지 않음 |
| YSH-004 상태·희소성 | 가정된 네 행 S8/P16, 시작점4개, 스케일·seed·반복 통제 | 군집은 모델 설정에 민감. 조건 탐색 도구로만 사용 |
| 공식 AE 구조 | 현재 raw train-only MinMax로 구조 참조 구현 | 과거 scaled snapshot/threshold의 완전 재현이 아님 |
| 날짜 기반 검증 | 동일 조합·4/16/100창 중복 감사 및 제거 | 날짜만 분리해도 독립 검증이 확보되지 않음 |

## 가설과 일치 여부

| 가설 | 판정 | 근거 |
|---|---|---|
| H1 네 행의 위치 구조 | 지원 부족 | p4의 사전5% 개선 기준 충족0/44 |
| H2 묶음 경계 | 미확정 | 시작 위상별 전이 우세가 날짜·변수별로 다름 |
| H3 lag16=네 위치×네 묶음 | 국소 신호 후보, 물리적 연결 미확정 | 일부 차분 상관 유지, H1/H2는 확보되지 않음 |
| H4 높은 가압력75작업 정렬 | 지원 부족 | >3bar 123+168행, 시작 phase도 다름 |
| H5 집계 품질 관계 | 제한적 탐색, 검증 부족 | 품질8일/완전합7일, 반복·누락·단위 불확실 |

## 해석

모델링 자체는 가능하다. 이번 결과는 '현재 입력으로 조건 유형과 특이 기록을 기술할 수 있다'는 수준이다. 실제 spot/제품/불량을 분류하는 모델이라고 부르려면 관측 단위와 행 정답을 추가로 확보해야 한다.

네 위치의 차이가 약하다는 결과는 곧바로 다른 용접 방식이나 다른 품번이 섞였다는 뜻이 아니다. 현재 품번과 장비명은 각각 하나다. 같은 부품의 위치별 값이 비슷한 경우, 한 행이 요약값인 경우, 수집 순서가 문서 가정과 다른 경우를 구분할 앵커가 없다.

이번 결과에서 모델/Feature 개수를 늘리는 우선순위는 낮다. 먼저 같은 데이터셋의 행 정의, spot ID, 작업 시작/종료 이벤트, 제품 instance, 최종 품질 연결을 확인하는 것이 정보 병목을 해소한다. 다른 용접기 CSV의 행을 늘려 합치는 것으로 이 문제는 해결되지 않는다.

## 계획에서 변경된 점

- 최신 사용자 확인을 반영해 lag4/8/12/16을 동등한 비교 후보로 설명하고, 이전 lag16은 보조 재검증으로 명시했다.
- common.py, diagnostics.py를 추가해 입력·설정과 보조 표 생성을 분리했다.
- 길이100~199의 구간은 학습 없이 short_segment_lags.csv에 가능한 원신호/차분 lag만 추가했다. p별 확인 예측과 위치별 주 분석은 계획대로 길이≥200에서 수행했다.
- 대조 블록32/128은 선택적 분석으로 실행하지 않았다. 주 대조64×100회는 실행했다. 추가 탐색으로 H1을 성립시키려 하지 않았다.
- AE용 torch를 YSH-005/.deps에 설치했다. Excel 읽기는 기존 YSH-004/.deps의 openpyxl을 읽기 전용으로 사용했다. 다른 실험 코드는 실행·수정하지 않았다.
- 원 가이드 PDF를 프로젝트에서 찾지 못해 데이터 사전 인용을 근거로 삼았다. 공식 원문을 직접 재확인했다고 주장하지 않는다.

## 오류/실패

- PyTorch CPU 전용 패키지 저장소에 openpyxl이 없어 최초 동시 설치가 실패했다. torch만 전용 폴더에 설치하고 기존 openpyxl을 사용하여 해결했다.
- 설치 도구는 별도 설치 경로의 fsspec과 시스템 datasets의 버전 제약 차이를 알렸다. 이번 실행은 datasets를 사용하지 않았고 공유 설치를 변경하지 않았다.
- 수치 계산·모델 적합·검산의 미해결 실행 오류는 없다. '물리적 경계 미확정'과 '새로운 조합의 확인 표본0'은 분석 결과이며 실행 실패가 아니다.
- YSH-004 추가 검증은 기존 중단 상태를 유지했다. YSH-005 완료를 YSH-004 검산 완료로 사용하지 않는다.

## 재현 확인
"""
    )
    manifest = json.loads((HERE / "outputs" / "manifest.json").read_text(encoding="utf-8"))
    add(
        f"CPU 최대4스레드. Python {manifest['python']}, numpy {manifest['numpy']}, pandas {manifest['pandas']}, scipy {manifest['scipy']}, sklearn {manifest['sklearn']}. torch 버전은 outputs/ae_status.json에 기록했다."
    )
    add(
        f"독립 검산 {verification['check_count']}개 통과. 행 할당/잔여 합계, idx·날짜 경계, 통제본 보존, 정규방정식으로 Ridge 재계산, 차분 상관, 정수 튜플ID로 중복 창 재검사, 저장 scaler/모델의 예측·문턱, AE 재구성 MSE, 원문 Result 및 일별 합계를 대조했다."
    )
    add("""```powershell
$python = 'F:/Kamp/.venv/Scripts/python.exe'
Set-Location F:/Kamp/experiments/YSH/ysh-005
& $python analyze_structure.py
& $python model_candidates.py
& $python compare_result.py
& $python diagnostics.py
& $python verify.py
& $python report.py
```

코드/config/계획/입력/출력 해시는 outputs/manifest.json에 있고, 실제 학습 행 목록·scaler 통계는 candidate_fit.json, ae_fit.json 및 저장 모델에 있다. 원본 Excel은 수정하지 않았다. 재현 명령은 YSH-005 산출물을 덮어쓰므로 이전 결과를 보존하려면 출력 폴더를 먼저 별도 보관해야 한다.

### 보고서에서 찾을 수 없는 상세 값의 위치

| 파일 | 내용 |
|---|---|
| phase_metrics.csv / phase_profiles.csv | 모든 p·구간·변수의 MAE/RMSE, 위치별 분포 |
| lag_metrics.csv / voltage_phase.csv | 원신호·차분·위치제거, 위치별 lag1..8 |
| phase_controls.csv | 100회 phase 대조 개별 결과 |
| candidate_map.csv / unassigned_rows.csv | 각 r의 네 행 원본 대응과 잔여 |
| boundary_candidates.csv / boundary_width_sensitivity.csv | 네 행 및2/8/16 경계 대조 |
| force_runs.csv | 임계값별 run 및 r별 완전/부분 포함 묶음 |
| candidate_scores.csv / candidate_profiles.csv | 군집·후보IF 점수와 조건 프로파일 |
| candidate_sensitivity.csv / row_sensitivity.csv | seed·scale·반복 처리 민감도 |
| row_scores.csv / flagged_row_details.csv | 행 점수·문턱 flag·원본 위치 대응 |
| overlap_audit.csv / overlap_rows.csv | 학습-확인 exact 중복 및 개별 행 표시 |
| purged_scores.csv / purged_candidate_models.csv | 중복 제외 뒤 잔여와 점수 집계 |
| daily_quality_comparison.csv / quality_correlations.csv | 날짜·유형·분모·상관·날짜제외 민감도 |

## 채택 / 보류 / 폐기

- **채택:** 원본행 대응을 보존한 조건·특이성 탐색, 시간 분리와 정확 중복 감사, Result의 날짜/유형별 기술통계.
- **보류:** 네 행을 실제4spot/작업/제품으로 지정, 군집을 품목 수로 해석, 개별 품질 예측 성능 주장.
- **비채택:** lag16을 전제로 만든 제품 ID, 300/4=75제품 계산, Result 개수에 맞춘 top-k/문턱 조정.

## 다음 실험 또는 다음 행동

1. 원 가이드의 수집 테이블/PLC 필드 정의 또는 제품1개 생산 로그를 확보해 한 행과 네 지점의 대응을 확인한다.
2. 대표 고가압·저전압 이벤트의 원본 위치를 현장/가이드와 대조한다. AE14행은 구체적인 확인 후보이지 불량 정답이 아니다.
3. 대응이 확인되면 그 단위로 제품 그룹을 나눈 새로운 기간의 검증 자료를 확보한다. 대응이 끝내 없으면 행·국소창 상태 분석으로 목표를 한정한다.
""")
    text = "\n".join(parts)
    assert not any("\u4e00" <= c <= "\u9fff" for c in text)
    (HERE / "실험결과.md").write_text(text, encoding="utf-8")
    print(f"REPORT COMPLETE: {len(text.splitlines())} lines", flush=True)


if __name__ == "__main__":
    main()
