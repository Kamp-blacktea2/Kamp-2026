"""Generate the numerical report from verified A outputs only."""

from common import *


def table(frame):
    def cell(x):
        if pd.isna(x):
            return "NA"
        if isinstance(x, (float, np.floating)):
            return f"{x:.6g}"
        return str(x).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(map(str,frame.columns)) + " |",
             "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(cell(x) for x in row) + " |" for row in frame.itertuples(index=False,name=None)]
    return "\n".join(lines)


def main():
    verification = json.loads((HERE/"outputs/verification.json").read_text(encoding="utf-8"))
    assert verification["passed"]
    candidates = read("type_candidates")
    corr = read("candidate_correlations")
    daily = read("daily_features")
    primary = daily[(daily.condition=="all") & (daily.reference=="train") & (daily["tail"]==.1)]
    quality = read("daily_quality")
    main_corr = corr[(corr.condition=="all") & (corr.reference=="train") & (corr["tail"]==.1)]
    manifest = json.loads((HERE/"outputs/manifest.json").read_text(encoding="utf-8"))
    parts = []
    def add(title, body):
        parts.extend([f"## {title}",body])
    parts.append("# YSH-006A 실험결과 — 불량 유형과 날짜별 공정조건의 탐색적 연관")
    add("Metadata",f"- 담당: YSH. 완료일: 2026-09-24. 유형: EDA / FEATURE / VALIDATION.\n- CPU 최대 4스레드, 공용 `.venv`. 분석 계산 {manifest['seconds']:.2f}초. GPU와 신규 패키지 설치 없음.\n- Python {manifest['python']}, numpy {manifest['numpy']}, pandas {manifest['pandas']}, scipy {manifest['scipy']}.\n- 설정 SHA-256: `{manifest['config_sha256']}`.\n- YSH-006B는 미실행. A의 결과를 B 설정 선택에 사용하지 않는다.")
    add("실제 수행 내용", "사전 후보 25개 × 건수/1,000기록당 기재 건수 × 3중복 조건 × 2기준 기간 × 2꼬리 기준 = 600개 비교를 모두 기록했다. 공정 Feature를 저장·해시 고정한 뒤 Result를 날짜 단위로 연결했다. 스케일러나 예측 모델을 학습하지 않았으며 단위를 보존했다.\n\n"+table(read("input_audit"))+"\n\nunique4는 날짜 내 동일 네 값의 총가중치를 1로 맞춘 분포다. 날짜 간 반복을 제거하거나 독립 관측을 확보한 것은 아니다. controlled100은 기존 보존 구간으로, Result가 있는 날짜는 3일뿐이므로 200개 비교가 모두 NA다.")
    add("계획에서 변경된 점", "후보·기준값 변경 없음. `common.py`를 공통 정의 모듈로 추가했다. 반복 민감도는 전체 600개 비교를 담은 `candidate_correlations.csv`와 그 요약 `repetition_sensitivity.csv`로 제공한다. 기존 구간 경계는 날짜·idx 불연속뿐 아니라 같은 날짜의 중복 idx를 격리하는 규칙까지 보존했다. 그림·HTML은 생성하지 않았다.")
    counts = candidates.groupby(["defect_type","status"]).size().unstack(fill_value=0).reset_index()
    add("핵심 결과", "**사전 기준을 모두 만족한 후속 후보는 파임과 관련된 전류 MAD 및 높은 전력 P 비중의 2개다.** 나머지는 조건 의존 또는 지지 부족이다. 이는 8일의 집계 연관이며 개별 불량 원인이나 예측 정확도가 아니다.\n\n"+table(counts)+"\n\n전류 MAD의 파임 건수 rho=0.593936, 1,000기록당 rho=0.674748. 건수의 날짜제외 범위 0.387068~0.762770이며 8/8 양의 방향이다. unique4 통제에서는 건수 rho=0.341800으로 약해진다. 전력 P의 기준 p90 초과 비중은 건수 rho=0.506754, 1,000기록당 rho=0.507346, 날짜제외 0.388368~0.854409, 양의 방향 8/8이다. unique4 건수 rho=0.452911이다. 따라서 ‘상대적으로 일관’은 사전 기준상 방향 유지라는 뜻이며 강한 독립 재현을 뜻하지 않는다.")
    add("날짜별 Result와 분모",table(quality)+"\n\n3월27일의 모든 유형 및 3월31일의 유형3은 NA다. 4월7일 파임은 명시적 0이다. 합계 39는 유형별 기재 합계로, 중복 없는 불량제품 39개라고 해석하지 않는다. 분모는 원본 기록 수이며 제품 불량률이 아니다. reference는 3월24~31일 8,470행이고 나머지는 later다. 모든 날짜의 연관 분석 자체는 사후 탐색이다.")
    add("식·단위와 기준값",table(read("proxy_definitions"))+"\n\nP/E/R은 표시값으로 만든 대리지표다. 실제 전력 파형·적분 열량·동적 저항곡선을 복원한 것이 아니다. H는 J가 아닌 A²·s다. F는 bar이며 힘 N이 아니다. E/F, H/F는 탐색용 상대조건이다.\n\n"+table(read("reference_thresholds").query("condition == 'all' and reference == 'train' and tail == 0.1")[["feature","reference_records","reference_transitions","median","low","high","jump","low_ties","high_ties"]])+"\n\nlow/high는 기준 p10/p90, jump는 유효 |차분|의 p90. 경계 동점은 제외하는 strict 부등호다. 중앙값·MAD는 스케일 보정하지 않았고 분위수는 linear다.")
    for kind,name in [(1,"파임"),(2,"용접부족"),(3,"크랙")]:
        subset = candidates[candidates.defect_type==kind]
        add(f"유형 {kind} {name} — 모든 사전 후보",table(subset[["metric","n","rho_count","rho_per1000","loo_min","loo_max","loo_sign_matches","loo_total","rho_weighted_count","rho_weighted_per1000","status"]])+"\n\nloo는 건수 기준 날짜 하나 제외 결과다. weighted는 날짜 내 동일 기록/전이에 총가중치1을 부여한 보조 조건이다. 아래 표는 각 후보의 실제 날짜별 값이며 tail/jump는 0~1 비중이다.\n\n"+table(primary[["date"]+subset.metric.tolist()])+"\n\n"+table(subset[["metric","influential_dates_count","influential_dates_per1000","failed_criteria"]]))
    add("Baseline 대비 변화", "원시변수 4개 × 중앙값/MAD/jump × 유형3개 × 목표2개 = 72개 비교다. 예측 모델을 학습하지 않았으므로 성능 향상 점수는 없다. 일관 후보는 원시 전류 MAD와 전력 P의 높은 꼬리 비중이다. P는 추가 탐색 표현으로 남지만 예측 정보가 실제로 늘었는지는 검증하지 않았다. 에너지 E와 전류-시간 H의 높은 꼬리는 기준에 미달했다. 기존 YSH-005와 날짜별 Result를 대조했다.\n\n"+table(read("raw_baseline")))
    sensitivity = corr[["candidate","condition","reference","tail","target","n","rho","reason"]]
    save(sensitivity,"repetition_sensitivity")
    sens = corr[corr.condition!="controlled100"].pivot(index=["candidate","target"],columns=["condition","reference","tail"],values="rho")
    sens.columns = ["/".join(map(str,c)) for c in sens.columns]
    add("기준값·중복 민감도 전체",table(sens.reset_index())+"\n\nfull은 전체기간을 이용한 기술통계 민감도이며 미래 예측 검증이 아니다. p5/p95는 꼬리 및 jump 경계만 바꾸며 날짜별 고정 p90 통계를 p95로 바꾸지 않는다. controlled100은 원본 기준 문턱과 원본 Result 분모를 유지했다.")
    add("보존 구간 coverage",table(daily[(daily.condition=="controlled100") & (daily.reference=="train") & (daily["tail"]==.1)][["date","original_records","distribution_records","transitions","coverage"]]))
    add("날짜제외 검증 상세",table(main_corr[["candidate","target","n","rho","loo_min","loo_median","loo_max","sign_matches","loo_valid","loo_total","influential_dates"]])+"\n\nNA 또는 |rho|<0.2가 되는 제외 날짜를 영향 날짜로 표시했다. 부호가 유지되어도 크기가 이 기준 아래로 떨어지면 일관 후보에서 제외했다. 날짜제외는 민감도 검사이며 독립 시험셋 성능이 아니다.")
    add("높은/낮은 기재 날짜 비교",table(read("high_low_dates")[["candidate","target","low_dates","high_dates","high_minus_low","reason"]])+"\n\n목표 p25 이하와 p75 이상을 동점 포함해 비교한 Feature 중앙값 차이다. 목표에 따라 사후 그룹을 정했으므로 예측 검증으로 사용하지 않는다.")
    examples = read("counterexamples")
    add("후보별 반례", "반례는 기대한 양의 방향과 반대로 움직이는 날짜 쌍이다. 개수만으로 유의성을 판정하지 않는다. 전체 쌍은 CSV에 보존했다. 아래에는 사전 순서상 첫 반례를 후보·목표별 하나씩 제시한다.\n\n"+table(examples.drop_duplicates(["candidate","target"])[["candidate","target","date_a","date_b","feature_a","feature_b","target_a","target_b"]]))
    block = read("repeated_blocks_quality")
    columns = [c for c in ["a_date","b_date","a_start","a_end","b_start","b_end","length","a_type1","b_type1","a_type2","b_type2","a_type3","b_type3"] if c in block]
    add("날짜 간 동일 블록과 품질",table(block[columns])+"\n\n날짜 쌍마다 가장 긴 100행 이상 블록 하나를 표시했다. 같은 궤적을 포함해도 날짜 전체의 Result는 같지 않을 수 있다. 이는 블록 자체의 불량 판정이 다르다는 뜻이 아니다. Result의 행/제품 대응이 없으므로 날짜 라벨을 해당 블록에 붙이지 않는다.")
    add("대리지표의 종속성",table(read("feature_dependence"))+"\n\nP/E/H 등은 같은 Raw에서 계산된다. 여러 후보에서 유사한 상관이 나와도 독립 증거 수가 늘지 않는다. 일별 일정한 기준 중앙값을 빼는 것 역시 Spearman을 바꾸지 않는다.")
    add("해석", "파임은 전류의 날짜 내 산포 및 높은 전력 P 비중과의 연관을 후속 확인할 만하다. P와 시간까지 곱한 E는 구별해야 하며 E의 높은 꼬리는 기준에 미달했다. 다만 반복 통제 후 효과가 약해져 기록 구성의 영향을 함께 봐야 한다. 용접부족의 낮은 I/E/H 비중은 건수와 음의 상관, 기록량 보정과 양의 상관으로 바뀌어 단순 저에너지 설명을 지지하지 않는다. 크랙의 전압·저항 급변은 보정 지표에서 커 보이지만 건수 관계와 하루 제외 결과가 약해 일관 후보가 아니다. 현재 자료로 세 불량 유형의 원인·안전 공정영역을 확정할 수 없다.")
    add("가설과 일치 여부", "파임의 변동성 가설은 전류 MAD 한 후보에서 제한적으로 부합했다. 높은 전력 P 비중도 제한적으로 부합했으나, E/H의 높은 꼬리·높은 가압 비중은 사전 기준에 미달했다. 용접부족 가설은 지지 부족이다. 크랙 가설은 주로 분모와 특정 날짜에 의존했다. 지지 부족은 해당 물리 메커니즘의 부재를 입증하지 않는다.")
    add("오류/실패", "첫 실행은 `tail` 열을 속성으로 참조해 pandas 메서드와 충돌하며 후보 판정에서 중단됐다. 열 인덱싱으로 수정 후 전체 분석을 재실행했다. 검산 초안은 중복 idx 격리 경계와 Result의 추가 열을 빠뜨려 중단됐고, 기존 경계 정의와 명시적 열 선택을 반영했다. 분석 후보·문턱·입력·결과를 이에 맞춰 조정하지 않았다. 최초 보고서 초안에서 일관 후보를 1개로 잘못 요약한 부분은 표와 대조하여 2개로 정정했다. 최종 검산은 아래와 같이 통과했다.")
    add("재현 확인",f"독립 계산 검산 **{verification['checks']:,}개 통과**. 원본 값/행 대응, SI 변환, 날짜·idx 중복 격리 경계, 원본 차분, 모든 기준 분위수·날짜 MAD·꼬리·공동 조건·jump, 600개 순위 상관, 모든 날짜제외 순위 상관, NA/0·원본 분모 및 기존 Result를 확인했다. Spearman은 검산에서 평균순위의 Pearson으로 다시 계산했다. 허용오차 rtol=1e-11, atol=1e-12.\n\n```powershell\nSet-Location F:/Kamp\n.\\.venv\\Scripts\\python.exe -B experiments/YSH/ysh-006a/analyze.py\n.\\.venv\\Scripts\\python.exe -B experiments/YSH/ysh-006a/verify.py\n.\\.venv\\Scripts\\python.exe -B experiments/YSH/ysh-006a/report.py\n```\n\n`manifest.json`은 계산 당시 입력·코드 해시, `verification.json`은 검산 이력이다. 최종 문서의 완료 상태 변경은 `delivery_manifest.json`에 별도 해시로 기록한다. 반복 실행 시 기존 outputs를 같은 설정으로 갱신한다.")
    add("결론", "사전 후보 25개 중 후속 확인 대상으로 남길 것은 전류 MAD 및 높은 전력 P 비중과 파임의 집계 관계 2개다. 조건 의존 후보는 7개, 지지 부족은 16개다. 기록량·반복·날짜의 제약이 커서 새로운 지도 모델이나 공정 임계값 설정을 정당화하는 결과는 아니다.")
    add("채택 / 보류 / 폐기", "- 채택: 단위가 명시된 대리지표와 날짜별 진단 표, 전류 MAD 및 높은 전력 P 비중 가설의 후속 확인.\n- 보류: 파생변수의 예측 유효성, 불량 원인 규명, 안전 공정영역, 행/제품 라벨링.\n- 폐기하지 않음: 이번 자료에서 미지지된 물리 가설 자체. 데이터 한계와 반증을 구분한다.")
    add("다음 실험 또는 다음 행동", "전류 MAD 및 높은 전력 P 비중과 파임의 관계는 새로운 생산일·실제 검사 수량·제품/spot 식별자 확보 후 확인한다. 동일 파형 재사용의 영향을 분리할 수 있는 검증 단위도 필요하다. B는 별도 사용자 실행 요청까지 계획 상태로 유지한다. B를 실행하더라도 이번 A의 결과에 맞춰 motif 설정을 바꾸지 않는다.")
    add("산출물 안내", "핵심 해석은 이 보고서에 포함했다. 세부 수치는 `outputs/tables/`의 CSV, 고정 후보·식은 `config.json`, 검산은 `outputs/verification.json`에 있다. 그림 탐색기나 HTML은 필요하지 않다.")
    (HERE/"실험결과.md").write_text("\n\n".join(parts)+"\n",encoding="utf-8")
    print("Report written",candidates.status.value_counts().to_dict())


if __name__ == "__main__":
    main()
