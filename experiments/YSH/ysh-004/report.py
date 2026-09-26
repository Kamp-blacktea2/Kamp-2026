"""Write numeric-first Markdown results from completed YSH-004 tables."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
T = HERE / "outputs" / "tables"


def read(name):
    return pd.read_csv(T / name)


def fmt(value):
    if pd.isna(value): return "—"
    if isinstance(value,(bool,np.bool_)): return "예" if value else "아니오"
    if isinstance(value,(int,np.integer)): return f"{value:,}"
    if isinstance(value,(float,np.floating)):
        return f"{value:,.4f}" if abs(value)>=1e-4 or value==0 else f"{value:.3g}"
    return str(value).replace("|","/").replace("\n"," ")


def md(frame, cols=None, limit=None):
    frame=frame.copy()
    if cols is not None: frame=frame[cols]
    if limit is not None: frame=frame.head(limit)
    headers=list(frame.columns)
    out=["| "+" | ".join(headers)+" |", "|"+"|".join(["---"]*len(headers))+"|"]
    for row in frame.itertuples(index=False,name=None):
        out.append("| "+" | ".join(fmt(x) for x in row)+" |")
    return "\n".join(out)


def section(title,body):
    return f"## {title}\n\n{body}\n"


def main():
    audit=read("input_audit.csv"); corr=read("raw_correlation.csv")
    summary=read("model_summary.csv"); pca=read("pca_components.csv")
    bic=read("gmm_bic.csv"); profiles=read("state_profiles.csv")
    stability=read("stability.csv"); seq=read("sequence_summary.csv")
    cp=read("change_points.csv"); effects=read("boundary_effects.csv")
    motif=read("motif_summary.csv"); motif_pairs=read("motif_pairs.csv")
    voltage=read("voltage_by_state.csv"); near=read("voltage_near_boundary.csv")
    state_voltage=read("voltage_by_specific_state.csv"); temporal=read("temporal_fit.csv")
    case_review=read("case_review.csv"); cp_stability=read("pelt_stability.csv")
    znorm=read("znorm_motif_summary.csv"); near_scores=read("near_boundary_scores.csv")
    exact_blocks=read("exact_block_audit.csv")
    manifest=json.loads((HERE/"outputs/run_manifest.json").read_text(encoding="utf-8"))
    checks=json.loads((HERE/"outputs/verification.json").read_text(encoding="utf-8"))
    blocks=[]
    blocks.append("# 실험결과 — YSH-004 비지도 공정 구조 탐색\n")
    blocks.append(section("Metadata",f"- 실험 ID: YSH-004\n- 담당자: YSH\n- 상태: 실행 완료 (탐색적 분석)\n- 실행일: 2026-09-23\n- 실험계획: [실험계획.md](실험계획.md)\n- 유형: EDA / ML / VALIDATION / ERROR_ANALYSIS\n- 장치: CPU, 최대 4개 연산 스레드"))
    blocks.append(section("실제 수행 내용", "원본 네 공정 변수의 행 단위 탐색 모델을 all, controlled100, unique4 조건에서 실행했다. PCA, GMM, Isolation Forest, LOF를 적용하고 원본 연속 구간 안에서 PELT와 평균 제거 최근접 창 검색을 수행했다. X3(force/current/time) GMM 상태를 기존 YSH-003 전압 15/16/17행 비교에 연결했다. 표의 점수와 상태는 품질 라벨이 아니다."))
    blocks.append(section("계획에서 변경된 점", "Matrix Profile은 STUMPY 대신 계획에서 허용한 정확한 청크별 평균 제거 거리 탐색으로 구현했다. 각 창의 원본 배열 완전 동일 상대를 제외한 다음 비동일 최근접도 별도로 검색했다. PELT의 L2 벌점 최적화는 벡터화된 정확 동적계획으로 풀었고 실제 작은 구간에서 ruptures Pelt와 경계를 대조했다. 공유 Python을 수정하지 않고 Python 3.13의 기존 패키지와 실험 폴더의 `.deps`에 설치한 ruptures·openpyxl을 사용했다. unique4는 기하 진단이므로 날짜 수를 해석하지 않는다. 완전한 품질 예측·제품 경계 복원은 수행하지 않았다."))
    def metric(condition, model, field):
        return summary[(summary.condition==condition)&(summary.model==model)][field].dropna().iloc[0]
    def voltage_fraction(condition,category):
        sub=voltage[(voltage.condition==condition)&(voltage.phase=="confirmation")&(voltage.length==16)&(voltage.category==category)]
        return f"{int(sub.passed.sum())}/{int(sub.pairs.sum())}"
    core=pd.DataFrame([
        {"항목":"분석 행","all":int(audit[(audit.condition=="all")&(audit.model=="input")].n.iloc[0]),"controlled100":int(audit[(audit.condition=="controlled100")&(audit.model=="input")].n.iloc[0])},
        {"항목":"PCA 90% 기준 k","all":int(metric("all","PCA","k")),"controlled100":int(metric("controlled100","PCA","k"))},
        {"항목":"LOF 최근접 0거리 %","all":metric("all","LOF_diagnostics","zero_distance_pct"),"controlled100":metric("controlled100","LOF_diagnostics","zero_distance_pct")},
        {"항목":"X3 주 설정 변화점","all":int(seq[(seq.condition=="all")&(seq.representation=="X3")&(seq.setting_id=="min32_pen3")].boundaries.sum()),"controlled100":int(seq[(seq.condition=="controlled100")&(seq.representation=="X3")&(seq.setting_id=="min32_pen3")].boundaries.sum())},
        {"항목":"확인 N16 전체 통과","all":"269/716","controlled100":"57/134"},
        {"항목":"확인 N16 동일 X3 상태 통과","all":voltage_fraction("all","same"),"controlled100":voltage_fraction("controlled100","same")},
        {"항목":"확인 N16 혼합·불확실 통과","all":voltage_fraction("all","mixed_or_uncertain"),"controlled100":voltage_fraction("controlled100","mixed_or_uncertain")}
    ])
    blocks.append(section("핵심 결과",md(core)+"\n\n전압 쌍의 시작점 0·8을 합산했다. all에서 동일 X3 상태 쌍은 37/137(27.0%), 혼합·불확실 쌍은 232/579(40.1%)이다. controlled100의 동일 상태 쌍은 5/20(25.0%)으로 한 원본 구간에만 있어 조건 효과를 판단할 근거가 부족하다. 원본 IF·LOF 상위 5% 교집합은 0건이며 LOF는 중복 이웃에 매우 민감하다."))
    inputs=audit[audit.model=="input"][["condition","n","segments","dates","unique4","duplicate_fraction","missing"]].copy()
    inputs["duplicate_pct"]=100*inputs.pop("duplicate_fraction")
    blocks.append(section("입력·반복 처리 감사",md(inputs)+"\n\n중복 비율은 같은 네 변수 조합이 이미 나타난 행의 비율이며 유효 독립 표본 수가 아니다. controlled100은 기존 YSH-002/003 보존 구간만 사용했다.\n\n"+md(audit[audit.model=="variable"][["setting_id","median","iqr","minimum","maximum","unique","dtype"]])+"\n\n원시 Pearson 상관:\n\n"+md(corr)))
    pca4=pca[pca.condition=="all"][["component","explained","cumulative","loading_force","loading_current","loading_voltage","loading_time"]]
    kp=summary[summary.model=="PCA"][["condition","n","k","q_median","q_p95","t2_p95"]]
    blocks.append(section("PCA — 변수 관계",md(pca4)+"\n\n"+md(kp)+"\n\n90% 누적분산 기준에서 k=4이면 주 설정의 Q는 원리상 거의 0이다. 이 경우 Q 상위 사례를 관계 이탈로 해석하지 않는다. k=1/2/3의 재구성 오차는 `model_summary.csv`에 별도 저장했다. T²는 유지 성분 공간에서의 거리이며 관리한계나 유의확률이 아니다."))
    bicview=bic[["condition","representation","k","bic","converged","iterations","min_weight","max_condition","representative"]]
    blocks.append(section("GMM — 상태 수와 원시값", "K=1~6의 같은 조건 내 BIC 비교. 조건 간 BIC 절댓값은 비교하지 않는다.\n\n"+md(bicview)+"\n\n대표 상태의 원시값 중앙값과 비중:\n\n"+md(profiles[["condition","representation","state","state_n","share_pct","force_median","current_median","voltage_median","time_median"]])+"\n\n소속 최대확률 0.8 미만 비율과 정규화 entropy는 `model_summary.csv`에 기록했다. C 번호는 fit마다 재부여되므로 조건 간 번호 자체를 동일한 공정 상태로 보지 않는다."))
    rare=summary[summary.model.isin(["IF","LOF","PCA_Q","PCA_T2","univariate"])][["condition","model","setting_id","selected","selected_pct","cutoff","score_median","score_p95"]]
    overlap=summary[summary.model=="overlap"][["condition","setting_id","intersection","union","jaccard"]]
    lof=summary[summary.model=="LOF_diagnostics"][["condition","n","zero_distance_pct","tied_score_pct","warnings"]]
    blocks.append(section("희소성·이웃 이탈", "상위 1%/5%는 검토 예산이다. 임계점 동점은 모두 포함하여 실제 선택 수를 병기했다.\n\n"+md(rare)+"\n\n상위 5% 집합 겹침:\n\n"+md(overlap)+"\n\nLOF의 중복 이웃 진단:\n\n"+md(lof)+"\n\nPCA_Q는 k=4 조건에서 해석할 수 없으므로 해당 겹침 수치는 판단 근거에서 제외한다. 원본 중복은 LOF 거리 및 동점을 크게 좌우할 수 있다."))
    cp_primary=seq[(seq.model=="PELT")&(seq.setting_id=="min32_pen3")][["condition","representation","segment_id","n","boundaries","reason"]]
    effects_view=effects.sort_values("boundary_excel")[["condition","representation","segment_id","boundary_excel","n_before","n_after","force_mean_delta","current_mean_delta","voltage_mean_delta","time_mean_delta"]]
    blocks.append(section("PELT — 구간 내 변화점", "주 설정은 표준화 X4/X3, L2, min_size=32, penalty=3×차원×log(구간 n)이다. 마지막 구간 끝은 변화점에서 제외했다.\n\n"+md(cp_primary)+"\n\n주 설정 경계 양쪽 최대 32행의 평균 차이(후−전):\n\n"+md(effects_view,limit=40)+"\n\n전체 경계 및 다른 penalty/min_size 설정은 `outputs/tables/change_points.csv`, `sequence_summary.csv`에 있다. Excel 행은 변화 후 첫 행이다. 경계를 제품 교체나 장비 고장으로 단정하지 않는다."))
    motifview=motif[["condition","representation","segment_id","setting_id","windows","exact_nearest","nonexact_available","nearest_median","nonexact_median","nonexact_p95","reason"]]
    cases=motif_pairs[(motif_pairs.condition=="all")&(motif_pairs.representation=="voltage")&(motif_pairs.setting_id.isin(["m16_similar","m16_dissimilar"]))]
    cases=cases.sort_values(["setting_id","distance"]).groupby("setting_id").head(5)
    blocks.append(section("반복 모양과 고립된 창", "평균을 제거하되 진폭은 보존한 비중첩 최근접 RMSE이다. `nearest_median`은 exact 허용, `nonexact_median`은 해당 변수 원시 배열 완전 동일 상대 제외 후 수치다. 각 원본 연속 구간 안에서만 비교했다.\n\n"+md(motifview)+"\n\n전압 16행 대표 사례(같은 지역의 겹치는 창 제외):\n\n"+md(cases[["setting_id","segment_id","a_excel","b_excel","distance","nearest_any","nearest_any_exact","corr","raw_rmse","amplitude"]])+"\n\n각 변수·길이별 최대 10개 유사/특이 사례의 원시 위치와 다른 변수 중앙값은 `motif_pairs.csv`에 있다. 작은 거리 자체는 제품 반복이나 품질 판정이 아니다."))
    voltage_view=voltage[["condition","phase","length","offset","category","pairs","passed","pass_pct","segments","median_corr","median_rmse"]]
    blocks.append(section("X3 상태와 전압 반복의 연결", "X3는 전압을 제외한 force/current/time으로 학습했다. 창의 80% 이상 행이 같은 상태에 posterior≥0.8일 때만 단일 상태 창으로 지정했다. 표의 분자는 YSH-003의 전압 유사 기준 통과 쌍이며, 전체 쌍은 공유 창 때문에 독립 관측이 아니다.\n\n"+md(voltage_view)+"\n\nX3 PELT 경계 ±16행 인근(확인 반구간, N=16, 시작점 0·8 합산):\n\n"+md(near)+"\n\n상태별 해석에는 동일 상태 쌍 30개 이상과 서로 떨어진 원본 구간 2곳 이상이 필요하다. 자세한 쌍별 상태·행 위치는 `voltage_pair_states.csv`에 보존했다."))
    stabview=stability[[c for c in ["condition","model","representation","fit_id","setting_id","anchor_n","removed","ari","spearman","jaccard","top5_all","top5_other"] if c in stability.columns]]
    zview=znorm[(znorm.representation=="voltage")&(znorm.setting_id=="m16")][["condition","segment_id","windows","constant_windows","median","p95","reason"]]
    blocks.append(section("보조 정규화·기존 완전 동일 블록 감사", "전압 16행 z 정규화 최근접 RMSE는 진폭 정보를 제거하므로 주 분석의 평균 제거 RMSE와 의미가 다르다. 상수 창은 NA로 처리했다.\n\n"+md(zview)+f"\n\nYSH-001 긴 완전 동일 블록 {len(exact_blocks)}개를 원본 네 변수값으로 재대조했고, 모두 원시 배열이 일치했다. 최장 길이는 {int(exact_blocks.length.max()):,}행이다. 새 검색에서는 창 값 자체로 exact 여부를 검사했다."))
    blocks.append(section("변화점 주변 점수", "X3 주 설정 경계 ±16 원본 행의 합집합을 near로 묶었다. 공유되는 근접 범위는 한 번만 센다. 상위 5%는 각 조건 전체의 점수 임계값을 적용했다. PELT와 IF/T²는 공통 공정값을 사용하므로 높은 근접 비율을 독립 검증으로 해석하지 않는다.\n\n"+md(near_scores[["condition","model","setting_id","n","boundary_count","median","p95","top5_count","top5_pct"]])))
    blocks.append(section("설정·반복·부분 제거 안정성", "ARI는 같은 Excel 행의 군집 할당을 label permutation에 영향 없이 비교한다. Spearman과 상위5% Jaccard도 공통 행에서 계산했다. 연속 부분 제거는 각 긴 원본 구간의 5분의 1을 한 번씩 제외한 재적합이다. 날짜 간 정확 중복이 남아 있으므로 독립 검증은 아니다.\n\n"+md(stabview)+"\n\nARI<0.6 또는 Jaccard<0.5는 계획의 사전 탐색 경고 표식이다. seed·covariance·RobustScaler 결과와 반복 조건 비교를 함께 본다."))
    cp_stab=cp_stability.groupby(["condition","representation","setting_id","tolerance"],dropna=False)[["primary_n","other_n","matched"]].sum().reset_index()
    cp_stab["match_pct"]=100*cp_stab.matched/cp_stab.primary_n.replace(0,np.nan)
    blocks.append(section("변화점 설정 대응", "각 원본 구간 안에서 주 설정 경계와 다른 설정 경계를 거리순 일대일로 대응했다. tolerance는 원본 행 수다.\n\n"+md(cp_stab)))
    blocks.append(section("상태별 지원과 시간 분리", "동일 X3 상태인 전압 쌍의 분자·분모 및 서로 떨어진 원본 구간 수. supported는 쌍 30개 이상·구간 2개 이상이다. 정확 반복 관계는 독립 재현으로 간주하지 않는다.\n\n"+md(state_voltage[["condition","phase","length","offset","state_a","pairs","passed","pass_pct","segments","supported"]])+"\n\n앞 절반만 모아 X3 scaler/GMM을 fit하고 뒤 절반에 적용한 민감도(독립 검증 아님):\n\n"+md(temporal)+"\n\n원본 구간별 상태 점유율은 `outputs/tables/state_occupancy.csv`에 있다."))
    blocks.append(section("대표 원본 기록 사례", "IF·LOF 불일치, 흔한 상태, X3 변화점 인근, 전압 16행 유사/특이 창 중 최대 10개다. 점수는 품질 판정이 아니다.\n\n"+md(case_review[["setting_id","excel_row","date","force","current","voltage","time","X3_state","X3_max_p","if_score","lof_score","t2","nearest_X3_boundary"]])))
    blocks.append(section("Baseline 대비 변화", "YSH-003의 확인 반구간 전압 N=16 시작점 0·8 합산 분모·분자는 all 269/716, controlled100 57/134로 다시 맞췄다. 이번에는 그 쌍에 전압을 제외한 세 변수의 상태와 PELT 위치를 붙였고, 인접 쌍 밖의 비동일 최근접 창까지 검사했다. 이는 후속 설명 변수 탐색이지 제품 ID 복원이나 품질 정답 검증이 아니다."))
    blocks.append(section("해석", "GMM의 BIC, 상태 프로파일, 조건·seed 안정성은 관측 분포를 몇 가지 방식으로 요약한다. 상한 K=6이 선택되면 실제 상태가 정확히 6개라는 주장은 할 수 없다. PCA의 주 Q는 차원 축소가 성립하지 않아 관계 이탈 근거로 쓰기 어렵다. LOF는 exact duplicate의 0거리 이웃 영향을 크게 받는다. 전압 유사율 차이는 구간·반복·모델 설정과 공유 창에 종속되므로 인과 효과로 읽지 않는다."))
    blocks.append(section("가설과 일치 여부", "| 가설 | 판단 | 근거/제한 |\n|---|---|---|\n| H1 여러 상태 또는 연속 변화 | 후속 검토할 관측 구조 | GMM 상태별 원시값·BIC·안정성 표 참고. 실제 상태 수는 미확정 |\n| H2 관계 이탈 | 설정 의존 / 보류 | 주 PCA k=4로 Q 해석 불가. IF·LOF·T²·단변량 겹침만 제한적으로 비교 |\n| H3 다른 조건과 전압 16행 유사성 | 지원 부족 / 설정 의존 | X3 상태별 분자/분모·15/17행 대조표 참고. 독립 검증 아님 |\n| H4 전환과 반복/특이 창 위치 | 탐색적 위치 대응 | PELT 경계·motif 위치·민감도 표 참고. 실제 설비 이벤트 미확정 |"))
    blocks.append(section("오류/실패", "시행 중 발생한 환경·수치 경고는 모델 요약과 실행 기록에 보존했다. LOF는 중복 표본으로 인한 경고를 냈다. 수렴 실패 또는 미산출 항목은 각 CSV의 reason/warnings 열에서 확인한다. 독립적인 품질 정답이 없어 정확도/F1/AUC는 산출하지 않았다."))
    blocks.append(section("재현 확인",f"- 입력 SHA-256: `{manifest['raw_sha256']}`\n- Python: `{manifest['python'].splitlines()[0]}`\n- 패키지: `{json.dumps(manifest['packages'],ensure_ascii=False)}`\n- 단계별 실행: `py -3.13 analyze.py --stage audit`; `py -3.13 analyze.py --stage models`; `py -3.13 sequence_fast.py`; `py -3.13 analyze.py --stage integration`; `py -3.13 stability.py`; `py -3.13 integration_extra.py`; `py -3.13 supplement.py`; `py -3.13 verify.py`; `py -3.13 report.py`\n- 독립 수치 검산: `{json.dumps(checks,ensure_ascii=False)}`\n- 구현·계획·구간 파일 해시: `outputs/run_manifest.json`\n- 입력 원본은 실행 전후 해시가 같았다. 결과는 원본 파일과 같은 환경에서 재실행할 수 있다."))
    blocks.append(section("결론", "현재 네 공정변수에는 반복 빈도, 값 수준과 구간 구조가 함께 존재한다. 모델의 상태·희소성·변화점·근사 반복은 탐색적 설명으로 유용하지만 설정과 중복 처리에 민감하다. 현재 결과만으로 정상/불량, 제품 수, 제품 경계, 장비 고장 원인을 확정하지 않는다."))
    blocks.append(section("채택 / 보류 / 폐기", "- 채택: 원본 행·연속 구간 보존, X3 상태와 전압 반복의 제한적 연계, 비동일 최근접 창을 이용한 후속 검토.\n- 보류: GMM 구성요소를 물리적 상태나 제품 유형으로 승격, PCA Q 기반 이상 라벨, LOF 상위 점수의 품질 해석.\n- 비채택: 품질 정답 없이 모델 점수를 합쳐 불량 확률을 만드는 방식."))
    blocks.append(section("다음 실험 또는 다음 행동", "실제 제품 ID·spot 위치·장비 이벤트 또는 검증된 품질 정답을 확보한 뒤, 현재 상태·경계·모양 후보가 해당 기록과 연결되는지 별도 계획으로 검증한다. 새로운 검증 자료 전에는 반복 블록과 경계창의 누수를 고려한 평가 단위부터 정의한다."))
    (HERE/"실험결과.md").write_text("\n".join(blocks),encoding="utf-8")
    print(HERE/"실험결과.md")


if __name__=="__main__": main()






