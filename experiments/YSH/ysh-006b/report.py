"""Text/table report of motif evidence and limits, built only from B outputs."""

from engine import *


def table(frame):
    def cell(value):
        if pd.isna(value):
            return "NA"
        if isinstance(value,(float,np.floating)):
            return f"{value:.6g}"
        return str(value).replace("|","/").replace("\n"," ")
    lines = ["| "+" | ".join(frame.columns)+" |", "| "+" | ".join(["---"]*len(frame.columns))+" |"]
    lines.extend("| "+" | ".join(cell(x) for x in row)+" |" for row in frame.itertuples(index=False,name=None))
    return "\n".join(lines)


def main():
    config = json.loads((HERE/"config.json").read_text(encoding="utf-8"))
    verified = json.loads((HERE/"outputs/verification.json").read_text(encoding="utf-8"))
    assert verified["passed"]
    catalog = read("motif_catalog")
    scores = read("match_summary")
    contexts = read("all_context_summary")
    g = read("null_G_summary")
    t = read("null_summary")
    wide = read("wide_gap_summary").set_index("family")
    occurrence = read("occurrences")
    primary = scores[(scores.setting=="primary")&(scores.condition=="all")&(scores.threshold==.25)&
                     (scores.exclusion==1)&~scores.initial_excluded]
    window_audit = read("window_audit")
    audit_primary = window_audit[window_audit.setting=="primary"].reset_index(drop=True)
    later_windows = int(audit_primary[audit_primary.date>"2020-03-31"].windows.sum())
    assert later_windows==3460
    confirmation = primary[["view","representation","anchor_excel","anchor_segment","anchor_date",
                            "discovery","discovery_eligible","same_back","other_date","temporal_confirmation"]].copy()
    lengths = audit_primary.rows.to_numpy()[confirmation.anchor_segment.to_numpy(int)]
    confirmation["same_back_eligible"] = lengths-4+1-(.6*lengths).astype(int)
    by_date = audit_primary.groupby("date").windows.sum()
    confirmation["other_date_eligible"] = audit_primary.windows.sum()-confirmation.anchor_date.map(by_date)
    forward = confirmation[confirmation.anchor_date<="2020-03-31"].copy()
    forward["temporal_eligible"] = later_windows
    save(confirmation,"confirmation_denominators")
    save(forward,"temporal_confirmation")
    sensitivity_rows = scores[(scores.setting=="primary")&(scores.condition=="all")&
                              (scores.threshold!=.25)&(scores.exclusion==1)&~scores.initial_excluded]
    sensitivity_index = sensitivity_rows.groupby(["anchor_excel","view","representation"]).discovery.max()
    context_index = contexts[(contexts.length==8)&(contexts.before==0)].set_index("family")
    gap_index = g.set_index(["family","segment"])
    order_index = t[t.metric=="max_rate"].set_index(["view","representation"])
    order_anchors = read("order_anchor_scores").set_index("family")
    control_index = occurrence[occurrence.condition=="controlled100"].groupby("family").agg(
        occurrences=("target_excel","size"),dates=("target_date","nunique"))
    decisions = []
    for row in catalog.itertuples():
        similar = sensitivity_index.get((row.anchor_excel,row.view,row.representation),0)>=3
        c = context_index.loc[row.family] if row.family in context_index.index else None
        n_context = int(c.valid) if c is not None else 0
        context_rate = float(c.rate) if c is not None else np.nan
        gap_key = (row.family,row.anchor_segment)
        gap = gap_index.loc[gap_key] if gap_key in gap_index.index else None
        order = order_index.loc[(row.view,row.representation)]
        control_family = row.family.removesuffix("_all")+"_controlled100"
        control_occ = control_index.loc[control_family] if control_family in control_index.index else None
        local = row.discovery>=3 and similar
        context_ok = n_context>=5 and context_rate>=.5
        order_above = bool(order.observed>order.p95)
        own_rate = order_anchors.loc[row.family,"rate"] if row.family in order_anchors.index else np.nan
        own_count = order_anchors.loc[row.family,"matches"] if row.family in order_anchors.index else 0
        own_order_above = bool(own_count>=3 and own_rate>order.p95)
        gap_above = bool(gap is not None and gap.gaps>=5 and gap.null_valid>0 and gap.max_share>gap.get("peak_p95",np.nan))
        wg = wide.loc[row.family]
        wide_agrees = bool(gap is not None and np.isfinite(gap["mode"]) and wg["mode"]==gap["mode"] and gap["mode"]>4)
        decisions.append(dict(family=row.family,view=row.view,representation=row.representation,
                              anchor_excel=row.anchor_excel,local=local,context_n=n_context,context_rate=context_rate,
                              context_supported=context_ok,T_view_above_p95=order_above,
                              T_own_rate=own_rate,T_own_above_max_null_p95=own_order_above,
                              G_above_p95=gap_above,wide_mode=wg["mode"],wide_same_nonminimum_mode=wide_agrees,
                              periodic_screen=local and context_ok and own_order_above and gap_above and wide_agrees,
                              controlled_occurrences=int(control_occ.occurrences) if control_occ is not None else 0,
                              controlled_dates=int(control_occ.dates) if control_occ is not None else 0))
    decisions = pd.DataFrame(decisions)
    save(decisions,"candidate_decisions")
    screening = decisions[decisions.periodic_screen].merge(
        catalog[["family","anchor_segment","anchor_date"]],on="family")
    screening = screening.merge(g,left_on=["family","anchor_segment"],right_on=["family","segment"],how="left")
    save(screening,"periodic_screening")
    overview = []
    for (view,rep), part in decisions.groupby(["view","representation"]):
        main = primary[(primary.view==view)&(primary.representation==rep)]
        ordered = part[part.local].sort_values("anchor_excel")
        disjoint = []
        for excel in ordered.anchor_excel:
            if not disjoint or excel-disjoint[-1]>=4:
                disjoint.append(excel)
        occ = occurrence[(occurrence.view==view)&(occurrence.representation==rep)&(occurrence.condition=="all")]
        overview.append(dict(view=view,representation=rep,anchors=len(part),local_anchors=int(part.local.sum()),
                             disjoint_local_anchors=len(disjoint),local_supported=len(disjoint)>=2,
                             context_anchors=int((part.local & part.context_supported).sum()),
                             context_rate_median=float(part[part.local].context_rate.median()),
                             T_own_pass_anchors=int(part.T_own_above_max_null_p95.sum()),
                             periodic_screen=int(part.periodic_screen.sum()),matches=int(main.matches.sum()),
                             exact_view=int(main.exact_view.sum()),low_change=int(main.low_change.sum()),
                             selected=int(main.selected.sum()),unique_pairs=occ.pair_id.nunique(),
                             T_above_p95=bool(part.T_view_above_p95.iloc[0])))
    overview = pd.DataFrame(overview)
    save(overview,"overview")
    sections = ["# YSH-006B 실험결과 — 국소 모양의 재등장과 간격 구조"]
    def add(title,body):
        sections.extend([f"## {title}",body])
    add("Metadata",f"- 실험: YSH-006B, 담당 YSH, 2026-09-24. EDA / SIGNAL_PATTERN / VALIDATION.\n- 공용 `.venv`, CPU 최대4스레드. 신규 설치·GPU 없음.\n- Raw data 시트만 분석. Result·A 출력 미사용.\n- config SHA-256: `{digest(HERE/'config.json')}`.\n- 상태: 검색·대조·검산·보고서 완료.")
    add("실제 수행 내용", "앞60%의 고정 anchor를 사용한 1행 sliding 검색이다. w4 주 분석, w8/16·burn0/32·위상+1/+2/+3·문턱0.125/0.25/0.50·제외폭2w·controlled100을 비교했다. Absolute/Shape/Delta와 F/I/V/t/FIV/FIVt를 구분했다. 네 행은 검색 길이이며 제품·spot 번호를 부여하지 않았다.\n\n"+table(read("window_audit").groupby(["setting","width"],as_index=False)[["windows","controlled_windows","anchors"]].sum()))
    add("계획에서 변경된 점", "주 판단 기준 변경 없음. 실행 전 부록에 한 요소씩 바꾸는 민감도와 대조의 분모·적용 범위를 고정했다. `engine.py`, `nulls.py`, `postprocess.py`, `order_scores.py`를 모듈로 추가했다. 최대20개 제한은 상세 예시만이며 전체 anchor 성적·맥락 요약을 계산했다. 대조G는 모든 주 anchor의 같은 구간과 대표20개의 다른 구간에 적용했다. 선택적 J는 미실행이며 변수 간 동반 변화의 물리적 의미는 미확정으로 남긴다.")
    add("핵심 결과", f"18개 view/표현 조합 중 비중첩 anchor 2개 이상에서 국소 후보 조건을 만족한 조합은 **{int(overview.local_supported.sum())}개**다. 순서 대조T의 최대 출현률95분위를 초과한 조합은 **{int(overview.T_above_p95.sum())}개**다. 아래 모든 조합을 함께 보고하며 큰 수치만 성공으로 선택하지 않는다.\n\n"+table(overview)+"\n\nselected는 anchor별 선택을 합한 수이며 독립 사건 수가 아니다. unique_pairs는 역방향 같은 쌍을 합쳤으나 서로 다른 pair의 창이 겹칠 수 있다. exact_view·low_change 범주는 중첩될 수 있다.")
    add("기준 스케일과 관측 해상도",table(read("scales"))+"\n\n3월24~31일 8,470행만으로 median/IQR을 고정했다. IQR 0이면 std를 사용한다. q는 관측 고유값 사이 최소 양의 차이(>1e-9)이며 센서 공식 분해능이 아니다. 거리0.25는 IQR 정규화 RMSE이고 물리적 허용오차가 아니다. Shape는 평균만 제거해 진폭 차이를 보존한다. Delta는 원래 순서의 차분이다. 단변수는 양쪽 range>q, 공동 view는 공통 활성 변수가2개 이상이어야 모양으로 센다.")
    add("단변수·공동 view 해석", "Absolute는 값 수준까지, Shape는 평균을 제외한 변화 모양, Delta는 인접 변화량을 비교한다. Absolute의 평평한 일치는 상태 지속으로 별도 집계했다. FIV/FIVt의 거리 분모는 선택한 전체 변수 수로 고정했다. 단변수와 공동 view의 숫자는 같은 사건의 독립 재현 횟수가 아니다.")
    context_table = contexts[(contexts.condition=="all")&(contexts.before==0)].groupby(
        ["view","representation","length"],as_index=False).agg(
            anchors_with_occurrences=("family","size"),median_context_rate=("rate","median"),
            total_valid_pairs=("valid","sum"),total_passed_pairs=("passed","sum"))
    add("전체 후보의 8행·16행 맥락 비교",table(context_table)+"\n\n각 anchor의 유지율 중앙값이다. 상대가 전혀 없었던 anchor는 이 표에 없으며, 유효 확장이0개이면 유지율은 NA다. 같은 상대가 여러 anchor에 포함되므로 쌍 합계를 독립 표본 수로 읽지 않는다. 전류의 완만한 변화가 오래 유지되는 것과 특정 용접 동작이 반복되는 것은 구별해야 한다.")
    mode_counts = wide["mode"].value_counts().rename_axis("gap").reset_index(name="anchor_view_cases")
    add("gap16과 길이16의 차이",table(mode_counts.head(15))+"\n\n넓은 자기 제외폭2w에서도 집계한 같은 구간의 최빈 간격 상위15개다. 전체 gap 분포는 CSV에 남겼다. gap16은 비슷한 네 행 창의 시작점이16행 떨어졌다는 뜻이다. 중간12행까지 같거나16행 전체가 반복된다는 뜻은 아니다. 같은 시작점의16행 맥락 유지율은 위 표에서 별도로 확인한다. gap4는 비중첩 선택이 만드는 최소 간격이기도 하다.")
    add("문턱의 수치적 크기", "주 기준 IQR은 F 0.06bar, I 0.14kA, V 0.007V, t 1ms다. 단변수 거리0.25는 해당 표현에서 F 0.015bar, I 0.035kA, V 0.00175V, t 0.25ms의 RMSE에 해당한다. Shape에서는 평균 제거 후의 오차이므로 원래 평균값 차이가 커도 통과할 수 있다. 짧고 완만한 전류 모양의 높은 매칭률을 특별한 cycle의 증거로 바로 읽으면 안 된다.")
    add("순서 대조 T — 100회",table(t)+"\n\n각 원본 구간의 앞/뒤 영역을 나누고 내부32행 블록을 행 벡터 단위로 섞었다. 원본과 대조 양쪽에서 블록 접합부를 넘는 창을 제외했다. 각 반복에서 동일 anchor 격자 전체를 다시 검색하고 최대 출현률 후보를 다시 선정했다. context_rate는 그 후보의 유효8행 확장 유지율이다. 원본 통계는 안전한 창만 사용하므로 일반 검색의 최대값과 직접 같지 않다. rank는 유효 대조 중 실측보다 작은 비율이며 확증적 p값으로 사용하지 않는다. 이 대조는 느린 변화까지 보존하지 않는다.")
    attempted = g[g.null_valid>0]
    add("간격 대조 G — 출현 수 고정",f"검토한 family×구간 {len(g):,}개 중 유효 대조가 있는 것은 {len(attempted):,}개다. 각 시도는100회이며 비중첩 위치 N개를 만들지 못한 반복은 NA로 보존했다. 모양의 빈도와 허용 위치를 고정하고 무작위 우선순위로 위치를 뽑았다.\n\n"+table(g.groupby("reason",dropna=False).agg(groups=("family","size"),min_valid=("null_valid","min"),max_valid=("null_valid","max")).reset_index())+"\n\n최빈 간격 한 지점이 아니라 전체 간격의 최대 점유율을 비교했다. 모든 family의 분포는 all_gap_distribution_full.csv, G 요약은 null_G_summary.csv에 있다. gap 단위는 원본 행이며 초·제품 수가 아니다. 원본과2w 제외폭의 최빈 간격이 다르면 주기 확정 근거를 낮춘다.")
    add("원본 조건의 간격 탐색 후보 전수",table(screening[["family","anchor_date","context_rate","mode","max_share","peak_p95","null_valid","T_own_rate","controlled_occurrences","controlled_dates"]])+"\n\n11개는 F Delta 6개와 FIV Delta 5개다. 최빈 간격은6행8개,5/12/16행 각각1개다. 이들 G는 모두100회가 유효하다. 하지만10개는 controlled100에서 anchor 자체가 제거되어 통제 후 평가가 불가능하다. 이것을 모양이 사라졌다는 반증으로 해석하지 않는다. FIV_delta_398만 통제 후307개 출현·4개 상대 날짜가 남으며, 이는 모양 출현이 남는다는 뜻이다. controlled100에서 G를 다시 검증한 결과나 독립 공정주기 확정은 아니다. 같은 궤적에서 나온 여러 anchor를 독립 주기 발견 수로 세지 않는다. 이 표는 사후 선별 통계이고 발견 기준으로 고정한 상세20개 목록을 교체하지 않았다.")
    details = catalog[catalog.detailed]
    add("발견 영역에서 고정한 상세 대표 목록",table(details[["family","anchor_excel","anchor_date","discovery","discovery_median","same_back","other_date"]])+"\n\n발견 영역의 비정확·비중첩 출현 수 내림차순, 중앙거리, 원본행 순으로 정했고 anchor끼리 겹치지 않게 했다. 뒤쪽/다른 날짜의 좋은 결과로 교체하지 않았다. 공통 모양 후보를 실제 물리 동작 family로 확정하지 않는다.")
    add("날짜 분리 확인 — 3월 anchor만 사용",table(forward.groupby(["view","representation"],as_index=False).agg(
        anchors=("anchor_excel","size"),occurrences=("temporal_confirmation","sum"),eligible_starts=("temporal_eligible","sum")))+
        "\n\n3월24~31일 anchor에서4월2·3·7일 상대만 검색한 성적이다. 각 anchor의 상대 시작점 분모는3,460개이며 합계에는 anchor별 반복 분모가 포함된다. 4월에 만든 anchor는 이 표와 temporal_confirmation.csv에서 제외했다. 기존 전체 confirmation.csv의4월 anchor에 있는 temporal_confirmation=0은 평가 대상 밖이라는 뜻으로 읽어야 한다. 앞/뒤 구간과 다른 날짜의 분모는 confirmation_denominators.csv에 보존했다. 이미 탐색한 날짜이고 정확 복제가 존재하므로 새 데이터 일반화 검증으로 부르지 않는다.")
    add("대표별 맥락·대조 판정",table(decisions[decisions.family.isin(details.family)])+"\n\nperiodic_screen은 국소 후보·8행 맥락·해당 anchor의 T(전체 최대값 대조)·G·넓은 제외폭의 동일 비최소 간격을 동시에 통과한 탐색 표시다. 다른 날짜의 독립 계보까지 증명한 판정이 아니다. G가 일부 반복만 유효하면 해당 null_valid와 함께 읽어야 한다. 세부 설정의 gap 변화도 아래에 남긴다.")
    raw_values = read("representative_raw_values")
    signatures = []
    for row in details.itertuples():
        values = raw_values[raw_values.family==row.family][[NAMES[j] for j in VIEWS[row.view]]].to_numpy()
        changes = np.any(np.diff(values,axis=0)!=0,axis=1)
        signatures.append(dict(family=row.family,changing_transitions=int(changes.sum()),
                               single_jump=bool(changes.sum()==1),anchor_date=row.anchor_date))
    add("대표의 단순 변화 여부",table(pd.DataFrame(signatures))+"\n\n네 행에는 전이가 세 개뿐이다. 한 번만 값이 바뀌는 대표는 단일 점프라는 대안 설명을 함께 남긴다. 상세 대표가 긴 구간이나 특정 변수에 몰리면 발견 출현 수 기반 순위의 영향일 수 있으므로 전체18조합 표와 함께 읽어야 한다.")
    all_context = read("all_context_summary")
    all_gap = read("all_gap_summary")
    force = read("force_context")
    for row in details.itertuples():
        add(f"대표 {row.family} — 실제 네 행",table(raw_values[raw_values.family==row.family])+"\n\n"+
            table(all_context[all_context.family==row.family][["length","before","total","valid","excluded","passed","rate"]])+"\n\n"+
            table(all_gap[all_gap.family==row.family][["segment","gaps","mode","max_share","multiple4","median","iqr","gap16_share","top3"]])+"\n\n"+
            table(g[g.family==row.family][[c for c in ["segment","n","null_valid","max_share","peak_p95","multiple4","multiple4_p95","reason"] if c in g]]))
    add("가압 조건 — 사후 기술통계",table(force)+"\n\n대표 목록을 고정한 뒤 계산했다. full은 네 행 모두 문턱 초과, partial은 일부만 초과, none은 모두 이하이다. all_windows는 전체 유효 검색 창 분포이고 occurrences는 비중첩 검출 위치다. 모집단과 선택 규칙이 달라 단순 비율 차이를 enrichment 또는 불량 원인으로 부르지 않는다.")
    settings = scores.groupby(["setting","condition","threshold","exclusion","initial_excluded"],as_index=False).agg(
        candidates=("anchor_excel","size"),matches=("matches","sum"),selected=("selected","sum"),
        discovery=("discovery","sum"),back=("same_back","sum"),other_date=("other_date","sum"))
    add("설정 민감도",table(settings)+"\n\n합계는 서로 다른 anchor의 같은 상대를 여러 번 포함한다. 길이8/16은 같은 주 anchor 위치를 유지한 유효 창만 계산했다. burn-in은 anchor 선정에만 적용하며 initial_excluded만 상대 앞100행도 제외한다. 동일 문턱이라도 길이·view에 따라 난이도가 달라 숫자의 대소를 성능 향상으로 해석하지 않는다.")
    control = occurrence[occurrence.condition=="controlled100"]
    add("정확 반복과 긴 복제 통제",f"주 비정확 출현은 선택 view의 원시 exact를 제외했다. controlled100에는 {len(control):,}개의 anchor별 비중첩 출현 기록, {control.target_date.nunique()}개의 상대 날짜가 남았다. 이 수는 독립 반복 횟수가 아니다. 원본11,939행 중 보존2,416행이며 원본 위치의 anchor·구간 경계를 유지했다. 원본창 또는 맥락 일부가 삭제 위치를 지나면 제외했다.\n\n"+table(read("window_audit").query("setting == 'primary'")[["segment","date","windows","controlled_windows","anchors"]])+"\n\nexact_lineage.csv는 대표 출현의100행 이상 exact block 소속을 기록한다. 비정확한 두 모양도 각각 다른 복제본에서 나온 것일 수 있다. 날짜 수가 늘었다는 사실만으로 독립 재현이라 판정하지 않았다.")
    add("Baseline 대비 변화", "기존 lag16을 찾도록 설정을 맞추지 않고 모든 간격을 집계했다. gap16_share는 분포 확정 후 대조할 보조 수치다. 앞선 실험은 다른 후보/거리/창을 사용했으므로 동일 성능 지표의 증가로 비교할 수 없다. 이번 기여는 실제 anchor→비정확 상대→맥락→간격→순서·위치 대조의 대응을 명시한 것이다.")
    add("해석", "네 점 모양은 양자화·정체·단일 변화로도 쉽게 반복될 수 있다. 검색 문턱을 통과한 것만으로 cycle이라 판단하지 않는다. T는 일반적인 짧은 모양과 순서가 있는 구조를 구분하는 근거이고 G는 출현 빈도와 비중첩 규칙으로 생기는 gap 집중을 점검한다. 두 대조의 제약과 긴 복제 계보를 함께 고려해야 한다. 실제 동작은 가압→통전→통전 후 유지→개방이지만 네 행을 이 네 단계에 대응시킨 결과가 아니다.")
    add("가설과 일치 여부",f"B1/B2: 국소 후보를 지원하는 조합은 {int(overview.local_supported.sum())}/18이며 전체 수치·저변화·exact 비중을 위에 제시했다. B3: gap 집중은 G 및2w 표와 비교해 조건부로 판단한다. B4: 날짜 간 출현을 확인했어도 독립 계보는 미확정이다. B5: 가압 관계는 사후 기술통계이고 동작·불량 대응의 입증은 아니다.")
    add("오류/실패", "초기 검색은 발견 영역 경계 밖 창의 억제 영향을 차단하도록 발견 성적을 별도 선택하게 수정한 뒤 재시작했다. 이후 같은 비중첩 규칙의 범위 계산을 캐시해 속도를 개선하고 재실행했다. 입력·후보 격자·거리 문턱은 변경하지 않았다. 연결 중단으로 G 프로세스가 종료되어 저장된 T100회를 재사용하고 G를 같은 seed로 다시 실행했다. G에는250개 시도 구간마다 원자적으로 교체하는 중간 저장을 추가했다. 대조의 미산출은 원래 N을 줄여 숨기지 않았다.")
    add("재현 확인",f"검산 {verified['checks']:,}개 수치 그룹/조건을 통과했다. raw 단위 거리 직접 계산, 날짜/idx 중복 경계, 창/anchor 위치, 모든 출현의 비중첩·exact 제외·문턱, 대표 원본 숫자, 맥락 표본, 대조 반복수·분위수·N 보존을 확인했다. T 분위수는 전수, G 분위수는100개 그룹 표본을 재계산했다. 전체 독립 재실행을 추가로100회 반복한 것은 아니다.\n\n```powershell\nSet-Location F:/Kamp\n.\\.venv\\Scripts\\python.exe -B experiments/YSH/ysh-006b/analyze.py\n.\\.venv\\Scripts\\python.exe -B experiments/YSH/ysh-006b/postprocess.py\n.\\.venv\\Scripts\\python.exe -B experiments/YSH/ysh-006b/nulls.py\n.\\.venv\\Scripts\\python.exe -B experiments/YSH/ysh-006b/verify.py\n.\\.venv\\Scripts\\python.exe -B experiments/YSH/ysh-006b/report.py\n```\n\nseed42·100회, config와manifest에 입력/코드 해시를 남겼다. outputs CSV를 동일 설정으로 갱신하는 재현 명령이다.")
    add("결론", "비정확한 국소 모양은18/18조합에서 발견됐고,14/18조합은 순서를 섞은 대조의 최대 출현률95분위를 넘었다. 전류 모양은 긴 맥락에서 잘 유지되는 편이나 전압의 짧은 모양은 대부분 긴 맥락에서 약해졌다. 원본 조건에서 간격 탐색 후보11개가 남았지만10개는 긴 복제 통제 후 anchor가 없어 평가 불가이고, 나머지1개도 독립 주기 검증을 마친 것은 아니다. 따라서 반복 신호의 존재는 후속 분석할 근거가 있지만 공통4행/16행 공정주기, 제품·spot·동작 경계와 초 단위 cycle은 확정하지 않는다.")
    add("채택 / 보류 / 폐기", "채택: 원본 위치를 유지하는 비정확 모양·맥락·gap 진단. 보류: 주기 후보의 독립 재현 및 실제 동작 대응. 비채택: 정확 복제나 짧은 최근접 일치만으로4행 제품·lag16 공정주기를 확정하는 해석.")
    add("다음 실험 또는 다음 행동", "대표 창을 실제 장비 이벤트·전류 파형·spot/제품 로그와 대조하고 새 생산일에서 고정 검색 규칙을 확인한다. A와의 결과 결합은 별도 후속 계획으로 둔다. YSH-007은 생성하지 않았다.")
    (HERE/"실험결과.md").write_text("\n\n".join(sections)+"\n",encoding="utf-8")
    print(overview.to_string(index=False))


if __name__=="__main__":
    main()


