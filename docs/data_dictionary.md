# 데이터 사전

# KAMP 용접기 AI 데이터셋 — 공식 가이드 기반 도메인 및 데이터 컨텍스트

## 0. 문서 목적

이 문서는 KAMP `용접기 AI 데이터셋` 및 이를 기반으로 한 경진대회 데이터를 분석하는 AI Agent에게
공정의 실제 의미, 데이터 수집 구조, 변수의 물리적 의미, 품질정보의 한계,
원 KAMP 분석방법과 현재까지 확인된 주의사항을 전달하기 위한 컨텍스트 문서이다.

이 문서에서 가장 중요한 원칙은 다음과 같다.

1. 본 데이터는 공식적으로 **SPOT 용접(저항 점용접)** 데이터이다.
2. `Item No`는 개별 physical product ID가 아니라 **생산품목/품번**이다.
3. Raw row별 양품/불량 label은 제공되지 않는다.
4. 품질정보는 **일별/Lot 수준의 불량 개수 및 불량유형** 형태로 제공된다.
5. 따라서 날짜별 불량정보를 Raw row 각각의 0/1 label로 사용해서는 안 된다.
6. 원 KAMP 분석도 이 문제 때문에 비지도 AutoEncoder anomaly detection을 사용하였다.
7. 데이터에는 공정주기·4지점 수집구조·반복 block 등 숨은 구조가 있을 가능성이 있으므로,
   row를 iid sample로 바로 취급하기 전에 데이터 생성과정을 이해해야 한다.
8. 가이드북 자체에도 행 개수, 전처리 후 개수 등 일부 내부 불일치가 있으므로
   실제 제공 파일의 값이 최종적인 분석 근거가 되어야 한다.

---

# 1. 공식 데이터셋 정보

데이터셋명:

`용접기 AI 데이터셋`

KAMP 등록일:

`2020-12-14`

업종:

`뿌리산업 / 용접접합`

목적:

`품질보증`

공식 분석 알고리즘:

`Anomaly Detection - AutoEncoder`

제공기관:

- KAIST
- 수행기관: 울산과학기술원(UNIST)
- 수행기관: ㈜이피엠솔루션즈

공식 활용 목적은 SPOT 용접에서 발생하는 품질문제를 분석하고,
불량 원인 및 주요 공정변수 간 관계를 파악하며,
생산조건 최적화 및 불량 예측 모델을 개발하는 것이다.

공식 출처 표기:

> 중소벤처기업부, Korea AI Manufacturing Platform(KAMP),
> 용접기 AI 데이터셋,
> KAIST(울산과학기술원, ㈜이피엠솔루션즈),
> 2020.12.14.,
> www.kamp-ai.kr

---

# 2. 어떤 제품을 용접한 데이터인가

제조 분야는 **자동차 부품 용접**이다.

가이드북은 현장을 `자동차 부품 D사 공장`으로 설명한다.

중요:

현재 공개 가이드에는 이 자동차 부품이 정확히 무엇인지,
예를 들어 차체 패널인지 브래킷인지 특정 부품명까지는 기록되어 있지 않다.

현재 경진대회 Raw Data의 생산품목 번호는:

`Item No = 65235-25800`

이다.

공식 데이터 정의에서 `Item No`는:

`생산품목`

으로 정의된다.

따라서 다음 해석을 사용한다.

`65235-25800 = 특정 생산품목/품번`

이며,

`65235-25800 = 실제 제품 instance 한 개의 고유 ID`

로 해석해서는 안 된다.

즉 실제 현장에서는 동일한 `65235-25800` 품목을 반복 생산했다고 보는 것이 자연스럽다.

다만 공개 데이터에는 각 physical product를 구분할 별도의
`product_instance_id`가 제공되지 않는다.

정확한 제품명, 차량 모델, 부품 위치 등은 공식 가이드에서 확인되지 않았다.

---

# 3. SPOT 용접 공정

본 데이터는 공식적으로 **Spot Welding / Resistance Spot Welding** 공정이다.

가이드북 설명:

용접하려는 재료를 두 전극(TIP) 사이에 두고 가압한 상태에서 전류를 통과시키면
접촉부의 전기저항에 의해 열이 발생하고,
이 저항열을 이용하여 재료를 접합한다.

주로 자동차 차체 및 자동차 부품 제조에 사용되는 용접 방식이다.

가이드북에서는 Spot 용접의 주요 4대 요소를 다음과 같이 설명한다.

- 용접전류
- 통전시간
- 가압력
- 전극

---

# 4. 저항 점용접의 기본 물리

저항용접의 발열은 기본적으로 다음 관계를 갖는다.

Q ∝ I² × R × t

여기서:

- Q = 발생 열량
- I = 용접전류
- R = 접촉부 및 재료의 전기저항
- t = 통전시간

가이드북에서도 특히 전류의 제곱에 발열량이 비례하므로
전류가 용접 결과에 중요한 변수라고 설명한다.

즉 Current의 작은 변화도 발열량에는 상대적으로 크게 작용할 수 있다.

예:

I가 1% 증가할 경우,
다른 조건이 동일하다면 I² 항은 약 2% 증가한다.

따라서 Current는 단순 선형변수로만 해석하면 안 된다.

---

# 5. Current의 도메인 의미

`weld current(kA)`는 용접 지점에서 측정된 전류이다.

공식 수집범위:

`12 ~ 18 kA`

현재 제공 데이터에서는 대부분 약:

`14.5 ~ 15.1 kA`

근처에 위치한다.

Current의 도메인적 역할:

- 전류가 증가하면 Joule heating이 크게 증가한다.
- 판재가 두꺼울수록 일반적으로 더 큰 전류가 필요하다.
- 전류가 너무 낮으면 충분한 nugget 형성이 어려울 수 있다.
- 전류가 지나치게 높으면 과도한 발열, expulsion, indentation 등의 문제가 발생할 수 있다.

따라서:

`Current ↑ = 항상 좋음`

또는

`Current ↓ = 항상 불량`

으로 해석해서는 안 된다.

Current는 Force, Time, Material, Electrode 상태와 함께 봐야 한다.

---

# 6. Weld Time의 도메인 의미

`weld time(ms)`는 전극에 용접전류가 실제로 흐르는 통전시간이다.

공식 수집범위:

`30 ~ 120 ms`

현재 Raw Data에서는:

`70 ~ 73 ms`

가 대부분이며 중앙값은 약 72 ms이다.

가이드북에는 각 용접 지점에서 약:

`0.072 sec`

동안 데이터를 수집한다고 명시되어 있다.

0.072 sec = 72 ms

이므로 현재 데이터의 `weld time ≈ 72 ms`와 직접적으로 일치한다.

중요:

72 ms는 전체 생산 cycle 시간이 아니다.

72 ms는 전류가 흐르는 용접 통전시간이다.

전체 공정에는:

- 가압
- 통전
- 유지
- 정지
- 작업 이동

등이 포함된다.

---

# 7. Force의 도메인 의미

컬럼명:

`weld force(bar)`

공식 설명:

`용접 지점에서 가해지는 압력`

공식 수집범위:

`1 ~ 12 bar`

주의:

bar는 엄밀히 force의 단위가 아니라 pressure의 단위이다.

따라서 이 컬럼을 Newton 단위의 electrode force와 동일하게 해석해서는 안 된다.

실제 force를 계산하려면 actuator의 유효면적 등이 필요하다.

가이드북은 Force의 물리적 영향을 다음과 같이 설명한다.

가압력이 증가하면 접촉저항이 작아져 유효발열량이 감소할 수 있다.

개념적으로:

Force ↑
→ 접촉면적 ↑
→ Contact Resistance ↓
→ I²Rt에 의한 유효발열 감소 가능

반대로 Force가 지나치게 낮으면 접촉저항 분포가 불균일해지고
스파크 등이 발생할 수 있다.

따라서 Force에도 적정 영역이 존재하며,

`Force ↑ = 무조건 좋은 조건`

`Force ↓ = 무조건 나쁜 조건`

이라는 단조 관계를 가정하면 안 된다.

---

# 8. Voltage의 도메인 의미

`weld Voltage(v)`는 용접 지점에서 측정된 전압이다.

공식 수집범위:

`1.5 ~ 3.5 V`

현재 데이터에서는 약:

`2.46 ~ 2.86 V`

범위가 주로 관찰된다.

저항용접에서 기본적으로:

V = I × R

관계가 존재하므로,

Current가 제어되는 상황에서 Voltage의 변화는
접촉저항 또는 공정 상태의 변화를 반영할 가능성이 있다.

단,

현재 데이터는 한 용접 cycle 내부의 waveform `V(t)`를 제공하지 않는다.

따라서 단일 행의:

R_proxy = V / I

는 실제 시간에 따른 `Dynamic Resistance Curve`가 아니라

`effective resistance proxy`

정도로만 해석해야 한다.

---

# 9. 전극(Electrode)

가이드북은 Spot welding의 4대 요소 중 하나로 전극을 명시한다.

전극의 접촉면적은 전류밀도와 연관되어 용접품질에 영향을 준다.

또한 전극의 냉각상태는:

- 용접품질
- 전극마모

에 영향을 줄 수 있다고 설명한다.

하지만 현재 데이터에는:

- electrode ID
- electrode wear
- electrode temperature
- electrode tip geometry
- cooling condition

등이 존재하지 않는다.

따라서 전극 상태는 중요한 **숨은 변수(hidden state)**일 가능성이 있다.

---

# 10. 실제 용접 과정

가이드북의 Spot 용접과정은 다음과 같다.

1. 타이머 전원 스위치 ON
2. 급수 및 에어밸브 개방
3. 용접 전원 ON
4. 건(Gun) 동작 확인
5. 초기 Nugget 시험
6. 설비조건 조정
7. 양호상태에서 연속작업
8. 작업완료 후 전원/급수/에어 종료

한 번의 용접 cycle 자체는 개념적으로:

초기 가압
→ 통전
→ 유지
→ 정지
→ 다음 작업

구조이다.

가압단계에서는 판재를 전극 사이에서 밀착한다.

통전단계에서는 실제 전류를 흘려 발열시킨다.

유지단계에서는 전류가 종료된 뒤에도 가압을 유지하여
용접부를 응고시킨다.

그 후 다음 작업까지 정지시간이 존재한다.

---

# 11. 현장 자동화 수준

중요:

현재 KAMP 원데이터를 수집한 현장은
완전한 로봇 자동용접 공정이 아니었다.

가이드북에서는:

`로봇 기반 용접 자동화 공정이 아닌 작업자 기반 용접공정`

이라고 명시한다.

즉 작업자가 Spot welding 작업을 수행하고 있었다.

다만 공정조건 데이터는 PLC와 MES를 통해 수집하였다.

수집된 정보:

- 전류
- 가압력
- 통전시간
- 기타 공정정보

판재 두께 관련 정보는 ERP-MES 작업오더와 연계되었다.

품질검사는 별도 검사 인력이 수행하였다.

따라서 현재 데이터의 반복패턴을
무조건 `robot teaching program`으로 설명해서는 안 된다.

---

# 12. 품질검사 방식

현장에서는 작업 후 품질을 검사하였다.

가이드북에 따르면:

- 양품/불량품 판정
- 불량품 등급
- 불량유형

등을 검사자가 별도로 기록하였다.

검사는 기본적으로 사람의 육안검사 의존도가 높았다.

가이드북에서 문제로 언급된 대표적인 용접불량:

- 과도한 용접 파임
- 판의 들뜸
- 용접 불균일
- 용접 Crack

현재 경진대회 Result 데이터에서는 대표적으로:

- defect type 1 = 파임불량
- defect type 2 = 용접부족
- defect type 3 = 크랙발생

이 기록되어 있다.

---

# 13. 데이터 수집 시스템

제조 분야:

`자동차 부품 용접`

공정:

`Spot 용접`

수집 시스템:

- Spot 용접 PLC
- MES
- MongoDB

수집 기간:

`2020-03-24 ~ 2020-04-07`

약 15일이다.

가이드북의 공식 설명:

`약 8초 주기로 4지점에서 센서 데이터를 수집`

하며,

`각 지점별 약 0.072초 동안 데이터를 수집`

한다.

이는 현재 데이터를 이해하는 데 매우 중요한 metadata이다.

---

# 14. "약 8초 / 4지점 / 0.072초"의 의미

공식 가이드에는 다음과 같이 명시되어 있다.

- 약 8초 주기로 4지점에서 센서 데이터를 수집
- 각 지점별 약 0.072초 동안 데이터를 수집

중요:

`0.072초`는 새로운 row가 생성되는 sampling interval이라고 해석하면 안 된다.

즉:

8 / 0.072 ≈ 111

을 계산하여

`약 111 rows = 한 제품`

이라고 해석하는 것은 근거가 없다.

0.072초는 각 용접지점에서의 측정/통전 구간이며,
현재 Raw Data의 대표적인 weld time 약 72 ms와 직접 대응한다.

따라서 보다 자연스러운 개념은:

약 8초의 작업/수집 cycle
→ 여러 작업단계 포함
→ 그 안에서 4개 지점의 용접 수행
→ 각 지점의 실제 통전/측정구간 약 72 ms

이다.

다만 공식 문서에서는 다음 mapping을 명확하게 제공하지 않는다.

- 1 Raw row = 1 지점인지
- 4 Raw rows = 1 cycle인지
- 4 지점 = 한 physical product의 4개 welding spot인지
- 8초 cycle = 제품 한 개의 생산시간인지

따라서 이를 데이터에서 별도로 검증해야 한다.

---

# 15. 4지점 정보와 주기 분석

공식 metadata에서 `4지점`이 확인되었기 때문에
향후 sequence EDA에서는 다음 lag를 우선적으로 살펴볼 수 있다.

L = 4, 8, 12, 16, 20, ...

특히 현재 데이터에서 Voltage lag-16 후보가 관찰된 적이 있으므로:

16 = 4 × 4

라는 관계를 탐색할 수 있다.

하지만:

`lag 16 signal → 제품당 16 spot`

으로 바로 결론내려서는 안 된다.

4지점이라는 공식 공정구조와 데이터에서 발견된 lag signal이
실제로 어떻게 연결되는지를 별도 검증해야 한다.

---

# 16. Raw 데이터 파일

공식 1차 가공 데이터:

`Welding Data Set_01.xlsx`

주요 컬럼:

- idx
- Machine_Name
- Item No
- working time
- Thickness 1(mm)
- Thickness 2(mm)
- weld force(bar)
- weld current(kA)
- weld Voltage(v)
- weld time(ms)

현재 실제 파일 기준 Row 수:

`11,939`

현재 실제 데이터에서:

Machine_Name = Spot-01

Item No = 65235-25800

Thickness 1 = 0.7 mm

Thickness 2 = 0.7 mm

으로 사실상 고정되어 있다.

따라서 실제로 변화하는 핵심 공정변수는:

- Force
- Current
- Voltage
- Weld Time

이다.

---

# 17. idx의 의미

가이드북 데이터 정의:

`idx = 생산순번`

따라서 idx는 최소한 production/event order에 해당한다.

다만 정확한 timestamp는 제공되지 않는다.

working time은 현재 날짜까지만 존재한다.

따라서:

x_t - x_(t-1)

과 같은 event-order difference는 분석할 수 있지만,

dx/dt

와 같은 실제 시간당 변화율은 계산할 수 없다.

idx는:

`event order / pseudo-time`

으로 사용한다.

---

# 18. Item No의 의미

가이드북 정의:

`Item No = 생산품목`

따라서 Item No는 개별 생산품의 unique ID가 아니다.

현재 모든 row에서:

`65235-25800`

이 반복된다.

따라서 가장 자연스러운 해석은:

동일한 자동차 부품 품번 `65235-25800`을
여러 번 반복 생산한 공정 데이터이다.

하지만 실제 개별 제품 instance를 구분하는 ID는 공개되지 않았다.

---

# 19. Working Time

`working time = 작업시간`

으로 정의되어 있다.

현재 데이터에서는 날짜 단위 값만 존재한다.

예:

2020-03-24
2020-03-25
...

따라서:

- 일별 생산량
- 일별 공정조건 분포
- 일별 불량집계

는 비교할 수 있다.

그러나 하루 내부의 정확한 시:분:초 정보는 사용할 수 없다.

---

# 20. 공식 공정변수 수집범위

가이드북 공식 수집범위:

| 변수 | 공식 범위 |
|---|---:|
| Thickness 1 | 0.3 ~ 2.3 mm |
| Thickness 2 | 0.3 ~ 2.3 mm |
| Weld Force | 1 ~ 12 bar |
| Weld Current | 12 ~ 18 kA |
| Weld Voltage | 1.5 ~ 3.5 V |
| Weld Time | 30 ~ 120 ms |

현재 경진대회 파일에서는 두께가 모두 0.7 mm로 고정되어 있다.

---

# 21. 개별 Label은 없다

가이드북은 종속변수를:

`제품 각각에 대한 불량 선별값`

이라고 설명하지만,

공개 데이터에서는 해당 값이:

`표시 없음 / Unlabeled`

상태라고 명시한다.

따라서 Raw Data의 각 row에 대해:

- 양품인지
- 불량인지
- 어떤 불량유형인지

알 수 없다.

---

# 22. Result 데이터의 의미

가이드북은 다음을 명확히 설명한다.

`개별 물품에 대한 양품 혹은 불량 여부는 알 수 없지만,
일별 불량 수 및 불량 유형 정보는 알 수 있다.`

또한 분석요약에서는 품질정보 수집방법을:

`생산 Lot 단위 불량 이력 데이터`

라고 설명한다.

따라서 Result 데이터는:

row-level label

이 아니라:

날짜 / Lot 수준 aggregate quality information

이다.

따라서 다음 행위는 금지한다.

- defect가 7개인 날짜의 Raw row 1000개 중 임의 7개를 불량으로 지정
- anomaly score 상위 7개를 true defect라고 가정
- 해당 날짜 전체 1000개를 불량으로 지정
- 날짜별 defect count를 각 행의 binary label처럼 복제

---

# 23. 현재 Result 데이터의 불량유형

현재 제공 데이터에서:

Type 1 = 파임불량

Type 2 = 용접부족

Type 3 = 크랙발생

형태이다.

Result에 특정 날짜/유형 row가 존재하지 않는 경우:

`0건`

이라고 자동 가정해서는 안 된다.

예를 들어 어떤 날짜에는 명시적인 `defect=0` row가 존재하므로,

`row 없음`

과

`명시적인 0`

은 의미가 다를 가능성이 있다.

Missing과 Zero를 분리해서 관리한다.

---

# 24. 원 KAMP 데이터의 Missing / Outlier 문제

가이드북 기술통계에는 공정변수에 Missing이 존재한다.

기술통계상 count:

- weld force: 11,921
- weld current: 11,924
- weld Voltage: 11,924
- weld time: 11,916

즉 11,939보다 일부 작은 count가 기록되어 있다.

가이드북 품질지수:

Completeness ≈ 99.94%

Validity ≈ 97.93%

로 제시한다.

---

# 25. 가이드북에 존재하는 극단값

가이드의 기술통계에서는 최대값이 다음처럼 기록되어 있다.

- Force max = 120
- Current max = 154.3
- Voltage max = 98.24
- Weld Time max = 2240

이는 공식 정상 수집범위를 크게 벗어난다.

예:

Force 공식범위 1~12 bar
vs 기록된 max 120

Current 공식범위 12~18 kA
vs 기록된 max 154.3

Voltage 공식범위 1.5~3.5 V
vs 기록된 max 98.24

Time 공식범위 30~120 ms
vs 기록된 max 2240

따라서 원본 데이터에는 명백한 입력오류 또는 비정상값이 존재했던 것으로 보인다.

현재 경진대회 제공 파일에서는 이러한 극단값이 대부분 제거된 것으로 보이므로,
현재 파일은 원 KAMP 데이터의 어느 정도 정제된 버전일 가능성이 있다.

하지만 이를 확정하려면 원본 버전과 현재 파일을 직접 비교해야 한다.

---

# 26. 원 KAMP의 데이터 품질 전처리

가이드북에서는:

- Missing 확인
- Validity 확인
- Outlier 검출
- Normalization

등을 수행한다.

전처리 결과로:

`총 11,692개의 데이터`

와

`258개의 Outlier (약 2.16%)`

가 기록되어 있다.

주의:

11,939 - 11,692 = 247이며,
문서의 `258 outlier`와 산술적으로 정확히 일치하지 않는다.

따라서 이 숫자는 문서 내부 불일치로 기록해 둔다.

---

# 27. Raw Row 수에 대한 문서 내부 불일치

분석요약 페이지에는:

`Row 수 = 23,901`

이라고 기록되어 있다.

그러나 실제 실습 코드와 현재 제공 파일에서는:

`11,939 rows`

가 확인된다.

따라서 공식 가이드 자체에:

`23,901 vs 11,939`

라는 불일치가 있다.

실제 분석에서는 현재 제공 파일의 11,939행을 기준으로 사용한다.

---

# 28. 날짜별 Raw row 개수

실습 출력과 현재 파일은 다음 날짜별 row 구조를 가진다.

2020-03-24 : 1200
2020-03-25 : 1352
2020-03-26 : 1000
2020-03-27 : 1648
2020-03-30 : 1470
2020-03-31 : 1800
2020-04-02 : 2000
2020-04-03 : 800
2020-04-07 : 669

합계:

11,939

이 숫자는 날짜별 분석과 train/test boundary를 재구성하는 데 중요하다.

# 29. 공식 Scaled Data 생성방식

KAMP 공식 2차 가공 데이터는:

`scaled_data.csv`

이다.

공식 가이드에서는 다음 4개 공정변수만 남긴다.

- weld force
- weld current
- weld Voltage
- weld time

그리고 sklearn의 `MinMaxScaler`를 사용하여 최소-최대 정규화를 수행한다.

MinMaxScaler 식:

z = (x - x_min) / (x_max - x_min)

가이드북 기술통계에서 확인되는 원 데이터의 min/max는 다음과 같다.

| Feature | Min | Max | Range |
|---|---:|---:|---:|
| weld force | 1.74 | 120 | 118.26 |
| weld current | 14.52 | 154.3 | 139.78 |
| weld Voltage | 2.46 | 98.24 | 95.78 |
| weld time | 70 | 2240 | 2170 |

따라서 공식 scaling은 사실상 다음과 같이 복원된다.

Force_scaled
= (Force - 1.74) / 118.26

Current_scaled
= (Current - 14.52) / 139.78

Voltage_scaled
= (Voltage - 2.46) / 95.78

Time_scaled
= (Time - 70) / 2170

이 값들은 기존 프로젝트에서 scaled_data를 역변환하여 경험적으로 추정한
offset/range와 정확히 일치한다.

따라서 scaling 공식 자체는 공식 가이드에 의해 강하게 확인되었다.

---

# 30. 현재 Raw와 scaled_data의 계보 문제

현재 경진대회 Raw Data는 공식 가이드의 기술통계와 다르다.

공식 원 데이터 기술통계에서는:

Force max = 120
Current max = 154.3
Voltage max = 98.24
Time max = 2240

와 같은 극단값이 존재하였다.

반면 현재 프로젝트에서 사용하는 Raw Data는
대체로 정상 공정범위 안에 있으며 위와 같은 극단값이 존재하지 않는다.

따라서 다음 가능성을 고려한다.

1. scaled_data가 극단값이 포함된 이전 Raw snapshot에서 생성됨
2. 현재 Raw는 이후 정제된 버전임
3. Raw와 scaled 사이에서 행 순서 재배열이 발생함
4. 일부 row가 제거/교체됨
5. 서로 다른 dataset version에서 생성됨

현재까지 확인된 사실:

- MinMax scaling 공식은 확인 가능
- 그러나 현재 Raw row 번호와 scaled_data row 번호는 1:1 대응하지 않음
- 따라서 scaled_data를 현재 Raw의 직접적인 변환본으로 가정하면 안 됨

현재 정책:

모델링에서는 제공 scaled_data를 그대로 사용하지 않고,

Raw
→ Train/Validation Split
→ scaler.fit(Train)
→ transform(Train/Validation)

방식을 우선한다.

제공 scaled_data는 모델 입력보다
dataset provenance / lineage 검증용 자료로 사용한다.

# 31. 원 KAMP 모델 — 비지도 AutoEncoder

가이드북은 개별 row의 양불 label이 없기 때문에
비지도 AutoEncoder anomaly detection을 사용한다.

논리:

Raw row label 없음
→ supervised classification 불가능
→ 정상/일반적인 process pattern 학습
→ reconstruction error가 큰 sample을 anomaly로 판정

중요:

AutoEncoder의 anomaly는:

`Defect Probability`

가 아니다.

정확한 표현은:

`Process Anomaly Score`

또는:

`Reconstruction-based Anomaly Score`

이다.

---

# 32. 원 AutoEncoder 구조

가이드 코드 기준 구조:

Input dimension = 4

Encoder:

4
→ 3
→ 2

Decoder:

2
→ 3
→ 4

즉 bottleneck dimension은 2이다.

활성함수:

RReLU

출력 layer에는 별도 activation을 사용하지 않는다.

손실함수:

MSELoss

Optimizer:

Adam

---

# 33. 원 AutoEncoder 학습 설정

가이드 코드의 하이퍼파라미터:

epoch = 50

batch_size = 64

learning_rate = 0.01

hidden_size = [3]

bottleneck/output_size = 2

loss = MSE

optimizer = Adam

훈련 과정에서는 DataLoader를 사용하며:

shuffle = True

로 batch order를 섞는다.

---

# 34. 원 KAMP Train/Test 분할

가이드에서는 데이터 순서를 이용해 단순 slicing한다.

Train:

8,470 rows

Test:

3,469 rows

합계:

11,939 rows

가이드가 명시한 불량개수:

Train aggregate defects = 28

Test aggregate defects = 11

중요한 재구성:

날짜별 row 수를 누적하면:

03/24 1200
+ 03/25 1352
+ 03/26 1000
+ 03/27 1648
+ 03/30 1470
+ 03/31 1800
= 8470

따라서 원 가이드 Train은 사실상:

2020-03-24 ~ 2020-03-31

Test는:

2020-04-02
2020-04-03
2020-04-07

의 시간순 holdout과 정확히 일치한다.

즉 row-random split이 아니라 chronological slicing이다.

---

# 35. 원 KAMP의 Aggregate defect count

Train에 기록된 `28 defects`,
Test에 기록된 `11 defects`는
개별 Raw row label 수가 아니다.

Result sheet의 날짜별 defect count를 합한 aggregate count에 해당한다.

따라서 모델이 어떤 정확한 Raw row를 불량으로 맞혔는지는 확인할 수 없다.

---

# 36. 원 AutoEncoder Threshold

가이드 코드의 anomaly threshold:

threshold
=
mean(train reconstruction loss)
+
8 × std(train reconstruction loss)

가이드에서 계산된 threshold:

약 0.08386612

이다.

즉 train reconstruction error의 평균보다 상당히 큰 영역만 anomaly로 판단하는
매우 보수적인 threshold를 사용하였다.

이 `8 sigma` 값은 데이터로 최적화한 이론적 threshold라기보다
가이드에서 사람이 설정한 heuristic에 가깝다.

---

# 37. 원 KAMP Test 결과

Test sample:

3,469

Result sheet aggregate defect count:

11

AutoEncoder anomaly count:

13

가이드북은 이를:

`일별 불량품 개수 관점에서 약 2건 차이`

라고 설명한다.

하지만 개별 row label이 없으므로:

Precision
Recall
F1-score
Accuracy

를 계산할 수 없다.

가이드북 스스로도:

`13개의 anomaly가 실제 11개 defect와 동일한 행인지 알 수 없기 때문에
개별제품 불량판별 성능은 평가할 수 없다.`

고 명시한다.

---

# 38. 원 KAMP 모델 평가의 핵심 한계

AutoEncoder가:

11 real defects
vs
13 predicted anomalies

를 만들었다고 해서:

11개 중 몇 개를 맞췄는지는 알 수 없다.

예를 들어 실제 defect row와 predicted anomaly row가 전혀 겹치지 않아도
개수만 비슷할 수 있다.

따라서 원 KAMP 결과는:

`Defect Classification Performance`

가 아니라:

`Aggregate anomaly-count consistency`

정도로만 해석해야 한다.

---

# 39. 현재 경진대회 평가와의 차이

현재 경진대회는 모델 비교에 F1-score 등을 요구한다.

하지만 원 KAMP 공개 데이터에는 row-level label이 존재하지 않는다.

따라서 현재 공개 데이터만 사용한다면
정상적인 supervised F1 계산이 원칙적으로 불가능하다.

향후 반드시 확인해야 할 것:

- 경진대회에서 별도 row-level target이 추가 제공되는가
- 평가 단계에서 hidden label이 존재하는가
- 제공 Result를 어떻게 정답으로 연결하라는 공식 지침이 있는가

이 문제가 해결되지 않은 상태에서 pseudo-label을 만들고 F1을 계산해서는 안 된다.

---

# 40. 원 KAMP가 인정한 데이터 한계

가이드북은 현재 Feature가 적기 때문에
딥러닝/기계학습의 장점을 충분히 살리기 어렵다고 직접 언급한다.

현재 핵심 Feature는 4개뿐이다.

- Force
- Current
- Voltage
- Time

따라서 더 많은 공정특성이 존재하면
보다 효과적인 분석이 가능하다고 설명한다.

이 부분은 향후 Feature Engineering 및 Physics Feature의 필요성을 뒷받침한다.

---

# 41. 물리 기반 Feature 후보

현재 공정 의미가 충분히 확인된 뒤에는
한 row 내부에서 다음 물리기반 Feature를 검토할 수 있다.

Power Proxy:

P_proxy = V × I

Effective Resistance Proxy:

R_proxy = V / I

Energy Proxy:

E_proxy = V × I × t

Current-Time Heating Proxy:

H_proxy = I² × t

단:

V와 I가 실제 동일 시간구간의 representative measurement라는 가정이 필요하다.

또한:

I² × (V/I) × t
=
V × I × t

이므로 algebraically equivalent한 Feature를
서로 독립적인 새로운 정보처럼 다루면 안 된다.

---

# 42. Force interaction Feature

저항용접의 물리상 Force는
Current/Voltage/Time과 interaction을 갖는다.

따라서 단순 Force 하나뿐 아니라:

Force × Current

Force × Time

Energy / Force

I²t / Force

등을 탐색할 수 있다.

하지만 이들 중 일부는 정확한 물리법칙이 아니라
engineering proxy이므로
결과보고서에서는 반드시 `proxy`라고 표현한다.

---

# 43. 순서 기반 Process-State Feature

현재 idx가 생산순번이므로 다음과 같은 과거 정보 Feature를 검토할 수 있다.

Δx_t = x_t - x_(t-1)

rolling median

rolling mean

rolling std

rolling IQR

EWMA

current value - recent baseline

rolling slope

change-point distance

state duration

regime ID

단:

현재 실제 timestamp가 없기 때문에
시간당 변화율이 아니라
`event-order 변화`로 해석한다.

---

# 44. Physics Feature와 State Feature의 차이

Physics Feature:

한 번의 용접 이벤트 내부의 상태를 재표현

예:

V×I
V/I
V×I×t
I²t

State Feature:

여러 생산 이벤트 사이의 공정상태를 재표현

예:

rolling median
rolling deviation
trend
regime duration

두 Feature block은 의미가 다르므로
향후 ablation에서도 분리해서 검증한다.

---

# 45. 현재 데이터의 반복 구조

현재 프로젝트 EDA에서는
서로 다른 날짜 사이에 매우 긴 exact process sequence가 확인되었다.

예:

- 약 1,639 rows
- 약 1,345 rows
- 약 1,184 rows
- 약 1,000 rows

동안 Force / Current / Voltage / Time이 동일한 연속구간이 존재한다.

이 현상은 단순 이상치가 아니라
데이터 생성구조 또는 공정 recipe/state와 관련될 가능성이 있다.

하지만 현재 단계에서:

`데이터 복사`

라고 확정해서도 안 되고,

`실제 생산 cycle`

이라고 확정해서도 안 된다.

---

# 46. 300개 High-Force block

현재 데이터에서 Force 주요 분포인 약 2.x bar와 다른
약 7.8~8.0 bar 수준의 별도 Force regime이 존재한다.

특히:

03/25
03/27
03/31
04/03

에서 주요 Force 영역 밖 데이터가 정확히 300개씩 확인되었다.

이는 random outlier보다는
특정 process block / recipe / state일 가능성을 우선 고려해야 한다.

300은:

300 / 4 = 75

로 공식적인 `4지점` 구조와 나누어떨어지기 때문에
후속 EDA에서 4-point acquisition structure와 관계를 확인할 가치가 있다.

하지만 현재로서는 연결이 확인된 것은 아니다.

---

# 47. Force와 불량의 관계

Force는 도메인적으로 중요한 변수이지만
Force 하나만으로 defect rate를 설명할 수 있다는 증거는 없다.

특히:

- 높은 Force가 존재하지 않는 날짜에도 높은 defect count가 발생
- 높은 Force regime이 있는 날짜라고 해서 항상 defect rate가 높지 않음

따라서:

Defect ≠ f(Force only)

이다.

보다 자연스러운 구조:

Defect
=
f(
Force,
Current,
Voltage,
Time,
interactions,
hidden equipment/material state
)

이다.

---

# 48. Current 후보

일부 날짜, 특히 2020-03-26에서는
Current 평균 및 중앙값이 다른 날짜보다 상대적으로 낮게 나타났다.

하지만:

`낮은 Current → defect`

라는 인과관계는 아직 검증되지 않았다.

날짜별:

- Current distribution
- low-current proportion
- defect rate
- defect type

을 함께 비교해야 한다.

---

# 49. Defect type을 분리해야 하는 이유

파임불량, 용접부족, 크랙은 동일한 failure mechanism을 갖는다고 가정할 수 없다.

예를 들어:

Force / Current / Time의 변화가

- 파임
- insufficient weld
- crack

에 서로 다른 방향으로 영향을 줄 수 있다.

따라서 총 defect count만 분석하지 말고
가능한 경우 defect type별 aggregate rate도 별도로 분석한다.

---

# 50. 정상영역이라는 표현을 조심할 것

데이터에서 가장 자주 등장하는 영역을:

`normal quality region`

이라고 부르면 안 된다.

올바른 표현:

- dominant process region
- major density region
- frequent process regime

현재 row-level quality label이 없으므로:

`빈도가 높다 = 양품`

이라는 관계는 증명되지 않았다.

---

# 51. Outlier와 Defect는 다른 개념

Anomaly:

데이터 분포에서 희귀한 공정상태

Defect:

실제 제품 품질문제

따라서:

Anomaly ≠ Defect

일 수 있다.

정상적인 recipe switch가 anomaly로 잡힐 수 있고,
반대로 일반적인 공정영역 안에서도 defect가 발생할 수 있다.

---

# 52. 4지점 주기 EDA 권고

가이드북의 공식 `4지점` metadata를 활용하여
우선 다음 lag 구조를 점검한다.

4
8
12
16
20
24
...

각 변수:

Force
Current
Voltage
Time

에 대해:

- raw autocorrelation
- first-difference autocorrelation
- exact match rate
- phase profile
- modulo-position distribution
- cycle-to-cycle correlation

을 본다.

하지만:

`idx mod 4`

를 곧바로 Feature에 넣지 않는다.

먼저 실제 반복성이 검증되어야 한다.

---

# 53. Voltage lag-16 후보

현재 EDA에서는 일부 구간에서 Voltage가:

lag 16
lag 32
lag 48

등에서 반복구조 후보를 보였다.

특히 first difference에서도 일부 signal이 유지되었다.

공식 metadata에서 4-point acquisition이 확인되었으므로:

16 = 4 × 4

관계를 후속 가설로 검증할 수 있다.

하지만 아직:

- 하나의 제품이 16 spot을 가진다
- 16 row가 한 제품이다

라고 결론내릴 근거는 없다.

---

# 54. Product Boundary는 여전히 미확정

Item No는 품목번호이지만
physical product instance ID가 없다.

따라서 다음을 아직 모른다.

- 한 제품이 몇 row인가
- 한 제품이 몇 Spot인가
- 제품 boundary가 어디인가
- 4지점이 한 제품 내부 4개의 Spot인가
- idx가 제품마다 reset되는지 여부의 의미
- Result defect count가 정확히 어떤 physical item을 대상으로 하는지

이 문제는 현재 데이터의 핵심 미확정 사항이다.

---

# 55. Working-time 기반 Validation 주의

현재 날짜는 9개뿐이다.

또 여러 날짜 사이에 긴 exact sequence가 존재한다.

따라서:

`날짜가 다르다 = 독립된 생산조건`

이라고 바로 가정하면 안 된다.

날짜 split을 사용할 경우에도
train/validation 사이의 exact 또는 near-exact sequence overlap을 확인해야 한다.

---

# 56. Original KAMP baseline은 참고용이지 정답이 아니다

원 KAMP AutoEncoder는 공식 reference baseline으로 의미가 있다.

그러나 다음 한계가 있다.

- Feature 4개
- row-level label 없음
- threshold heuristic
- anomaly count와 aggregate defect count만 비교
- 개별 F1 계산 불가
- 데이터 반복구조 고려 없음
- 4-point sampling structure를 모델에 명시적으로 활용하지 않음
- process physics interaction을 Feature로 사용하지 않음

따라서 현재 경진대회에서는
원 baseline을 그대로 재현한 뒤 개선하는 접근이 적절하다.

---

# 57. 권장 실험 순서

Phase 1 — Data Understanding

- Official metadata 확인
- 4-point structure 분석
- exact repeated block 분석
- Force regimes
- Current regimes
- missing / outlier / duplicate
- scaled provenance
- product boundary 가능성

Phase 2 — Validation Design

- temporal holdout
- repeat-aware split
- purged split
- sequence overlap audit

Phase 3 — Simple Baseline

- Isolation Forest
- Robust distance
- LOF
- Original AutoEncoder reproduction

Phase 4 — Physics Features

- VI
- V/I
- VIt
- I²t

Phase 5 — State Features

- rolling median
- deviation
- volatility
- trend
- regime/change point

Phase 6 — Quality Relationship

- daily anomaly burden
- daily defect count
- defect-type-specific relationship

Phase 7 — Final Competition Model

경진대회가 별도 row-level target을 제공할 경우:

- supervised baseline
- tree boosting model
- alternative model
- ensemble/calibration
- FN / FP analysis

를 수행한다.

---

# 58. AI 분석 시 금지사항

다음 해석은 증거 없이는 하지 않는다.

`Item No = 개별 제품 ID`

`1 row = 1 physical product`

`1 row = 반드시 1 spot`

`4 rows = 반드시 1 product`

`16 rows = 반드시 1 product`

`높은 Force = 불량`

`낮은 Current = 불량`

`빈도가 높은 영역 = 정상 품질`

`AutoEncoder anomaly = defect probability`

`날짜가 다르면 완전히 독립`

`Result row 없음 = defect 0`

`긴 exact sequence = 무조건 데이터 복사`

---

# 59. AI 분석 시 권장 표현

확정된 사실:

`가이드북에서 명시`

데이터에서 직접 확인:

`Raw Data에서 관찰`

가능하지만 확인되지 않음:

`가설`

물리적으로 가능한 해석:

`도메인 가설`

모델 출력:

`Anomaly Score`

품질과 검증된 경우에만:

`Defect Risk`

라고 표현한다.

---

# 60. 문서 내부 불일치 / Provenance Issues

현재 공식 가이드에도 다음 불일치가 존재한다.

1. 분석요약 Row 수:
   23,901

   실제 실습 및 현재 파일:
   11,939

2. 전처리 결과:
   11,692 rows
   258 outliers

   그러나:
   11,939 - 11,692 = 247

3. 전처리 후 11,692라고 설명하지만
   AutoEncoder train/test split은:
   8,470 + 3,469 = 11,939

   즉 모델 실습에서 실제로 어떤 버전을 사용했는지 설명이 완전히 일치하지 않는다.

4. 공식 정상범위보다 매우 큰 extreme max가 기술통계에 존재한다.

이러한 이유로:

`가이드북 설명`
+
`실제 파일 검증`

을 항상 같이 수행한다.

---

# 61. 원 분석환경

가이드북의 원 분석환경:

OS:
Ubuntu 14 이상

CPU:
Intel Xeon 2.3 GHz

RAM:
13 GB

GPU:
Tesla K80

주요 도구:

- Python
- Anaconda
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- scikit-learn
- PyTorch

현재 분석에서는 동일 환경을 재현할 필요는 없으며
모델 구조 및 preprocessing logic만 재현하면 된다.

---

# 62. 자동차 Spot Welding 관련 추가 도메인 맥락

가이드북은 Spot welding이 자동차 부품산업에서 매우 널리 사용된다고 설명한다.

가이드북 예시:

승용차 한 대 조립 시 약 3,400~3,800개의 용접 타점이 존재할 수 있으며,
이 중 Spot welding의 비중이 매우 높다고 설명한다.

자동화된 현장에서는:

- robot teaching coordinate
- PLC current
- weld time
- pressure
- inspection result

등을 서로 매핑할 수 있기 때문에
AI 분석에 더 적합하다고 설명한다.

현재 원 데이터 현장은 작업자 기반이기 때문에
위치정보 및 개별 검사결과 매핑이 충분하지 않았던 것이
현재 데이터셋의 핵심 한계이다.

---

# 63. 경진대회 관점의 핵심 해석

현재 경진대회 문제를 단순한 4-feature tabular 문제로 보지 않는다.

실제 문제는:

Observed Process Data
=
Force
+ Current
+ Voltage
+ Time

에서

숨겨진:

- product instance
- spot position
- electrode state
- material condition
- process regime
- quality outcome

을 얼마나 잘 설명할 수 있는가에 가깝다.

따라서 모델 complexity보다 먼저
데이터 생성구조와 관측단위를 이해해야 한다.

---

# 64. 가장 중요한 현재 연구질문

Q1.
가이드의 `4지점` 구조가 Raw sequence에서 어떻게 나타나는가?

Q2.
하나의 physical product는 몇 개의 Raw row를 가지는가?

Q3.
Voltage lag-16은 4-point acquisition과 관계가 있는가?

Q4.
300-row high-force block은 어떤 공정상태인가?

Q5.
Force × Current × Voltage × Time interaction은 defect type과 어떤 관계가 있는가?

Q6.
긴 exact repeated sequence는 공정 recipe 반복인가,
데이터 생성/전처리 결과인가?

Q7.
개별 row label이 없는 상황에서
aggregate defect information을 어떻게 검증에 활용할 것인가?

Q8.
경진대회에서 F1을 계산할 수 있는 hidden/추가 label이 존재하는가?

---

# 65. 최종 요약

이 데이터는:

`자동차 부품 D사의 작업자 기반 Resistance Spot Welding 공정`

에서 얻어진 데이터이다.

생산품목:

`Item No = 65235-25800`

이며 정확한 부품명은 공개 가이드에서 확인되지 않는다.

공정 데이터는:

- Pressure-like Weld Force
- Current
- Voltage
- Weld Time

으로 구성된다.

공식 수집구조는:

`약 8초 주기`
+
`4지점 센서 수집`
+
`각 지점 약 0.072초`

이다.

72 ms는 현재 데이터의 대표적인 Weld Time과 정확히 대응한다.

개별 제품의 양품/불량 label은 공개되지 않았으며
일별/Lot 수준 defect count 및 defect type만 존재한다.

따라서 원 KAMP에서는
4-feature AutoEncoder anomaly detection을 사용하였다.

하지만 현재 분석에서는:

공정구조
→ 반복구조
→ 4-point structure
→ validation
→ baseline
→ physics/state feature
→ quality relationship

순으로 분석하는 것이 적절하다.

가장 중요한 원칙:

`Anomaly ≠ Defect`

`Frequent ≠ Good`

`Different Date ≠ Independent`

`Item No ≠ Product Instance`

`Observed 4 variables ≠ Complete physical state`

이다.
