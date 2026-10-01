# 공용 Python 환경

모든 실험은 `F:/Kamp/.venv` 하나를 사용한다. 실험별 `.deps`나 다른 Python 설치 경로를 추가하지 않는다.

## 최초 설정 및 의존성 동기화

전제: uv와 Python 3.13이 설치되어 있어야 한다. 스크립트는 새로운 Python을 자동 다운로드하지 않는다.

```powershell
Set-Location F:/Kamp
.\setup-environment.ps1
```

`requirements.in`은 직접 사용하는 패키지와 버전의 목록이고, `requirements.txt`는 uv로 생성한 하위 의존성과 배포 파일 해시까지 포함한 설치 목록이다. 이번 잠금 파일은 Windows x64 / Python 3.13 / CPU 환경에서 검증한다. PyTorch는 CPU 저장소를 사용한다.

## 실험 실행

설치는 실험마다 반복하지 않는다. 가상환경 활성화 없이 해당 Python으로 바로 실행할 수 있다.

```powershell
Set-Location F:/Kamp
.\.venv\Scripts\python.exe experiments/YSH/ysh-005/analyze_structure.py
.\.venv\Scripts\python.exe experiments/YSH/ysh-005/model_candidates.py
```

위 명령은 실제 실험을 다시 실행하고 해당 출력 파일을 갱신한다. 환경을 확인할 때에는 `uv pip check --python .venv/Scripts/python.exe`를 사용한다.

## 패키지를 추가하거나 버전을 바꿀 때

루트 `requirements.in`을 수정한 다음 한 번만 실행한다.

```powershell
.\setup-environment.ps1 -UpdateLock
```

직접 실행하려면 아래 순서를 사용한다.

```powershell
uv pip compile requirements.in --python .venv/Scripts/python.exe --torch-backend cpu --generate-hashes --output-file requirements.txt
uv pip sync requirements.txt --python .venv/Scripts/python.exe --torch-backend cpu --require-hashes --strict
```

일반 분석 스크립트는 패키지를 자동 설치하지 않는다. 의존성이 추가되거나 변경되면 공용 환경을 동기화하고 필요한 호환성 검사를 수행한다. uv 캐시는 유지하며 실험별로 패키지 사본을 만들지 않는다.

## YSH-007 시각 검토 notebook

[review_ysh007.ipynb](../experiments/YSH/ysh-007/review_ysh007.ipynb)은 기존 CSV 산출물과 Raw4 원본 값만 읽는다. 모델 학습·설정 선택은 실행하지 않는다. VS Code 등의 Jupyter notebook 화면에서 프로젝트 .venv Python 커널을 선택해 실행한다. 공용 requirements에 ipykernel·nbformat·plotly가 포함되어 있다. 산출물 outputs는 Git에서 제외되므로 이후 같은 실행을 다시 검토하려면 로컬 CSV를 보존해야 한다.
## 기존 실험의 환경 이력

- YSH-004/005의 핵심 수치·모델 라이브러리는 기존 실행 버전을 유지했다.
- YSH-001/002는 과거 분석에 pandas 3.0.1 / numpy 2.3.5를 사용했고, 일부 시각화와 YSH-004/005는 pandas 2.2.3 / numpy 2.2.4를 사용했다. 현재 공용 환경은 후자의 버전으로 통일했다. 과거 보고서와 manifest의 실행 환경 기록은 당시 이력이다.
- 기존 실험의 requirements 파일은 공용 requirements를 참조한다. YSH-001 실행 스크립트도 공용 Python을 기본값으로 사용한다.
- YSH-004/005의 `.deps` 경로 주입은 제거했다. 공용 환경 검사 후 기존 설치 폴더 두 개도 제거했다.
- 기존 실험 결과·모델·원본·당시 manifest는 보존한다. 환경 이전의 확인 내용은 `docs/environment_validation.json`에 별도로 기록하며, 실험 전체 재실행과 구분한다.


## 이번 환경 이전의 확인 범위

- 33개 설치 패키지의 의존성 일관성 검사 통과.
- YSH-001~005 분석 모듈을 공용 Python에서 불러오는 검사 통과.
- 원본 Excel 읽기·해시·구간 경계, YSH-005 Ridge 오차, 저장된 GMM/IF 예측 및 CPU AE 재구성 결과가 기존 기록과 일치.
- ruptures는 실제 공정값으로 실행 확인. matplotlib는 메모리 렌더, Plotly는 내장 JavaScript 로딩 확인.
- 전체 실험 재학습은 하지 않았다. 기존 표·모델·보고서·실행 manifest를 덮어쓰지 않았다.

다시 검사하려면 프로젝트 루트에서 다음을 실행한다. 결과는 `docs/environment_validation.json`에 저장된다.

```powershell
.\.venv\Scripts\python.exe -B check-environment.py
```
