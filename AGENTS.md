# 프로젝트 작업 규칙

이 파일은 저장소 전체에 적용합니다. `experiments/`와 `src/`에서는 해당 디렉터리의 AGENTS.md도 함께 따릅니다.

- 공식 자료로 확정되지 않은 대회 목표, 데이터 구조, 지표, 모델링 방법을 추측하지 않습니다. 불명확한 내용은 TODO로 남깁니다.
- 원본 데이터, 비밀값, 인증 파일, 대용량 산출물을 commit하지 않습니다. stage 전에 대상 경로와 diff를 확인합니다.
- `main`을 직접 수정하거나 commit/push하지 않습니다. 작업 브랜치에서 변경하고 PR로 통합합니다.
- `experiments/NBJ/`, `experiments/YSJ/`, `experiments/YSH/`는 각 개인의 공간입니다. 담당자의 작업과 기존 변경을 존중하고, 요청 범위 밖의 개인 코드를 임의 수정·이동·삭제하지 않습니다.
- `git reset --hard`, `git clean -fd/-fdx`, 강제 push, 브랜치 강제 삭제, 이력 재작성 등의 위험한 Git 명령을 임의 실행하지 않습니다. 명시적 요청과 영향 범위 확인 없이 수행하지 않습니다.
- 사용자 변경을 덮어쓰거나 관계없는 파일을 정리하지 않습니다. 작업 전 Git 상태와 적용되는 규칙을 확인합니다.
- 실험은 질문/가설 중심으로 관리하고 유형은 metadata로 기록합니다. 검증된 재사용 코드만 `src/`로 승격합니다.
- 현재는 초기 구조 준비 단계입니다. 실제 실험 요청 전에는 EXP 폴더, 임의의 데이터 예시, 모델 구현, 불필요한 의존성을 추가하지 않습니다.
- 변경 목적에 맞는 확인을 수행하고, 실행하지 않은 실험·검증을 완료했다고 기록하지 않습니다.

TODO: 대회 규정, 실행 환경, 공용 코드의 검증 명령이 확정되면 관련 문서와 함께 갱신합니다.


## Python 실행 환경

- Python 의존성은 루트 `requirements.in`에서 관리하고 `uv pip compile`로 `requirements.txt`를 갱신한다.
- 모든 실험은 프로젝트 루트의 공용 `.venv`를 사용한다. 초기 설정·의존성 변경 시에만 `setup-environment.ps1`로 동기화한다.
- 실험마다 `.deps`, 별도 가상환경, `pip install --target` 설치를 만들지 않는다. 개인 Python이나 Codex 런타임을 우회 실행 환경으로 사용하지 않는다.
- 새 패키지가 필요하면 공용 requirements에 추가하고 기존 실험과의 호환성을 확인한다. 일반 실험 실행 시 자동 설치하지 않는다.
- 실행 예: 루트에서 `.\.venv\Scripts\python.exe experiments/YSH/ysh-005/analyze_structure.py`. 자세한 설정은 `docs/environment.md` 참조.
