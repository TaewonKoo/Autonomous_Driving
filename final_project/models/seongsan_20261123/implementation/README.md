# 실제 계산 구현 스냅샷

`paper_training/observed_driving.py`의 `observation`, `decode`, `step`이 실제 입력·행동·보상 계산 근거입니다.
`paper_training/agents.py`에서는 `Policy`와 `OnPolicy`의 PPO 분기만 이 세 모델에 사용했습니다.
다른 알고리즘이나 `numeric_correction=True`·`failure_time_tail=True` 분기는 이 모델 설정에 해당하지 않습니다.

핵심 6개 파일은 동결한 탐색 소스와 해시를 대조했습니다. trainer에는 기록된 파일 교체 재시도 수정이 포함돼 있습니다.
`environments.py`와 패키지 초기화 파일은 현재 import 의존성으로 추가했으며, 원래의 6파일 해시 목록에는 없었습니다.
지도 두 파일은 1000회 평가 때 동결한 해시와 일치합니다. 실행 시 환경은 이 원본 route에서 배경 flow와 자차 설정을 다시 생성합니다.
현재 디렉터리는 기존 모듈의 소스 배치를 보존합니다. 실제 병렬 평가는 상위 폴더의 `evaluate_models.py`와 `EVALUATION.md`를 사용하세요.

가중치·프로파일만 저장됐으며 Adam 내부 상태는 모델에 없습니다. 후보의 출발 가중치 파일은 추가 업로드하지 않았고 상위 설정에 SHA256을 기록했습니다.
동일 시드라도 병렬 수집 순서 때문에 비트 단위의 동일 재학습을 보장하지 않습니다.
