# 성산 BC와 주행 데이터 수집 v1

성산 4배 교통에서 학습된 BC가 직접 행동하는 폐루프 수집기다. 맑음 500만, 옅은 안개 250만, 짙은 안개 250만 **실제 전이**를 목표로 한다. 저장소에는 학습용 교사 라벨, 모델, 평가 결과와 작은 완결 에피소드 샘플을 포함한다. 1,000만 전이는 향후 본 수집 목표다.

## 실행

Python 3.13.9, libsumo/SUMO 1.27.1에서 확인했다. 아래 명령의 작업 폴더는 `final_project/seongsan_bc`다. 학습과 추론에는 PyTorch, Parquet 도구에는 PyArrow가 필요하다. CPU로 추론할 수 있으며 CUDA는 학습에 선택적으로 사용한다.

```sh
python -m pip install -r requirements.txt
python collect.py --weather light --start-seed 83000000 --episodes 1 --out outputs/light-demo
python parquet_io.py outputs/light-demo/83000000/transitions.csv outputs/light-demo/83000000.parquet
python evaluate.py --start-seed 83000000 --episodes 100 --workers 8 --out paired-evaluation.jsonl
python distill_driver.py student_retrained collect_optimized --seed 25 --epochs 35
```

수집 시 `--weather clear|light|dense`와 새 출력 폴더를 지정한다. `--workers`로 독립 에피소드 수를 병렬화한다. 평가와 수집은 같은 `collect()`와 [날씨 설정](configs/collection.json)을 사용한다. 평가 스크립트는 시드별 결과를 남기고 임시 전이 파일을 정리한다. 본 수집은 평가·학습 시드와 겹치지 않는 범위로 배치를 나누고 실제 저장 전이를 누적한다. 배치마다 에피소드 번호가 다시 시작하므로 파일 밖에서는 날씨·배치·시드까지 포함해 식별한다. 데이터 분할은 원본 시드 기준으로 한다.

## 모범 운전자와 학습 데이터

[learned_driver.py](learned_driver.py)의 `structured_action`이 계획 차로, 전후방 간격·상대속도, 선행차, 신호·정지 의무와 교차 충돌 예측에 따라 가감속과 차로 이동을 선택한다. 최적화된 교사 설정은 [parameter_policy.json](driver_study/parameter_policy.json), 평가에 잠근 값과 모델 해시는 [v2_selection.json](driver_study/v2_selection.json)에 있다. 일곱 값은 순서대로 순항 속도 7.312m/s, 감속 2.925m/s², 옆 차로 앞·뒤 기본 간격 8.230m·8.657m, 상대속도에 적용하는 여유 시간 1.076초, 교차 충돌 시간 여유 1.014초, 교차 지점 정지 여유 3.463m다. 실행 시 JSON의 원래 정밀도를 쓴다.

[collect_optimized](driver_study/collect_optimized)는 4배 교통 120회, 시드 54100000–54100119의 24개 학습 묶음이다. `labels.npz` 안에 입력 x, 교사 행동 y, 입력 이름 names가 있고 총 113,270개 라벨이다. 각 묶음의 rows.json은 시드별 주행 결과다. 안개 +5m는 학생 정책의 후속 평가·데이터 수집 설정이다.

다음 PowerShell 명령으로 최적화 교사 시연을 별도 이름으로 재생성할 수 있다.

```powershell
$params = (Get-Content driver_study/parameter_policy.json -Raw | ConvertFrom-Json).parameters
python learned_driver.py teacher collect_regenerated 54100000 120 --traffic 4 --params $params
```

## BC 정책

배포 가중치 [student_opt_bc.pt](driver_study/student_opt_bc.pt)는 TorchScript 형식이다. SHA-256은 `e5de39ff305184d70fb7bd9d2527e79dce5e6901c3603fe1327b28b6cf218321`이다. 수집 시 검증한다.

17개 입력의 순서는 [스키마](SCHEMA.md)에 있다. [Student.forward](distill_driver.py)는 속도, 간격, 제동 거리, 임계값 지시자 등을 계산해 27개 특징으로 변환한다. 가감속 헤드는 27→96→64→3, 횡방향 헤드는 27→48→32→3이며 은닉층은 Tanh다. 두 헤드의 로짓을 더해 9가지 조합의 점수를 만들고 최댓값을 택한다. 행동 번호 i의 가감속은 [-4,0,3][i//3] m/s², 차로 방향은 [-1,0,1][i%3] (우측·유지·좌측)이다.

학습은 Adam, 초기 학습률 0.001, 배치 512, 두 분류 헤드의 가중 교차 엔트로피 합을 사용한다. 클래스 가중치는 라벨 빈도의 -0.35승이고 epoch 70%, 90% 지점에서 학습률에 0.3을 곱한다. 위 재학습 명령은 코드 기본값인 seed 25, 35 epochs를 명시한 재현용 설정이다. 원 체크포인트 생성 명령 전체를 입증하는 로그는 확인되지 않았다. 정확한 배포 정책 비교에는 제공한 가중치를 사용한다.

## 환경과 수집 조건

성산 전체 경로, 기존 교통의 4배, 0.2초 간격, speedMode 96과 laneChangeMode 0을 사용한다. 동봉한 도로와 경로 XML은 기반 브랜치의 `new map/generated` 파일과 SHA-256이 일치한다. 기준 브랜치는 `seongsan-models-20261123`, 커밋은 `938453fcb3c853811927d23c247b71162cc8b703`이다.

| 날씨 | 목표 전이 | 차량/신호/정지선의 50% 감지 거리(m) | 위치 오차 표준편차(m) |
|---|---:|---|---:|
| 맑음 | 5,000,000 | 기존 시야 규칙 | 0 |
| 옅은 안개 +5m | 2,500,000 | 40 / 45 / 29 | 0.6 |
| 짙은 안개 +5m | 2,500,000 | 20 / 33 / 17 | 1.4 |

합성 안개에서 거리 d의 감지 확률은 1/(1+exp((d−d50)/5))다. 차량 관측과 신호 판독에는 기존 시야 후보 제한이 먼저 적용된다. 홀수 시드 시야 30m, 짝수 50m다. 정지 의무 지점은 기존 BC의 정지 접근 후보에 마스킹·거리 오차를 적용한다. 지도 기반 제어 지점 거리도 기존 정책 입력에 남아 있다. 센서 난수 시드는 시뮬레이션 시드+100000000이고 대상별 감지 잠재변수 상관계수는 0.85다. 이 값들은 성산용 합성 실험 설정이다.

## 성능과 소형 데이터

| 같은 시드 100개 | 완주 | 충돌 | 정지 의무 위반 |
|---|---:|---:|---:|
| 맑음 | 98 | 2 | 0 |
| 옅은 안개 +5m | 98 | 0 | 2 |
| 짙은 안개 +5m | 58 | 26 | 16 |

시드 범위는 83000000–83000099다. [안개 비교](evaluation/plus5_comparison.csv), [원래 독립 평가](driver_study/v2_holdout.json)와 에피소드별 결과를 포함했다. 기존 독립 평가의 4배 교통에서는 500회 중 완주 489회, 충돌 10회, 시간 초과 1회였다. 짙은 안개에서는 충돌률 5% 목표를 넘는다. 전체 수집 데이터에는 실패 주행도 포함한다.

[samples](samples)에는 시드 83000000의 맑음 1,022개, 옅은 안개 1,037개, 짙은 안개 1,003개 전이가 들어 있다. 세 주행 모두 완주했다. 샘플은 에피소드 단위 Parquet이며 다음 상태 복원에는 `parquet_io.read_episode(path)`를 사용한다.
