# 5tonomous (6조) | 성산 저시정 자율주행

**최종 정책은 규칙 기반 교사의 주행을 모방해 학습한 BC 모델 `student_opt_bc`입니다.** 이 모델이 제주 성산 SUMO 도로에서 날씨별 관측을 받아 직접 주행했고, 그 결과 11,837개 에피소드와 10,001,226개 전이를 수집했습니다. PPO(W39)는 개발 중 시도한 강화학습 정책으로, 학습 코드와 평가 결과를 아래 추가 실험에 남겼습니다.

[GitHub 저장소](https://github.com/TaewonKoo/Autonomous_Driving) · [전체 데이터셋 고정 버전](https://huggingface.co/datasets/taewonkoo/seongsan-driving-transitions/tree/67b6c4ab50f142677300d8dd0d9a59739e8c9dd5) · [BC 가중치](bc/driver_study/student_opt_bc.pt)

## BC로 한 번 주행하기

BC 기준 환경은 Windows, Python 3.13.9, SUMO 1.27.1입니다. 저장소 루트에서 아래 명령을 실행하면 맑음 조건으로 한 번 주행하고, `bc/outputs/clear-demo/83000000/transitions.csv`와 에피소드 결과를 생성합니다. 다시 실행할 때는 새 출력 폴더 이름을 지정하세요.

```sh
cd bc
python -m pip install -r requirements.txt
python collect.py --weather clear --start-seed 83000000 --episodes 1 --out outputs/clear-demo
```

같은 `bc/` 폴더에서 다음 명령으로 세 날씨를 평가하거나, 포함된 교사 라벨로 학생 정책을 다시 학습할 수 있습니다.

```sh
python evaluate.py --start-seed 83000000 --episodes 1 --workers 1 --out weather-eval.jsonl
python distill_driver.py student_retrained collect_optimized --seed 25 --epochs 35
```

[교사 설정·시연 재생성·모델 구조·재학습 안내](bc/README.md)

## 저장소 구조

```text
bc/
  learned_driver.py, distill_driver.py   교사 행동 생성과 BC 학습
  driver_study/                         최종 가중치, 교사 라벨, 교차로 정보
  implementation/, configs/             수집 맵(v3), 환경, 날씨 설정
  evaluate.py, evaluation/              BC 평가 코드와 결과
  collect.py, parquet_io.py             주행 수집과 Parquet 변환
  samples/, SCHEMA.md                   날씨별 샘플과 칼럼 정의
data/README.md                          전체 데이터 이용 안내
experiments/paired_braking/             동일 장면 제동 실험, 집계, SUMO 영상
models/final_ppo/, paper_training/      추가 실험 PPO 모델과 환경 코드
maps/seongsan/, evaluate.py             최종 맵(v4)과 루트 PPO 평가 코드
```

## BC 학습과 평가

교사 시연 120회에서 얻은 113,270개 라벨로 가감속과 차로변경을 학습했습니다. 학생 정책은 관측 17개에서 특징 27개를 계산하고, 0.2초마다 가감속(−4, 0, +3 m/s²)과 차로 방향(우측·유지·좌측)을 조합한 9개 행동 중 하나를 선택합니다.

| 수집 맵 v3 · 교통량 4배 · 날씨별 같은 시드 100개 | 완주 | 충돌 | 정지 의무 위반 |
|---|---:|---:|---:|
| 맑음 | 98 | 2 | 0 |
| 옅은 안개 | 98 | 0 | 2 |
| 짙은 안개 | 58 | 26 | 16 |

짙은 안개에서는 충돌률 5% 목표를 달성하지 못했습니다. 전체 공개 데이터에는 성공과 실패 주행이 모두 포함됩니다. [평가 자료](bc/evaluation/plus5_comparison.csv)

## 데이터 구조

HF의 `data/clear/`, `data/light_fog/`, `data/dense_fog/`에는 날씨별 Parquet가 있습니다. 루트의 `episodes.parquet`에서 에피소드별 파일과 row group을 찾을 수 있습니다. 각 행은 자차의 0.2초 주행 전이이며, 28개 칼럼으로 구성됩니다.

| 한 행에서 읽는 정보 | 대표 칼럼과 의미 |
|---|---|
| 시점과 조건 | `episode`, `t`, `weather_id`, `obs_profile_id` |
| 정책이 본 정보 | `observation`은 실제 BC 입력 17개. `detected_vehicles`는 감지된 주변차의 ID·상대 위치·속력 목록 |
| 실제 교통 상태 | `state`는 해당 시점의 전체 차량과 신호 목록. `signal_exists_map`과 관측 신호색 `red_obs/yellow_obs/green_obs`를 별도로 기록 |
| 행동과 결과 | `action_raw`는 [요청 가감속, 차로 방향]. 실제 차로변경 여부, 환경이 계산한 `reward`, 충돌·위반·종료 여부도 기록 |
| 다음 시점 | `next_observation`, `next_state`. Parquet 중간 행의 중복 값은 같은 에피소드의 다음 행에서 복원 |

[28개 칼럼 전체 정의와 입력 순서](bc/SCHEMA.md) · [날씨별 샘플](bc/samples/) · [데이터 이용 안내](data/README.md)

| 날씨 | 에피소드 | transition | 감지 거리 (차량/신호/정지선, 50% 확률) |
|---|---:|---:|---|
| 맑음 | 5,327 | 5,000,539 | 기존 시야 규칙 |
| 옅은 안개 | 2,705 | 2,500,663 | 40 / 45 / 29 m |
| 짙은 안개 | 3,805 | 2,500,024 | 20 / 33 / 17 m |

## 도로와 교통

- 성산일출봉 주변 외곽 루프. 회전교차로, 신호 교차로, 비신호 교차로, 합류부, 전방향 정지(STOP)가 있습니다.
- 모든 도로에 차로를 하나씩 더하고 제한속도를 60 km/h로 맞춘 맵입니다. 도로 우선순위는 차로를 늘리기 전 차로 수를 기준으로 정합니다.
- 일반 차량 경로 42개(출발지 7 × 목적지 6). 기본 교통량은 합계 700대/시간이며, "교통량 N배"는 이 값에 N을 곱한 것입니다.
- AV는 고정 출발점에서 전체 경로를 한 바퀴 돌면 완주입니다. 충돌, 적색 신호 위반, STOP 미정지, 경로 이탈, 시간 초과는 실패입니다.
- SUMO의 자동 안전 기능은 끕니다(`speedMode 96`, `laneChangeMode 0`). 가감속과 차로 변경은 모두 정책이 정합니다.

## 저시정 제동 실험

동일한 주행 상태에서 같은 BC 모델에 맑음·안개 관측을 각각 입력했습니다. 맑음에서 제동한 7,086개 시점 중 옅은 안개에서는 927개(13.1%), 짙은 안개에서는 2,474개(34.9%)에서 제동을 선택하지 않았습니다. 짙은 안개에서 선행차를 놓치고 제동이 사라진 691개 시점 중, 앞차 간격과 속도를 복원하자 528개(76.4%)에서 제동이 돌아왔습니다. 실제 궤적에는 맑음 행동만 적용했습니다.

[재현 코드와 집계](experiments/paired_braking/) · [SUMO-GUI 영상](experiments/paired_braking/sumo_gui_perception.mp4) · [대표 장면](experiments/paired_braking/sumo_gui_perception_keyframe.png)

## 추가 실험 — PPO (W39)

### 정책 비교

| | PPO | BC |
|---|---|---|
| 관측 | 63개 (자차·주변 차량, 교차로 상충 차량, 경로, 신호 색, STOP, 0.8초 전 값 일부) | 17개 (속도, 앞차·옆 차로 간격, 신호·정지선 거리, 충돌 예상 지점 거리 등) |
| 행동 | 9개 = 가감속 {−4, 0, +3} m/s² × 차로 {오른쪽, 유지, 왼쪽}, 0.2초마다 | 같음 |
| 보상 | 충돌 −30, 그 밖의 실패 −10, 도착 +10, 교차로 통과 +5, 목표 속도 추종, 차로 변경·위험 차로 변경 페널티 | 학습에 쓰지 않음 (교사 행동 모방). 데이터에는 환경 보상을 기록 |

주행 중에는 정책 하나만 행동을 정합니다. 규칙 기반 보조나 시나리오 전환은 없습니다.

### 성능 (맑은 날, 최종 맵, 1000 에피소드, seed 920000–920999)

| 교통량 | PPO 충돌 | PPO 완주 | PPO 속도 | BC 충돌 | BC 완주 | BC 속도 |
|---|---|---|---|---|---|---|
| 1배 | 0.9% | 98.8% | 33.0 km/h | 0.1% | 99.9% | 22.6 km/h |
| 1.5배 | 4.1% | 94.8% | 33.4 km/h | 0.4% | 99.6% | 22.3 km/h |
| 2배 | 3.9% | 95.7% | 31.6 km/h | 0.6% | 99.4% | 22.4 km/h |
| 4배 | 19.2% | 76.4% | 28.4 km/h | 2.1% | 97.7% | 21.6 km/h |

- 속도는 완주한 에피소드의 평균입니다.
- PPO는 교통량 2~3배까지 학습했고 4배는 학습 범위 밖입니다. BC는 4배 교통으로 만든 시연을 학습했습니다.
- BC를 자신의 데이터 수집 맵(v3)에서 평가하면 충돌 0.3% / 0.8% / 0.4% / 1.4%입니다(같은 순서).

PPO 기준 환경은 Python 3.11, SUMO 1.27.1입니다. 아래 명령은 저장소 루트에서 실행합니다.

```bash
python -m pip install -r requirements.txt
# PPO: 화면 없이 평가 (결과는 results/eval_<교통량>x.jsonl)
python evaluate.py --traffic 2 --episodes 1000 --seed 920000

# PPO: sumo-gui로 주행 보기
python view_policy_gui.py --checkpoint models/final_ppo/model.pt --map pluslane_v4 --traffic 2 --speed 4

# PPO: 데이터 수집 (수업 collect_pomdp_data.py 형식, data/<name>/)
python collect_seongsan_data.py --traffic 1,1.5,2 --transitions 10000000 --name ppo_w39

# 맵 다시 만들기 (maps/seongsan/generated 와 같은 도로망이 생성됨)
python -c "from maps.seongsan.builder import build; build()"
```

[학습 설정과 명령](models/final_ppo/training_config.json)
