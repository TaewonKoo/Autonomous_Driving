# 저시정 환경을 고려한 성산 자율주행 (6조)

제주 성산일출봉 일대 도로를 SUMO로 구현하고, 일반 차량이 많은 교통 속에서 자율주행 차량(AV)이 한 바퀴를 완주하도록 학습한 프로젝트입니다. 두 가지 정책을 만들었습니다.

- **PPO**: 강화학습 정책(W39). 빠르게(31~33 km/h) 주행합니다.
- **BC**: 규칙 기반 교사 운전자를 모방한 정책(`student_opt_bc`). 느리지만(약 22 km/h) 교통량 4배에서도 안정적입니다. Hugging Face 데이터셋은 이 정책으로 수집했습니다.

| 링크 | |
|---|---|
| 데이터셋 (Hugging Face) | https://huggingface.co/datasets/taewonkoo/seongsan-driving-transitions |

## 저장소 구성

수업 권장 구분(Environment / Training / Evaluation / Data Collection)에 따라 정리했습니다.

| 구분 | PPO | BC (`bc/`) |
|---|---|---|
| **Environment** | `maps/seongsan/` 맵 생성 코드와 최종 맵(`generated/`)<br>`paper_training/` 관측·행동·보상·종료 판정 | `bc/implementation/` 데이터 수집 당시의 환경과 맵(v3) |
| **Training** | `train_rule_curriculum.py` | `bc/learned_driver.py` (교사), `bc/distill_driver.py` (BC 학습) |
| **Evaluation** | `evaluate.py`, `view_policy_gui.py` (sumo-gui) | `bc/evaluate.py` |
| **Data Collection** | `collect_seongsan_data.py` (수업 형식 jsonl.gz + npz) | `bc/collect.py`, `bc/parquet_io.py` (Hugging Face 데이터) |
| **Model** | `models/final_ppo/` (체크포인트, 입력 이름, 학습 설정) | `bc/driver_study/student_opt_bc.pt` |

`data/README.md`에 데이터셋 설명이, `bc/samples/`에 날씨별 샘플 에피소드가 있습니다.

## 도로와 교통

- 성산일출봉 주변 외곽 루프. 회전교차로, 신호 교차로, 비신호 교차로, 합류부, 전방향 정지(STOP)가 있습니다.
- 모든 도로에 차로를 하나씩 더하고 제한속도를 60 km/h로 맞춘 맵입니다. 도로 우선순위는 차로를 늘리기 전 차로 수를 기준으로 정합니다.
- 일반 차량 경로 42개(출발지 7 × 목적지 6). 기본 교통량은 합계 700대/시간이며, "교통량 N배"는 이 값에 N을 곱한 것입니다.
- AV는 고정 출발점에서 전체 경로를 한 바퀴 돌면 완주입니다. 충돌, 적색 신호 위반, STOP 미정지, 경로 이탈, 시간 초과는 실패입니다.
- SUMO의 자동 안전 기능은 끕니다(`speedMode 96`, `laneChangeMode 0`). 가감속과 차로 변경은 모두 정책이 정합니다.

## MDP 요약

| | PPO | BC |
|---|---|---|
| 관측 | 63개 (자차·주변 차량, 교차로 상충 차량, 경로, 신호 색, STOP, 0.8초 전 값 일부) | 17개 (속도, 앞차·옆 차로 간격, 신호·정지선 거리, 충돌 예상 지점 거리 등) |
| 행동 | 9개 = 가감속 {−4, 0, +3} m/s² × 차로 {오른쪽, 유지, 왼쪽}, 0.2초마다 | 같음 |
| 보상 | 충돌 −30, 그 밖의 실패 −10, 도착 +10, 교차로 통과 +5, 목표 속도 추종, 차로 변경·위험 차로 변경 페널티 | 학습에 쓰지 않음 (교사 행동 모방). 데이터에는 환경 보상을 기록 |

주행 중에는 정책 하나만 행동을 정합니다. 규칙 기반 보조나 시나리오 전환은 없습니다.

## 성능 (맑은 날, 최종 맵, 1000 에피소드, seed 920000–920999)

| 교통량 | PPO 충돌 | PPO 완주 | PPO 속도 | BC 충돌 | BC 완주 | BC 속도 |
|---|---|---|---|---|---|---|
| 1배 | 0.9% | 98.8% | 33.0 km/h | 0.1% | 99.9% | 22.6 km/h |
| 1.5배 | 4.1% | 94.8% | 33.4 km/h | 0.4% | 99.6% | 22.3 km/h |
| 2배 | 3.9% | 95.7% | 31.6 km/h | 0.6% | 99.4% | 22.4 km/h |
| 4배 | 19.2% | 76.4% | 28.4 km/h | 2.1% | 97.7% | 21.6 km/h |

- 속도는 완주한 에피소드의 평균입니다.
- PPO는 교통량 2~3배까지 학습했고 4배는 학습 범위 밖입니다. BC는 4배 교통으로 만든 시연을 학습했습니다.
- BC를 자신의 데이터 수집 맵(v3)에서 평가하면 충돌 0.3% / 0.8% / 0.4% / 1.4%입니다(같은 순서).

## 설치

Python 3.11, SUMO 1.27.1 기준입니다.

```bash
pip install -r requirements.txt
```

## 실행

```bash
# PPO: 화면 없이 평가 (결과는 results/eval_<교통량>x.jsonl)
python evaluate.py --traffic 2 --episodes 1000 --seed 920000

# PPO: sumo-gui로 주행 보기
python view_policy_gui.py --checkpoint models/final_ppo/model.pt --map pluslane_v4 --traffic 2 --speed 4

# PPO: 데이터 수집 (수업 collect_pomdp_data.py 형식, data/<name>/)
python collect_seongsan_data.py --traffic 1,1.5,2 --transitions 10000000 --name ppo_w39

# 맵 다시 만들기 (maps/seongsan/generated 와 같은 도로망이 생성됨)
python -c "from maps.seongsan.builder import build; build()"

# BC: 평가와 수집 (작업 폴더 bc/)
cd bc
python evaluate.py --start-seed 83000000 --episodes 100 --workers 8 --out paired-evaluation.jsonl
python collect.py --weather dense --start-seed 83000000 --episodes 1 --out outputs/dense-demo
```

PPO 학습 명령과 하이퍼파라미터는 `models/final_ppo/training_config.json`에 기록돼 있습니다. BC의 교사 재생성과 재학습 명령은 `bc/README.md`에 있습니다.

## 데이터셋 요약

| 날씨 | 에피소드 | transition | 감지 거리 (차량/신호/정지선, 50% 확률) |
|---|---:|---:|---|
| 맑음 | 5,327 | 5,000,539 | 기존 시야 규칙 |
| 옅은 안개 | 2,705 | 2,500,663 | 40 / 45 / 29 m |
| 짙은 안개 | 3,805 | 2,500,024 | 20 / 33 / 17 m |

합계 10,001,226 transition, 28개 칼럼(수업 기본 19개 + 저시정·규칙 9개), Parquet 약 12.1 GB입니다. 자세한 내용은 `data/README.md`와 `bc/SCHEMA.md`를 보세요.
