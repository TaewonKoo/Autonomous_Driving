# 다른 PC에서 전체 경로 평가

필요한 파일은 모두 이 브랜치의 `final_project/models/seongsan_20261123/` 아래에 있습니다.
`main`이 아닌 **`seongsan-models-20261123` 브랜치**를 받으세요. 모델만 개별 다운로드하면 지도·환경 코드가 빠집니다.

## 1. 다운로드·설치

```shell
git clone --branch seongsan-models-20261123 --single-branch https://github.com/TaewonKoo/Autonomous_Driving.git
cd Autonomous_Driving/final_project/models/seongsan_20261123
python -m venv .venv
```

Python3.11에서 검증했습니다. 가상환경 활성화는 Windows PowerShell에서 `.venv\Scripts\Activate.ps1`, Linux에서는 `source .venv/bin/activate`입니다.
활성화 후 아래 명령을 실행하세요. 학습·평가에 쓰인 라이브러리 버전을 고정했으며 CPU 추론을 사용합니다.

```shell
python -m pip install -r requirements-evaluation.txt
python evaluate_models.py --model r08c05 --check-only
```

`libsumo` 패키지의 내장 시뮬레이터를 사용하므로 이 평가에는 별도 SUMO GUI나 CARLA가 필요하지 않습니다.
체크포인트·환경 코드·지도 해시와 SUMO1.27.1을 대조합니다. 다른 환경 파일을 조용히 섞어 평가하지 않습니다.

## 2. 기준100회 병렬 평가

```shell
python evaluate_models.py --model r08c05 --seed-start 800000 --episodes 100 --workers 4 --output evaluation_results/r08c05_dev100
```

시드800000~800099를 정확히 한번씩 평가합니다. worker별 독립 프로세스를 사용하고 각 CPU 추론 thread는1입니다.
경로`ego_loop`, 교통배수1, dt=.2초, 제한500초, deterministic argmax, speedMode96/laneChangeMode0으로 고정합니다.
병렬화는 사례의 시드 배분만 바꾸고 관측·행동·보상·환경 계산을 바꾸지 않습니다.

출력은 `evaluation.json`(조건·요약·사례별 결과), `episodes.csv`, worker별 원래 평가기 출력입니다.
`--trace`를 붙이면 스텝별CSV와 입력/행동NPZ를 저장합니다. 출력 폴더는 새 경로를 사용하세요.
기존100회 개발 결과는 `reference_evaluations/`에 있어 사례별로 대조할 수 있습니다. 이100개는 모델 선정에 재사용한 개발 사례입니다.
공개 평가 묶음을 검증하면서 R08C05를4프로세스로100회 실행했고 **100개 사례의 모든 episode 기록이 기존 기록과 일치**했습니다.
완주70·충돌4·신호위반7·정지위반5·경로실패14회입니다. 검증 기록은 `evaluation_validation.json`에 있습니다.
이는 앞서 공개한 별도1000회 평가의 충돌67/1000(6.7%)와 다른 사례 집합입니다.

## 3. 여러 PC에 나눠서 실행

PC A:

```shell
python evaluate_models.py --model r08c05 --seed-start 800000 --episodes 50 --workers 4 --output evaluation_results/part_a
```

PC B:

```shell
python evaluate_models.py --model r08c05 --seed-start 800050 --episodes 50 --workers 4 --output evaluation_results/part_b
```

두 출력 폴더를 한 PC에 모은 뒤 병합합니다.

```shell
python evaluate_models.py --model r08c05 --seed-start 800000 --episodes 100 --merge evaluation_results/part_a/evaluation.json evaluation_results/part_b/evaluation.json --output evaluation_results/merged100
```

가중치·소스·지도·평가조건이 같은지 검사하고, 중복·누락·범위 밖 시드는 오류로 처리합니다.
동일 버전에서도 OS·CPU 부동소수점 계산 차이에 따른 결과 차이가 있을 수 있으므로 각 shard의 환경 버전을 결과에 기록합니다.
원래 결과와 차이가 나면 동일 시드의 초기속도·배경차량수·첫 행동·종료 스텝부터 비교하세요.

## 4. 다른 두 모델

위 명령의 `--model`만 `r06c03` 또는 `r08c01`로 바꾸면 같은 환경과 시드에서 평가합니다.
모델별 내부 프로파일을 읽으므로 R08C05의 설정을 다른 모델에 덮어 적용하지 않습니다.

## 실제 파일 위치

| 용도 | 파일 |
|---|---|
| 병렬 실행·결과 병합 | `evaluate_models.py` |
| 원래 전체 경로 평가 함수 | `implementation/train_observed_driving.py`의 `evaluate` |
| 성산 관측·행동·보상·종료·리셋 | `implementation/paper_training/observed_driving.py`의 `ObservedDriving` |
| PPO 정책·가중치 로드에 필요한 모델 클래스 | `implementation/paper_training/agents.py` |
| 신경망 생성 | `implementation/utils/networks.py` |
| 실제 지도·배경교통·경로 | `implementation/maps/seongsan/generated/seongsan.net.xml`, `seongsan.rou.xml` |
| 의존 패키지 | `requirements-evaluation.txt` |
| 모델별 실제 설정 | 각 체크포인트와 동명인 `*.settings.json` |

루트 저장소의 오래된 `final_project` 코드를 대신 import하지 말고 이 폴더의 실행 스크립트를 사용하세요.
원래 학습 프로그램은 기본 stage가signal·traffic이0이므로, 직접 실행한다면 반드시 아래처럼 전체경로 옵션을 명시해야 합니다.

```shell
python implementation/train_observed_driving.py evaluate --checkpoint seongsan_ppo_r08c05_seed25.pt --stage full --traffic 1 --episodes 100 --evaluation-seed 800000 --policy-only --output evaluation_results/original_serial100
```

이 직접 실행 명령은 직렬이며 스텝별 trace를 저장합니다. 일반 비교평가에는 위의 `evaluate_models.py` 명령을 권합니다.
