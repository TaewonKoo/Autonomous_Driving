# 자율주행 파이널 프로젝트

## 새로 만든 성산 고성리 맵

사진의 빨간 경로를 따른 폐루프와 2차로 로터리를 구현했습니다.
`python view_map.py`로 열고 ▶를 누르면 빨간 차량과 일반 차량이 주행합니다.
[맵 설명](maps/seongsan/README.md) · [도로 미리보기](maps/seongsan/preview.svg)

기존 `train.py`와 `view_road.py`는 아래에 설명된 원본 고속도로 예제를 실행합니다.
새 맵의 학습 환경 연동은 다음 단계입니다.

수업 저장소에서 파이널 프로젝트에 필요한 코드와 안내만 가져온 작업 폴더입니다.
Python 원본 코드는 그대로 유지했습니다. 현재 도로는 기본 직선 고속도로 예제이며,
팀이 선정한 실제 한국 도로에 맞게 수정해야 합니다.

## 폴더 열기

VS Code에서 이 `final_project` 폴더를 열고, 터미널도 이 폴더에서 실행하세요.

## 준비 및 실행

SUMO가 설치되어 있어야 합니다. 자세한 설치 방법은
[Windows 설치 안내](project_materials/1_SUMO_Windows_Setup.md)를 참고하세요.
해당 안내의 저장소 다운로드 단계는 이미 완료된 상태입니다.

PowerShell에서 가상환경을 만들고 패키지를 설치합니다.
가상환경을 활성화하지 않고도 다음과 같이 실행할 수 있습니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
sumo --version
.\.venv\Scripts\python.exe view_road.py
```

화면 없이 도로 확인:

```powershell
.\.venv\Scripts\python.exe view_road.py --nogui --seconds 120
```

강화학습:

```powershell
.\.venv\Scripts\python.exe train.py
```

학습한 모델 평가 (`run_실험폴더`를 실제 생성된 폴더명으로 교체):

```powershell
.\.venv\Scripts\python.exe test.py results/run_실험폴더/model.pt --episodes 100 --nogui
```

환경 점검용 데이터 수집 및 선택 사항인 모방학습:

```powershell
.\.venv\Scripts\python.exe collect_pomdp_data.py --policy keep-lane --episodes 1 --name env_check
.\.venv\Scripts\python.exe train_bc.py --data data/env_check.npz --epochs 5
```

`env_check`는 실행 점검용 데이터입니다. 최종 모방학습에는 충분한 양의 양질의 주행 데이터를 수집하세요.
`data/`, `results/`, `env/sumo/`는 필요한 실행 과정에서 자동 생성됩니다.
기존 학습 모델을 제외했으므로 모델 평가는 직접 학습한 이후에 실행할 수 있습니다.

## 주로 수정할 파일

| 작업 | 파일 |
|---|---|
| 도로, 차선, 교통량 설정 | `env/road_config.py` |
| 교차로, 연결, 경로 생성 | `env/road_builder.py` |
| 관측, 행동, 기본 보상 설정 | `env/mdp_config.py` |
| 차량 제어, 관측 계산, 종료 조건 | `env/sumo_env.py` |
| PPO 학습 설정 및 보상 덮어쓰기 | `train.py` |
| 최종 평가 | `test.py`, `utils/evaluator.py` |
| 데이터 수집 | `collect_pomdp_data.py` |

`env/sumo/`의 기본 도로 파일은 실행 시 다시 생성되므로,
도로를 영구적으로 바꾸려면 `road_builder.py`와 설정을 수정하세요.
netedit으로 직접 제작할 경우 자동 덮어쓰기 구조와 `.gitignore`도 함께 수정해야 합니다.

## 남긴 자료와 제외한 파일

도로 환경, PPO/BC 학습, 평가, 데이터 수집 코드와 프로젝트 소개·Windows 설치·도로 구성·학습·준비·평가 문서를 남겼습니다.
BC는 선택 사항이지만 평가 코드에서도 사용하므로 관련 코드를 유지했습니다.

제외한 항목:

- `results/`: 기존 예제 모델, 학습 로그, TensorBoard 기록, 설정 사본
- `data/dataset.npz`: 기존 예제 주행 데이터
- `env/sumo/`: 기본 코드로 재생성되는 도로 XML 및 시각 요소
- macOS 설치 안내, 컨설팅 안내, 원본 저장소의 목차용 README

추천 읽기 순서:
[프로젝트 소개](project_materials/0_Project_Introduction.md) →
[프로젝트 준비](project_materials/5_Project_Preparation.md) →
[평가 기준](project_materials/6_Project_Evaluation.md).

## 원본 출처

- [수업 GitHub 저장소](https://github.com/bmil-ssu/Artificial-Intelligence-and-Control-for-Autonomous-Driving)
- [프로젝트 자료](https://github.com/bmil-ssu/Artificial-Intelligence-and-Control-for-Autonomous-Driving/tree/main/project_materials)
- 다운로드 기준: 2026-10-27, 정확한 원본 커밋은 `UPSTREAM_COMMIT.txt`에 기록

초기 원본 정리 시 추가한 파일은 `README.md`, `requirements.txt`, `.gitignore`, `UPSTREAM_COMMIT.txt`입니다.
이후 사진 기반 성산 맵을 위한 `maps/`와 `view_map.py`를 추가했습니다.
별도의 Git 이력이나 원본 저장소의 원격 연결은 포함하지 않았습니다.
