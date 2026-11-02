# 자율주행 파이널 프로젝트

## 새로 만든 성산 고성리 맵

사진의 빨간 경로를 따른 폐루프와 2차로 로터리를 구현했습니다.
남쪽 왕복 2차로 직선도로·사거리·삼거리와 실제 사진 기반 합류부까지 확장했습니다.
북쪽 내부 연결도로와 가상 식당은 삭제했고, 일출로 곡선과 작은 빗금·유도봉 구역을 사진에 맞춰 보정했습니다.
전체 범위는 약 414m × 369m, 기본 경로는 출발·도착 위치가 같은 약 1.06km 최외곽 루프입니다.
출발지는 북서쪽 사거리에서 도로를 따라 동쪽으로 100m 떨어진 지점입니다.
사거리 쪽으로 출발한 뒤 기존 방향으로 순환하고 같은 지점으로 돌아옵니다.
로터리 남쪽 북상 진입은 1→2차로, 오른쪽 삼각형 접근부의 북상은 2→1차로입니다.
신호등은 왼쪽 아래 사거리 1곳만 유지하고, 은행·동쪽 삼거리·북동쪽 삼거리는 정지선에서 일시정지 후 통행합니다.
변경 전 신호등 4곳 버전은 상위의 `final_project_backup2`에 보존했습니다(맵 주요 파일 26개 일치 확인).
차로 전환부를 완만하게 연결하고 불필요한 정지선을 정리했습니다.
전체 도로를 연속 곡선으로 보간했습니다. 검은 포장면과 흰 테두리는 실제 차로와 교차로를
합친 하나의 경계를 사용하며, 차로 점선도 실제 차로 위치를 따라 그립니다.
남쪽 오른쪽 도로·하단 도로와 마을 연결로·동쪽 본선의 불필요한 굴곡은 직선으로 정리했습니다.
포장면이 전체 차로 폭을 포함하는지도 생성 시 검사해 바깥 차로가 잘려 보이는 것을 방지합니다.
로터리 연결부의 포장면 틈을 메우고 경계선·차로 점선·진행 화살표를 정리했습니다.
동쪽 진입로는 좌회전 정지선 차로와 우회전 양보 차로로 나뉩니다. 우회전은 별도 우회 차로로 본선에 합류합니다.
동쪽 삼거리는 새 항공사진·거리뷰를 기준으로 본선을 직선화하고 분기점을 당겼습니다.
우회전 차로를 본선에 접선으로 연결하고, 실제 차로 사이 교통섬의 안쪽으로 노란 빗금을 배치했습니다.
오른쪽 두 차로 진입부와 우회전 차로를 같은 축으로 정렬해 바깥 흰 경계를 약 36.2m의 직선으로 이었습니다.
본선 합류부 끝의 8m는 본선과 평행하게 연결했습니다.
우회전 차로는 양보 점선, 나머지 진입부는 차로 폭에 맞는 정지선으로 표시하며 기본 막대의 겹침을 제거했습니다.
횡단보도는 전체 맵에서 모두 제거했습니다.
왼쪽 정지선은 본선 입구의 사선 위치에 놓이고 중앙선도 그 선까지 이어집니다.
오른쪽 양보선은 이전보다 7.5m 아래쪽으로 옮겼습니다.
분기부의 작은 꺾임과 불필요한 흰 막대도 정리했습니다.
로터리 동쪽 골목 분기점은 중앙선으로 나눠 우회전 진입·진출만 허용합니다.
삼각형 왼쪽은 남하 2차로·북상 1차로이며 남하 좌회전은 중간 연결로에서만 가능합니다. 중앙선은 모두 노란색입니다.
`python view_map.py`로 열고 ▶를 누르면 빨간 차량과 일반 차량이 주행합니다.
기존 짧은 경로는 `python view_map.py --route original`로 선택할 수 있습니다.
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
