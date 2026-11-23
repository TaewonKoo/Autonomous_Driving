# new map — 성산 최신 맵

맵 생성 파일 시각: **2026-11-24T03:06:49+09:00**. 방향별 도로 79개, 일반 차로 179개, 기본 루프 약 1031.1m입니다.

프로젝트의 최신 `maps/seongsan` 생성 파일과 소스를 별도로 보관한 맵입니다. 원본 파일을 그대로 복사했으며, 업로드를 위해 도로 설계나 차량 설정을 변경하지 않았습니다. 생성 시각·실제 차로 수·파일별 SHA-256은 [MAP_SNAPSHOT.json](MAP_SNAPSHOT.json)에 기록했습니다.

## SUMO에서 열기

SUMO 1.27.1을 설치한 뒤 이 폴더에서 실행합니다.

```shell
sumo-gui -c generated/seongsan.sumocfg
```

SUMO GUI의 파일 열기 메뉴에서 `generated/seongsan.sumocfg`를 선택해도 됩니다. 기본 재생 지연은 GUI에서 조절할 수 있습니다. 이 설정 파일의 빨간 차량은 SUMO 기본 시연 차량입니다.

## 파일 구성

| 위치 | 내용 |
|---|---|
| `generated/seongsan.sumocfg` | 실행 설정 |
| `generated/seongsan.net.xml` | 실제 도로망·차로·신호·연결 |
| `generated/seongsan.rou.xml` | 시연 경로·배경 교통 |
| `generated/seongsan.add.xml` | 노면·정지선·도류구역 등 표시 |
| `generated/seongsan.gui.xml` | 화면 설정 |
| `generated/seongsan.nod.xml`, `seongsan.edg.xml`, `seongsan.con.xml` | 편집용 도로 정의 |
| `generated/manifest.json` | 맵 생성 설정 |
| [preview.svg](preview.svg), `previews/` | 전체·구간별 미리보기 |
| `source/maps/seongsan/` | 최신 맵 생성 코드 원본 |
| `source/view_map.py` | 원본 맵 열기·검사 도구 |

## 소스에서 다시 만들기

Python 3.11 환경을 권장합니다. 지도 생성에 필요한 패키지 버전은 `requirements.txt`에 고정했습니다.

```shell
python -m pip install -r requirements.txt
python rebuild.py
```

`rebuild.py`는 이 폴더의 `generated/`와 미리보기를 다시 만듭니다. 별도 경로로 생성하려면 다음처럼 실행합니다.

```shell
python rebuild.py --output build/generated
```

기본 시연 차량 없이 생성하려면 `--background-only`, 배경 교통 없이 생성하려면 `--no-traffic`을 사용합니다.

기존 세 학습 모델과 그 평가 환경은 같은 브랜치의 `final_project/models/seongsan_20261123/`에 있습니다. 이 `new map` 폴더는 최신 지도 자산을 별도로 배포한 것입니다.
