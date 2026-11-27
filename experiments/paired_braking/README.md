# 같은 장면에서 안개가 제동 판단에 미치는 영향

교통량 4배의 성산 수집 맵(v3)에서 최종 BC가 맑음 관측으로 주행한다. 매 0.2초마다 같은 상태에 맑음·옅은 안개·짙은 안개 관측을 적용해 같은 정책의 출력을 비교한다. 복원 비교에서는 안개 관측의 선행차 간격과 속도만 맑음 값으로 되돌린다. 실제 차량 움직임에는 맑음 행동만 적용한다.

| 맑음에서 제동한 7,086개 시점 | 옅은 안개 | 짙은 안개 |
|---|---:|---:|
| 제동을 선택하지 않은 시점 | 927 (13.1%) | 2,474 (34.9%) |
| 그중 선행차가 누락된 시점 | 18 | 691 |
| 선행차 간격·속도 복원 후 다시 제동한 시점 | 13 / 18 | 528 / 691 |

[시드별 집계](episode_counts.csv) · [집계와 에피소드 단위 부트스트랩 95% 구간](summary.json) · [SUMO-GUI 영상](sumo_gui_perception.mp4) · [대표 장면](sumo_gui_perception_keyframe.png)

## 재현

저장소 루트에서 실행한다. BC 의존성은 루트 README의 BC 설치 절차를 따른다. 전체 집계는 영상의 유무와 관계없이 실행할 수 있다.

```sh
python experiments/paired_braking/study_braking.py
python experiments/paired_braking/analyze.py
```

첫 명령은 episode_counts.csv에 기록한 고정 시드 100개를 10개 프로세스로 실행하고 frames/에 결과를 저장한다. 두 번째 명령은 같은 폴더의 두 집계 파일을 다시 쓴다. 원 실험에서는 공개 맑음 에피소드 목록에서 난수 시드 2701로 100개를 비복원 추출했다.

## 대표 장면과 영상

대표 장면은 선행차 정보를 복원하자 제동이 돌아온 후보 중 선행차에 접근 중이고 전후 영상 구간이 충분한 419개 시점에서 골랐다. 차간거리가 후보들의 중앙값에 가장 가까운 장면이 시드 91006700의 333번째 시간 인덱스(66.6초)다.

```sh
python experiments/paired_braking/study_braking.py 91006700
python -m pip install Pillow imageio-ffmpeg
python experiments/paired_braking/sumo_perception_video.py
```

영상 재생성은 Windows의 맑은 고딕과 SUMO-GUI 1.27.1이 필요하다. trace.json에 기록한 시점별 감지 목록으로 차량 색을 지정하고 실제 SUMO-GUI 화면을 캡처한다. 빨강은 자차, 청록은 인지 차량, 회색은 미인지 차량이다. 영상 좌우의 차량 위치는 동일하다.
