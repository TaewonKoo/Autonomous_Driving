# 전이 스키마

CSV의 한 행은 0.2초 전이다. Parquet은 중첩 배열·구조체, float32, int32, bool을 사용한다. CSV에는 벡터와 구조체를 JSON으로 쓴다.

| 칼럼 | 값과 단위 |
|---|---|
| episode, t | 배치 내부 에피소드 번호(1부터), 에피소드 내 시간 인덱스(0부터). seed는 배치의 episodes.json과 경로에 있다. |
| observation, next_observation | 실제 BC의 현재·다음 입력 float[17]. 종료·시간제한 마지막 행의 다음 관측은 영벡터다. |
| action_raw | 선택한 행동의 [가감속(m/s²), 차로 방향]. |
| action.accel_raw, action.lane_change_raw, action.lane_change | 요청 가감속 -4/0/3, 요청 방향 및 해석된 방향 -1/0/+1. |
| action.lane_change_applied | 실제 같은 도로에서 차로 인덱스가 바뀌었는지. |
| reward, terminated, truncated | 스텝 보상, 환경 종료, 시간제한 종료. |
| state, next_state | 시각 sim_time_s, ego_id=0, vehicles 목록(vehicle_id, lane_id, x_m, y_m, speed_mps, heading_deg), signals 목록(tls_id, link_index, state_gt). 방향은 SUMO 북쪽 0도·시계 방향. 신호는 SUMO 문자의 ASCII 정수. |
| detected_vehicles | 감지된 대상별 track_id, rel_x_m(전방+), rel_y_m(좌측+), observed_speed_mps. 미감지 목록은 비어 있다. |
| event.collided, event.arrived | 자차 충돌과 성공 도착 여부. |
| event.lane_before, event.lane_after | 전후 차로 정수 ID. 차량 소실 후에는 null. lookup.json에 차로·신호 매핑이 있다. |
| weather_id, obs_profile_id | 날씨 0 맑음, 1 옅은 안개, 2 짙은 안개. 프로필 3은 이번 BC 17입력 +5m 설정. |
| signal_exists_map, red_obs, yellow_obs, green_obs | 경로상 지도 신호 유무와 관측 색 0/1. 색 미판독 시 세 값 모두 0. |
| stopline_dist_m | 경로상 다음 도달 가능한 신호 또는 전방향 정지 지점까지의 실제 거리(m), 없으면 -1. |
| red_violation, stop_violation | 해당 스텝의 적색 진입, 정지 의무 위반 여부. |

Parquet에서 중간 행의 next_state와 next_observation은 null로 두고 같은 에피소드 다음 행의 현재 값으로 복원한다. 마지막 행에는 실제 저장한 다음 값을 직접 쓴다. 변환기는 에피소드 연속성과 중복 값 일치를 검사한다.

## BC 입력 순서

0 속도(m/s), 1 제한속도(m/s), 2 계획 차로 방향(-1/0/1), 3·4 옆 차로 앞·뒤 간격(m), 5·6 옆 차로 앞·뒤 차량 상대속도(m/s), 7 선행차 간격(m), 8 선행차 속도(m/s), 9 정지 의무 지점 거리(m), 10 판독한 적·황 신호 거리(m), 11 신호 판독 여부(0/1), 12 지도 제어 지점 거리(m), 13 경로 인덱스, 14 시야 범위(m), 15 현재 차로 시작점에서 주행한 거리(m), 16 예측 교차 충돌 지점 거리(m). 간격·충돌 후보 기본값 100m, 정지·신호 후보 기본값 999m. 정확한 선택 규칙은 learned_driver.features와 collect.py에 있다. 내부 27차원은 모델 파생 특징이고, 별도 논의한 55차원 관측은 이 체크포인트의 입력에 포함되지 않는다.
