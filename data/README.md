# 데이터

전체 데이터는 용량(약 12.1 GB) 때문에 Hugging Face에 있습니다.

**https://huggingface.co/datasets/taewonkoo/seongsan-driving-transitions**

- BC 정책 `student_opt_bc`가 교통량 4배에서 주행하며 수집했습니다 (`bc/collect.py`).
- 11,837 에피소드, 10,001,226 transition. 맑음 50%, 옅은 안개 25%, 짙은 안개 25%.
- 한 행은 0.2초 transition 하나이고 28개 칼럼입니다.
  - 수업 기본 칼럼 19개: `episode`, `t`, `observation`, `action_raw`, `action.*`, `reward`, `terminated`, `truncated`, `state`, `detected_vehicles`, `event.*`, `next_observation`, `next_state`
  - 저시정·규칙 칼럼 9개: `weather_id`, `obs_profile_id`, `signal_exists_map`, `red_obs`, `yellow_obs`, `green_obs`, `stopline_dist_m`, `red_violation`, `stop_violation`
- 칼럼별 단위와 결측값은 `bc/SCHEMA.md`, 날씨별 샘플 에피소드는 `bc/samples/`에 있습니다.
- 충돌·위반 주행도 포함합니다. 완주 시연만 쓰려면 Hugging Face의 `episodes.parquet`에서 `outcome == "complete"`인 에피소드를 고르면 됩니다.

PPO 정책으로도 같은 기본 칼럼(jsonl.gz + npz)을 수집할 수 있습니다: `python collect_seongsan_data.py --name <이름>` (결과는 이 폴더 아래에 저장).
