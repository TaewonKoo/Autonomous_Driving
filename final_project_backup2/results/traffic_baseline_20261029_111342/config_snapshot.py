"""사진 기준 좌표. x는 오른쪽, y는 아래쪽 (두 번째 첨부사진 기준).

실측/측량 데이터가 아니다. 축척은 사용자 제공 거리 약 600m로 보정했다.
차로 수만 사용자 지정: 서쪽 왕복 2차로, 로터리 동쪽 왕복 4차로.
북쪽 해안 구간은 우선 왕복 2차로로 가정한다.
"""

SCALE = 0.543344  # 로터리 동쪽 출구부터 북쪽 좌회전 진입부까지 약 600m
REFERENCE_DISTANCE_M = 600
REFERENCE_EDGES = ["eomeong_road", "east_sweep", "east_main",
                   "godeungeo_road", "godeungeo_approach"]
ISLAND_RADIUS_PIXELS = 17 / 0.75  # 중앙섬도 사진 좌표와 같은 축척 적용
LANE_WIDTH = 3.2
SPEED_LOCAL = 50 / 3.6
SPEED_MAIN = 60 / 3.6
SPEED_RING = 20 / 3.6
RING_LANES = 2  # 로터리 안쪽/바깥쪽 모두 반시계 방향
TRAFFIC_PER_PORT = 100  # 각 외부 진입점의 시간당 총 수요
TRAFFIC_END = 3600
SEED = 23
STEP_LENGTH = 0.2
GUI_DELAY_MS = 200  # 0.2초 스텝당 200ms: 대략 실제 시간과 같은 재생 속도

# +1: STREETS에 적힌 방향만, -1: 역방향만. 미등록 도로는 양방향.
ONE_WAY_STREETS = {"inner_join": -1}  # 동쪽 큰길 inner_exit → 마을 inner_e
STRAIGHT_ONLY_CONNECTIONS = {"inner_join_rev": "inner_east_rev"}
ONE_WAY_STREETS.update({"restaurant_entry": 1, "restaurant_exit": 1})
SOUTH_LANES_PER_DIRECTION = 2  # 사용자 지정: 남쪽 두 간선도로 모두 왕복 4차로
ROAD_WIDTH_OVERRIDES = {
    "triangle_top": 3.0, "triangle_west_leg": 3.0, "triangle_east_leg": 3.0,
    "restaurant_entry": 2.6, "restaurant_exit": 2.6, "restaurant_parking": 2.6,
}
ROAD_SPEED_OVERRIDES = {}  # 후속 현장 보정용 (m/s)

# 졸띠 해변승마 쪽 inner_e는 무신호. 아래쪽 east_cross만 신호 적용.
SIGNAL_CONTROLLERS = {
    "signal_bank": ["bank"],
    "signal_east": ["east_cross"],
    "signal_godeungeo": ["godeungeo"],
}
# 현장 신호 시간 미제공: 임시 고정 주기, netconvert가 방향별 현시 생성.
SIGNAL_TIMING = {"cycle_seconds": 90, "yellow_seconds": 3, "all_red_seconds": 2}
SCHOOL_ZONE = {
    "speed_kmh": 30,
    "from": "범하다 부근 / 제주은행", "to": "보룡제과 앞 / 은행 남동쪽 CU",
    "edges": ["school_approach", "west_bank", "west_lower", "bank_link"],
    "sign_nodes": ["school_north", "west_lower", "cu_bank"],
    "approach_edges": ["bank_village", "roundabout_approach"],  # CU/로터리 쪽 진입 전 감속 준비
}
# 북쪽 접근로는 사용자 지정 마을 도로 50km/h, 보호구역은 30km/h.
SCHOOL_NORTH_APPROACH_EDGES = ["start_south", "hamabo_return"]
# 같은 교통량 안에서 은행–CU 분기도 양방향으로 이용하는 예시 경로를 지정.
TRAFFIC_VIA_EDGES = {
    ("port_north", "port_se"): "bank_link",
    ("port_se", "port_north"): "bank_link_rev",
    ("port_south", "port_ne"): ["restaurant_entry", "restaurant_exit"],
    ("port_sw", "port_se"): "triangle_west_leg",
}

POINTS = {
    "hamabo": (64, 263),
    "school_north": (95, 355),
    "west_cross": (120, 416),
    "bank": (145, 511),
    "cu_bank": (245, 572),
    "cu_approach": (300, 608),
    "west_lower": (155, 784),
    "rb_n": (155, 824),
    "rb_w": (121, 858),
    "rb_s": (155, 892),
    "rb_e": (189, 858),
    "rb_ne": (179.04163, 833.95837),
    "eomeong": (340, 819),
    "east_curve": (700, 742),
    "east_cross": (953, 520),
    "godeungeo": (1160, 315),
    "inner_exit": (1103, 372),
    "coast_bend": (1002, 138),
    "coast_top": (601, 24),
    "north_cross": (334, 156),
    "northwest": (51, 218),
    "inner_w": (355, 381),
    "inner_c": (604, 464),
    "inner_e": (835, 434),
    "inner_s": (363, 649),
    # 남쪽 확장. 좌표는 기존 600m 보정 축척에 맞춘 사진 기반 초기 배치.
    "triangle_west": (600, 774),
    "triangle_south": (710, 1000),
    "south_cross": (155, 3100),
    "south_t": (950, 3000),
    "restaurant_in": (800, 1800),
    "restaurant_out": (815, 1900),
    "restaurant_gate": (980, 1850),
    # 바깥 도로의 경계: 생성/소멸은 이 지점에서만 발생한다.
    "port_north": (5, 20),
    "port_nw": (-130, 245),
    "port_west": (-95, 438),
    "port_sw": (-85, 881),
    "port_south": (155, 3520),
    "port_se": (1390, 3000),
    "port_ne": (1340, 205),
    "port_south_west": (-230, 3100),
    "port_restaurant": (1070, 1850),
}

# (도로 ID, 시작점, 끝점, 방향별 차로 수, 속도, 중간 형상점)
# 기본은 양방향 생성. ONE_WAY_STREETS 지정 시 해당 방향만 생성.
# *_rev가 반대 방향이다.
STREETS = [
    ("start_south", "hamabo", "school_north", 1, SPEED_LOCAL, [(88, 330)]),
    ("school_approach", "school_north", "west_cross", 1, SPEED_LOCAL, []),
    ("west_bank", "west_cross", "bank", 1, SPEED_LOCAL, []),
    ("west_lower", "bank", "west_lower", 1, SPEED_LOCAL, [(154, 620)]),
    ("roundabout_approach", "west_lower", "rb_n", 1, SPEED_LOCAL, []),
    # 보룡제과–사계정 사이: 기존 북쪽 진입로와 구분되는 북동쪽 진출입로.
    # 좁은 마을길로 우선 왕복 2차로를 적용한다.
    ("bakery_sagyejeong", "rb_ne", "inner_s", 1, SPEED_LOCAL,
     [(214, 794), (281, 726), (324, 687)]),
    ("eomeong_road", "rb_e", "eomeong", 2, SPEED_MAIN, [(240, 846)]),
    ("east_sweep", "eomeong", "triangle_west", 2, SPEED_MAIN, [(470, 796)]),
    # 삼각형 분기부의 세 변은 방향별 1차로로 좁아진다.
    ("triangle_top", "triangle_west", "east_curve", 1, SPEED_LOCAL, []),
    ("triangle_west_leg", "triangle_west", "triangle_south", 1, SPEED_LOCAL,
     [(638, 825), (685, 928)]),
    ("triangle_east_leg", "east_curve", "triangle_south", 1, SPEED_LOCAL,
     [(745, 830), (738, 925)]),
    ("east_main", "east_curve", "east_cross", 2, SPEED_MAIN,
     [(767, 700), (843, 626)]),
    ("godeungeo_road", "east_cross", "inner_exit", 2, SPEED_MAIN,
     [(1050, 423)]),
    ("godeungeo_approach", "inner_exit", "godeungeo", 2, SPEED_MAIN, []),
    ("coast_turn", "godeungeo", "coast_bend", 1, SPEED_LOCAL,
     [(1115, 268), (1067, 202)]),
    ("coast_north", "coast_bend", "coast_top", 1, SPEED_LOCAL,
     [(946, 111), (824, 78), (712, 50)]),
    ("north_return", "coast_top", "north_cross", 1, SPEED_LOCAL,
     [(520, 55), (427, 102)]),
    ("northwest_return", "north_cross", "northwest", 1, SPEED_LOCAL,
     [(275, 175), (177, 200)]),
    ("hamabo_return", "northwest", "hamabo", 1, SPEED_LOCAL, []),
    # 사진에 보이는 내부 연결로: 일반 차량이 폐루프를 드나든다.
    ("inner_west", "west_cross", "inner_w", 1, SPEED_LOCAL, [(254, 397)]),
    ("inner_center", "inner_w", "inner_c", 1, SPEED_LOCAL, [(492, 451)]),
    ("inner_east", "inner_c", "inner_e", 1, SPEED_LOCAL, []),
    ("inner_join", "inner_e", "inner_exit", 1, SPEED_LOCAL, [(1040, 399)]),
    ("inner_diagonal_n", "north_cross", "inner_w", 1, SPEED_LOCAL,
     [(342, 255)]),
    ("inner_diagonal_s", "inner_w", "inner_s", 1, SPEED_LOCAL,
     [(365, 520)]),
    ("bank_link", "bank", "cu_bank", 1, SPEED_LOCAL, []),
    ("bank_village", "cu_bank", "cu_approach", 1, SCHOOL_ZONE["speed_kmh"] / 3.6, []),
    ("cu_village", "cu_approach", "inner_s", 1, SPEED_LOCAL, []),
    ("inner_south", "inner_s", "inner_c", 1, SPEED_LOCAL, [(477, 549)]),
    ("cross_link", "inner_e", "east_cross", 1, SPEED_LOCAL, []),
    # 왼쪽 로터리~사거리: 형상 중간점 없이 일직선, 방향별 2차로.
    ("south_straight", "rb_s", "south_cross", SOUTH_LANES_PER_DIRECTION, SPEED_MAIN, []),
    ("south_connector", "south_cross", "south_t", SOUTH_LANES_PER_DIRECTION, SPEED_MAIN, []),
    ("coast_south_north", "triangle_south", "restaurant_in", SOUTH_LANES_PER_DIRECTION, SPEED_MAIN,
     [(720, 1200), (750, 1550)]),
    ("coast_facility_link", "restaurant_in", "restaurant_out", SOUTH_LANES_PER_DIRECTION, SPEED_MAIN, []),
    ("coast_south_south", "restaurant_out", "south_t", SOUTH_LANES_PER_DIRECTION, SPEED_MAIN,
     [(850, 2250), (880, 2600)]),
    # 식당: 차로 폭 2.6m의 별도 진입/진출로와 주차장 생성·소멸 경계.
    ("restaurant_entry", "restaurant_in", "restaurant_gate", 1, 10 / 3.6, [(865, 1840)]),
    ("restaurant_exit", "restaurant_gate", "restaurant_out", 1, 10 / 3.6, [(905, 1880)]),
    # 경계 도로는 지도 밖까지 연장해 차량 생성/소멸을 표현한다.
    ("boundary_north", "port_north", "northwest", 1, SPEED_LOCAL, []),
    ("boundary_nw", "port_nw", "northwest", 1, SPEED_LOCAL, []),
    ("boundary_west", "port_west", "west_cross", 1, SPEED_LOCAL, []),
    ("boundary_sw", "port_sw", "rb_w", 1, SPEED_LOCAL, []),
    ("boundary_south", "port_south", "south_cross", SOUTH_LANES_PER_DIRECTION, SPEED_MAIN, []),
    ("boundary_se", "port_se", "south_t", SOUTH_LANES_PER_DIRECTION, SPEED_MAIN, []),
    ("boundary_ne", "port_ne", "godeungeo", 2, SPEED_MAIN, []),
    ("boundary_south_west", "port_south_west", "south_cross", SOUTH_LANES_PER_DIRECTION, SPEED_MAIN, []),
    ("restaurant_parking", "port_restaurant", "restaurant_gate", 1, 10 / 3.6, []),
]

# 북쪽 진입 → 서 → 남 → 동으로 반시계 회전 후 동쪽 출구 진출.
RING_NODES = ["rb_n", "rb_w", "rb_s", "rb_e", "rb_ne"]
# 각 출입구의 수학적 각도. 360도 이후도 반시계 순서가 유지되도록 표기.
RING_ANGLES = [90, 180, 270, 360, 405]
ORIGINAL_EGO_LOOP = [
    "start_south", "school_approach", "west_bank", "west_lower", "roundabout_approach",
    "ring_0", "ring_1", "ring_2", "eomeong_road", "east_sweep", "triangle_top",
    "east_main", "godeungeo_road", "godeungeo_approach", "coast_turn", "coast_north",
    "north_return", "northwest_return", "hamabo_return",
]
EGO_LOOP = [
    "start_south", "school_approach", "west_bank", "west_lower", "roundabout_approach",
    "ring_0", "ring_1", "south_straight", "south_connector",
    "coast_south_south_rev", "coast_facility_link_rev", "coast_south_north_rev",
    "triangle_east_leg_rev", "east_main", "godeungeo_road", "godeungeo_approach",
    "coast_turn", "coast_north", "north_return", "northwest_return", "hamabo_return",
]
EGO_ROUTES = {"expanded": EGO_LOOP, "original": ORIGINAL_EGO_LOOP}
REFERENCE_EDGES.insert(2, "triangle_top")
FEATURE_EDGE_GROUPS = {
    "south_straight": ["south_straight", "south_straight_rev"],
    "south_connector": ["south_connector", "south_connector_rev"],
    "southern_coast": ["coast_south_north", "coast_south_north_rev",
                       "coast_south_south", "coast_south_south_rev"],
    "triangle": ["triangle_top", "triangle_top_rev", "triangle_west_leg", "triangle_west_leg_rev",
                 "triangle_east_leg", "triangle_east_leg_rev"],
    "restaurant_entry": ["restaurant_entry"],
    "restaurant_exit": ["restaurant_exit"],
}

LABELS = [
    ("출발 / 복귀 · 하마보 앞", (87, 263)),
    ("서쪽 도로 · 왕복 2차로", (11, 610)),
    ("어린이보호구역 30km/h · 범하다–보룡제과", (180, 570)),
    ("은행–CU 구간도 30km/h", (240, 610)),
    ("고성리 로터리 · 반시계 2차로", (175, 923)),
    ("보룡제과–사계정 사이 연결로", (225, 760)),
    ("성산일출봉 어멍횟집 방면", (390, 864)),
    ("아래쪽 큰길 · 왕복 4차로 · 60km/h", (785, 688)),
    ("편도 1차로 · 큰길 → 마을 · 직진만", (800, 380)),
    ("보호구역 밖 마을 도로 50km/h", (370, 315)),
    ("성산고등어 쌀밥 방면", (1004, 364)),
    ("좌회전 → 북쪽 해안 구간", (817, 32)),
    ("서쪽으로 돌아 출발점 복귀", (150, 122)),
    ("삼각형 분기·합류 / 좁아지는 구간", (770, 1050)),
    ("로터리–사거리 직선 / 왕복 4차로·60km/h", (215, 1560)),
    ("오른쪽 도로 / 왕복 4차로·60km/h", (1000, 2350)),
    ("식당 진입 →", (850, 1765)),
    ("← 식당 진출", (845, 1970)),
    ("가정한 식당·주차장", (1020, 1790)),
    ("남쪽 사거리", (-170, 3200)),
    ("남쪽 삼거리", (990, 3080)),
]
