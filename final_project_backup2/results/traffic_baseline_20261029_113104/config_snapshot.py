"""사진 기준 성산 외곽/남쪽 확장 도로. 내부 연결로와 가상 식당은 삭제한 버전."""
SCALE = 0.543344
REFERENCE_DISTANCE_M = 600
REFERENCE_EDGES = ['eomeong_road', 'east_sweep', 'main_sweep', 'east_main', 'east_main_north', 'godeungeo_road', 'godeungeo_approach']
ISLAND_RADIUS_PIXELS = 17 / 0.75
LANE_WIDTH = 3.2
SPEED_LOCAL, SPEED_MAIN, SPEED_RING, SPEED_MERGE = 50/3.6, 60/3.6, 20/3.6, 30/3.6
RING_LANES = 2
SOUTH_LANES_PER_DIRECTION = 2
TRAFFIC_PER_PORT, TRAFFIC_END, SEED = 100, 3600, 23
STEP_LENGTH, GUI_DELAY_MS = 0.2, 200
ONE_WAY_STREETS = {}
STRAIGHT_ONLY_CONNECTIONS = {}
ROAD_WIDTH_OVERRIDES = {e: 3.0 for e in ['minor_west', 'minor_coast', 'triangle_top', 'triangle_west_leg', 'triangle_east_leg']}
ROAD_SPEED_OVERRIDES = {}
# 외곽의 기존 신호 위치 유지. 내부 도로 연결은 새 도로망에 맞춰 재생성.
SIGNAL_CONTROLLERS = {'signal_bank':['bank'], 'signal_east':['east_cross'], 'signal_godeungeo':['godeungeo']}
SIGNAL_TIMING = {'cycle_seconds':90, 'yellow_seconds':3, 'all_red_seconds':2}
SCHOOL_ZONE = {
    'speed_kmh':30, 'control_speed_kmh':29,
    'from':'범하다 부근', 'to':'보룡제과 앞',
    'edges':['school_approach','west_bank','west_lower'],
    'sign_nodes':['school_north','west_lower'], 'approach_edges':['roundabout_approach'],
}
SCHOOL_NORTH_APPROACH_EDGES = ['start_south','hamabo_return']
TRAFFIC_VIA_EDGES = {
    ('port_sw','port_se'):['minor_west','triangle_west_leg'],
    ('port_se','port_sw'):['triangle_west_leg_rev','minor_west_rev'],
    ('port_south','port_ne'):['triangle_east_leg_rev','minor_coast_rev'],
    ('port_ne','port_south'):['minor_coast','triangle_east_leg'],
    ('port_north','port_se'):['minor_west','triangle_top','triangle_east_leg'],
}
POINTS = {
    'hamabo':(64,263), 'school_north':(95,355), 'west_cross':(120,416),
    'bank':(145,511), 'west_lower':(155,784),
    'rb_n':(155,824), 'rb_w':(121,858), 'rb_s':(155,892), 'rb_e':(189,858),
    'eomeong':(340,819), 'triangle_west':(385,811.04), 'east_curve':(700,742),
    'triangle_northeast':(900,571.07), 'east_cross':(953,520), 'inner_exit':(1103,372),
    'godeungeo':(1160,315), 'coast_bend':(1002,138), 'coast_top':(601,24),
    'north_cross':(334,156), 'northwest':(51,218),
    'merge_west':(690,955), 'merge_ne':(750,965), 'triangle_south':(710,1030),
    'south_cross':(155,3100), 'south_t':(950,3000),
    'port_north':(5,20), 'port_nw':(-130,245), 'port_sw':(-85,881),
    'port_south':(155,3520), 'port_se':(1390,3000), 'port_ne':(1340,205),
    'port_south_west':(-230,3100),
}
# (ID, 시작점, 끝점, 방향별 차로 수, m/s, 중간 형상점). 기본은 양방향.
STREETS = [
    ('start_south','hamabo','school_north',1,SPEED_LOCAL,[(88,330)]),
    ('school_approach','school_north','west_cross',1,SPEED_LOCAL,[]),
    ('west_bank','west_cross','bank',1,SPEED_LOCAL,[]),
    ('west_lower','bank','west_lower',1,SPEED_LOCAL,[(154,620)]),
    ('roundabout_approach','west_lower','rb_n',1,SPEED_LOCAL,[]),
    ('eomeong_road','rb_e','eomeong',2,SPEED_MAIN,[(240,846)]),
    ('east_sweep','eomeong','triangle_west',2,SPEED_MAIN,[]),
    ('main_sweep','triangle_west','east_curve',2,SPEED_MAIN,[(470,796),(600,774)]),
    ('east_main','east_curve','triangle_northeast',2,SPEED_MAIN,[(767,700),(843,626)]),
    ('east_main_north','triangle_northeast','east_cross',2,SPEED_MAIN,[]),
    ('godeungeo_road','east_cross','inner_exit',2,SPEED_MAIN,[(1050,423)]),
    ('godeungeo_approach','inner_exit','godeungeo',2,SPEED_MAIN,[]),
    ('coast_turn','godeungeo','coast_bend',1,SPEED_LOCAL,[(1115,268),(1067,202)]),
    ('coast_north','coast_bend','coast_top',1,SPEED_LOCAL,[(946,111),(824,78),(712,50)]),
    ('north_return','coast_top','north_cross',1,SPEED_LOCAL,[(520,55),(427,102)]),
    ('northwest_return','north_cross','northwest',1,SPEED_LOCAL,[(275,175),(177,200)]),
    ('hamabo_return','northwest','hamabo',1,SPEED_LOCAL,[]),
    ('minor_west','triangle_west','merge_west',1,SPEED_LOCAL,[(430,850),(510,858),(600,905)]),
    ('minor_coast','triangle_northeast','merge_ne',1,SPEED_LOCAL,[(866,675),(815,795),(765,915)]),
    ('triangle_top','merge_west','merge_ne',1,SPEED_MERGE,[]),
    ('triangle_west_leg','merge_west','triangle_south',1,SPEED_MERGE,[(695,982),(703,1015)]),
    ('triangle_east_leg','merge_ne','triangle_south',1,SPEED_MERGE,[(735,992),(717,1020)]),
    ('south_straight','rb_s','south_cross',SOUTH_LANES_PER_DIRECTION,SPEED_MAIN,[]),
    ('south_connector','south_cross','south_t',SOUTH_LANES_PER_DIRECTION,SPEED_MAIN,[]),
    ('coast_south','triangle_south','south_t',SOUTH_LANES_PER_DIRECTION,SPEED_MAIN,
     [(720,1200),(750,1550),(800,1800),(815,1900),(850,2250),(880,2600)]),
    ('boundary_north','port_north','northwest',1,SPEED_LOCAL,[]),
    ('boundary_nw','port_nw','northwest',1,SPEED_LOCAL,[]),
    ('boundary_sw','port_sw','rb_w',1,SPEED_LOCAL,[]),
    ('boundary_south','port_south','south_cross',SOUTH_LANES_PER_DIRECTION,SPEED_MAIN,[]),
    ('boundary_se','port_se','south_t',SOUTH_LANES_PER_DIRECTION,SPEED_MAIN,[]),
    ('boundary_ne','port_ne','godeungeo',2,SPEED_MAIN,[]),
    ('boundary_south_west','port_south_west','south_cross',SOUTH_LANES_PER_DIRECTION,SPEED_MAIN,[]),
]
RING_NODES = ['rb_n','rb_w','rb_s','rb_e']
RING_ANGLES = [90,180,270,360]
ORIGINAL_EGO_LOOP = [
    'start_south','school_approach','west_bank','west_lower','roundabout_approach',
    'ring_0','ring_1','ring_2','eomeong_road','east_sweep','main_sweep','east_main',
    'east_main_north','godeungeo_road','godeungeo_approach','coast_turn','coast_north',
    'north_return','northwest_return','hamabo_return',
]
EGO_LOOP = [
    'start_south','school_approach','west_bank','west_lower','roundabout_approach',
    'ring_0','ring_1','south_straight','south_connector','coast_south_rev',
    'triangle_east_leg_rev','minor_coast_rev','east_main_north','godeungeo_road',
    'godeungeo_approach','coast_turn','coast_north','north_return','northwest_return','hamabo_return',
]
EGO_ROUTES = {'expanded':EGO_LOOP, 'original':ORIGINAL_EGO_LOOP}
FEATURE_EDGE_GROUPS = {
    'south_straight':['south_straight','south_straight_rev'],
    'south_connector':['south_connector','south_connector_rev'],
    'southern_coast':['coast_south','coast_south_rev'],
    'main_curve':['main_sweep','main_sweep_rev','east_main','east_main_rev'],
    'minor_west':['minor_west','minor_west_rev'], 'minor_coast':['minor_coast','minor_coast_rev'],
    'small_merge':['triangle_top','triangle_top_rev','triangle_west_leg','triangle_west_leg_rev','triangle_east_leg','triangle_east_leg_rev'],
}
FIELD_POLYGON = [(455,830),(845,650),(728,935),(700,935)]
TRAFFIC_ISLAND = [(705,975),(733,981),(714,1006)]
LABELS = [
    ('출발 / 복귀 · 하마보 앞',(87,263)),
    ('어린이보호구역 30km/h',(180,570)),
    ('고성리 로터리 · 반시계 2차로',(-70,970)),
    ('일출로 · 연속 왕복 4차로·60km/h',(385,720)),
    ('좁은 연결도로 · 왕복 2차로',(425,920)),
    ('섭지코지 방면 연결도로',(925,805)),
    ('작은 도류구역 · 노란 빗금/유도봉',(780,1060)),
    ('성산고등어 쌀밥 방면',(1004,364)),
    ('좌회전 → 북쪽 해안 구간',(817,32)),
    ('로터리–사거리 직선 / 왕복 4차로·60km/h',(215,1560)),
    ('오른쪽 도로 / 왕복 4차로·60km/h',(1000,2350)),
    ('남쪽 사거리',(-170,3200)), ('남쪽 삼거리',(990,3080)),
]
