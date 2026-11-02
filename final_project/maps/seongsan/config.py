"""사진 기준 성산 외곽/남쪽 확장 도로. 내부 연결로와 가상 식당은 삭제한 버전."""
import math

from .geometry import smooth_streets

BASE_SCALE = 0.543344
ROAD_CURVE_HANDLE_M = 24
ROAD_CURVE_STEP_M = 3
STRAIGHT_STREETS = {'minor_west', 'godeungeo_road', 'godeungeo_approach',
                    'coast_merge_approach', 'coast_south', 'minor_coast'}
GLOBAL_LENGTH_FACTOR = 0.47
NORTH_LENGTH_FACTOR = 0.90
NORTH_COMPRESSION_ANCHOR_Y = 858
EAST_ENTRY_REFERENCE_FACTOR = 0.65
MERGE_LENGTH_FACTOR = 0.80
MERGE_MARKING_FACTORS = {'upper': 0.95, 'lower': 0.80}
SOUTH_LENGTH_FACTOR = 0.15
SOUTH_COMPRESSION_START_Y = 1200
SCALE = BASE_SCALE * GLOBAL_LENGTH_FACTOR
REFERENCE_DISTANCE_M = 600
REFERENCE_EDGES = ['eomeong_road', 'east_sweep', 'main_sweep', 'east_main', 'east_main_north', 'godeungeo_road', 'godeungeo_approach']
ISLAND_RADIUS_PIXELS = 17 / 0.75
RING_CENTER = (155, 858)
RING_RADIUS_PIXELS = 34
LANE_WIDTH = 3.2
SPEED_LOCAL, SPEED_MAIN, SPEED_RING, SPEED_MERGE = 50/3.6, 60/3.6, 20/3.6, 30/3.6
SCHOOL_CONTROL_SPEED_KMH = 28.5
RING_LANES = 2
SOUTH_LANES_PER_DIRECTION = 2
TRAFFIC_PER_PORT, TRAFFIC_END, SEED = 100, 3600, 23
STEP_LENGTH, GUI_DELAY_MS = 0.2, 200
# Red-marked branches: top east -> west, lower right south -> north.
# The blue west side remains one lane in each direction (two lanes total).
ONE_WAY_STREETS = {'triangle_top': -1, 'triangle_east_leg': -1}
STRAIGHT_ONLY_CONNECTIONS = {
    # The fork divides the northbound lanes; southbound traffic keeps going south.
    'east_signal_approach': 'east_turn_pocket',
    'triangle_top_rev': 'minor_west_rev',
    'triangle_east_upper_rev': 'minor_coast_rev',
    'triangle_east_leg_rev': 'triangle_east_upper_rev',
    'main_sweep_rev': 'east_sweep_rev',
    'triangle_west_leg': 'coast_merge_approach',
}
RIGHT_ONLY_CONNECTIONS = {'minor_west_rev': 'main_sweep'}
ROAD_WIDTH_OVERRIDES = {e: 3.0 for e in ['minor_west', 'minor_coast', 'east_signal_approach', 'triangle_top', 'triangle_west_upper', 'triangle_east_upper', 'triangle_cross', 'triangle_west_leg', 'triangle_east_leg']}
# Begin slowing on the short approach before reaching the school-zone sign.
ROAD_SPEED_OVERRIDES = {'start_south': SCHOOL_CONTROL_SPEED_KMH / 3.6}
ROAD_STOP_OFFSETS = {edge: 4.5 for edge in
                     ['east_main','east_main_north_rev','east_signal_approach_rev','east_right_slip']}
ROAD_STOP_OFFSETS.update({'east_signal_approach_rev':1.5,'east_right_slip':12})
# Overrides use directed edge IDs, allowing a widening in just one direction.
ROAD_LANE_OVERRIDES = {'south_rb_approach_rev': 2, 'coast_merge_approach_rev': 1,
                       'east_turn_pocket_rev': 2,
                       'minor_west': 2, 'triangle_west_upper': 2, 'triangle_west_leg': 2}
NODE_OPTIONS = {
    'coast_lane_change': {'type': 'zipper', 'radius': 0, 'keepClear': False},
    'south_lane_change': {'radius': 0, 'keepClear': False},
    'east_signal_split': {'radius': 0, 'keepClear': False},
    'east_slip_split': {'type': 'unregulated', 'radius': 1, 'keepClear': False},
    'east_cross': {'radius': 1, 'keepClear': False},
}
ZIPPER_TAPER_LENGTH_M = 18
LANE_TAPERS = {
    'coast_lane_change': {'direction': (0, 1), 'southbound': 2, 'north_before': 2, 'north_after': 1, 'width': LANE_WIDTH},
    'south_lane_change': {'direction': (0, 1), 'southbound': 1, 'north_before': 1, 'north_after': 2, 'width': LANE_WIDTH},
    'east_signal_split': {'direction': (15.5, 45), 'southbound': 1, 'north_before': 1, 'north_after': 2, 'width': 3.0},
}
EXTRA_CONNECTIONS = [
    {'from': 'coast_south_rev', 'to': 'coast_merge_approach_rev', 'fromLane': lane, 'toLane': 0, 'visibility': 40}
    for lane in (0, 1)
]
LANE_CONNECTION_RULES = {
    'east_signal_approach_rev': [
        {'fromLane': 0, 'to': 'east_main_rev', 'toLane': 1},
    ],
    'east_turn_pocket_rev': [
        {'fromLane': 0, 'to': 'east_right_slip', 'toLane': 0},
        {'fromLane': 1, 'to': 'east_signal_approach_rev', 'toLane': 0},
    ],
    'east_right_slip': [
        {'fromLane': 0, 'to': 'godeungeo_road', 'toLane': 0, 'visibility': 45},
    ],
    'triangle_west_upper': [
        {'fromLane': 0, 'to': 'triangle_west_leg', 'toLane': 0},
        {'fromLane': 1, 'to': 'triangle_west_leg', 'toLane': 1},
        {'fromLane': 1, 'to': 'triangle_cross', 'toLane': 0},
    ],
}
CONTINUOUS_CENTERLINE_CONNECTIONS = [
    ('coast_merge_approach', 'coast_south'),
    ('south_rb_approach', 'south_straight'),
    ('east_turn_pocket', 'minor_coast'),
    ('minor_west', 'triangle_west_upper'),
    # Follow the two-way diagonal, not the edge of the one-way merging arm.
    ('triangle_cross', 'triangle_east_upper_rev'),
    ('triangle_east_upper_rev', 'minor_coast_rev'),
    ('east_sweep', 'main_sweep'),
    ('triangle_west_leg', 'coast_merge_approach'),
    ('east_main_north', 'godeungeo_road'),
]
SIGNAL_CONTROLLERS = {'signal_south':['south_cross']}
STOP_JUNCTIONS = ['bank', 'triangle_northeast', 'godeungeo']
NODE_OPTIONS.update({node:{**NODE_OPTIONS.get(node,{}),'type':'allway_stop'} for node in STOP_JUNCTIONS})
SIGNAL_TIMING = {'cycle_seconds':90, 'yellow_seconds':3, 'all_red_seconds':2}
SCHOOL_ZONE = {
    'speed_kmh':30, 'control_speed_kmh':SCHOOL_CONTROL_SPEED_KMH,
    'from':'범하다 부근', 'to':'보룡제과 앞',
    'edges':['school_approach','west_bank','west_lower'],
    'sign_nodes':['school_north','west_lower'], 'approach_edges':[],
}
SCHOOL_NORTH_APPROACH_EDGES = ['start_south','hamabo_return']
HIDDEN_LANE_ARROWS = ['east_signal_approach_0']
CUSTOM_ARROW_EDGES = {**{f'ring_{i}': 'straight' for i in range(4)},
                      'west_lower': 'straight', 'boundary_sw': 'straight',
                      'south_rb_approach_rev': 'straight', 'eomeong_road_rev': 'straight'}
CUSTOM_LANE_ARROWS = {'east_turn_pocket_rev_0': 'right', 'east_turn_pocket_rev_1': 'left'}
TRAFFIC_VIA_EDGES = {
    ('port_sw','port_se'):['minor_west','triangle_west_upper','triangle_west_leg'],
    ('port_se','port_sw'):['triangle_west_leg_rev','triangle_west_upper_rev','minor_west_rev'],
    ('port_south','port_ne'):['triangle_east_leg_rev','minor_coast_rev'],
    ('port_ne','port_south'):['minor_coast','triangle_east_upper','triangle_cross_rev','triangle_west_leg'],
    ('port_ne','port_sw'):['minor_coast','triangle_top_rev','minor_west_rev'],
    ('port_nw','port_ne'):['minor_west','triangle_west_upper','triangle_cross','triangle_east_upper_rev','minor_coast_rev'],
    ('port_ne','port_nw'):['minor_coast','triangle_east_upper','triangle_cross_rev','triangle_west_upper_rev','minor_west_rev'],
    ('port_se','port_nw'):['triangle_east_leg_rev','east_turn_pocket_rev','east_signal_approach_rev'],
}
POINTS = {
    'hamabo':(64,263), 'school_north':(95,355), 'west_cross':(120,416),
    'bank':(145,511), 'west_lower':(155,784),
    'rb_n':(155,824), 'rb_w':(121,858), 'rb_s':(155,892), 'rb_e':(189,858),
    'eomeong':(340,819), 'triangle_west':(385,811.04), 'east_curve':(700,742),
    'triangle_northeast':(900,571.07), 'east_cross':(953,520), 'inner_exit':(1103,372),
    'godeungeo':(1160,315), 'coast_bend':(1002,138), 'coast_top':(601,24),
    'north_cross':(334,156), 'northwest':(51,218),
    'merge_west':(690,955), 'merge_ne':(776,961),
    'merge_cross_west':(707,1030), 'merge_cross_east':(753,998),
    'triangle_south':(720,1130),
    'south_lane_change':(155,1040), 'coast_lane_change':(720,1350),
    'east_signal_split':(852,755),
    'east_slip_split':(886,689),
    'south_cross':(155,3100), 'south_t':(720,3100),
    'port_north':(5,20), 'port_nw':(-130,245), 'port_sw':(-85,881),
    'port_south':(155,3520), 'port_se':(1390,3100), 'port_ne':(1340,205),
    'port_south_west':(-230,3100),
}
def cubic_points(start, control1, control2, finish, samples=32):
    return [tuple((1-t)**3*start[i]+3*(1-t)**2*t*control1[i]+3*(1-t)*t*t*control2[i]+t**3*finish[i]
                  for i in (0,1)) for t in (j/samples for j in range(samples+1))]


# (ID, 시작점, 끝점, 방향별 차로 수, m/s, 중간 형상점). 기본은 양방향.
STREETS = [
    ('start_south','hamabo','school_north',1,SPEED_LOCAL,[(88,330)]),
    ('school_approach','school_north','west_cross',1,SPEED_LOCAL,[]),
    ('west_bank','west_cross','bank',1,SPEED_LOCAL,[]),
    ('west_lower','bank','rb_n',1,SPEED_LOCAL,[(154,620),(155,784)]),
    ('eomeong_road','rb_e','eomeong',2,SPEED_MAIN,[(240,846)]),
    ('east_sweep','eomeong','triangle_west',2,SPEED_MAIN,[]),
    ('main_sweep','triangle_west','east_curve',2,SPEED_MAIN,[(470,796),(600,774)]),
    ('east_main','east_curve','triangle_northeast',2,SPEED_MAIN,[(767,700),(843,626)]),
    ('east_main_north','triangle_northeast','east_cross',2,SPEED_MAIN,[]),
    ('godeungeo_road','east_cross','inner_exit',2,SPEED_MAIN,[(971,502),(1050,423)]),
    ('godeungeo_approach','inner_exit','godeungeo',2,SPEED_MAIN,[]),
    ('coast_turn','godeungeo','coast_bend',1,SPEED_LOCAL,[(1115,268),(1067,202)]),
    ('coast_north','coast_bend','coast_top',1,SPEED_LOCAL,[(946,111),(824,78),(712,50)]),
    ('north_return','coast_top','north_cross',1,SPEED_LOCAL,[(520,55),(427,102)]),
    ('northwest_return','north_cross','northwest',1,SPEED_LOCAL,[(275,175),(177,200)]),
    ('hamabo_return','northwest','hamabo',1,SPEED_LOCAL,[]),
    ('minor_west','triangle_west','merge_west',1,SPEED_LOCAL,[(430,850),(510,858),(600,905)]),
    ('east_signal_approach','triangle_northeast','east_slip_split',1,SPEED_LOCAL,[]),
    ('east_turn_pocket','east_slip_split','east_signal_split',1,SPEED_LOCAL,[(866,710)]),
    ('east_right_slip','east_slip_split','east_cross',1,SPEED_MERGE,
     cubic_points(POINTS['east_slip_split'],(918,608),(942,531),POINTS['east_cross'])[1:-1]),
    ('minor_coast','east_signal_split','merge_ne',1,SPEED_LOCAL,[(810,875)]),
    ('triangle_top','merge_west','merge_ne',1,SPEED_MERGE,[(733,969),(754,969)]),
    ('triangle_west_upper','merge_west','merge_cross_west',1,SPEED_MERGE,[(695,970)]),
    ('triangle_east_upper','merge_ne','merge_cross_east',1,SPEED_MERGE,[]),
    ('triangle_cross','merge_cross_west','merge_cross_east',1,SPEED_MERGE,[(742,1009)]),
    ('triangle_west_leg','merge_cross_west','triangle_south',1,SPEED_MERGE,[]),
    ('triangle_east_leg','merge_cross_east','triangle_south',1,SPEED_MERGE,[(734,1037),(725,1070)]),
    ('south_rb_approach','rb_s','south_lane_change',1,SPEED_MAIN,[]),
    ('south_straight','south_lane_change','south_cross',1,SPEED_MAIN,[]),
    ('south_connector','south_cross','south_t',SOUTH_LANES_PER_DIRECTION,SPEED_MAIN,[]),
    ('coast_merge_approach','triangle_south','coast_lane_change',SOUTH_LANES_PER_DIRECTION,SPEED_MAIN,[(720,1200)]),
    ('coast_south','coast_lane_change','south_t',SOUTH_LANES_PER_DIRECTION,SPEED_MAIN,
     [(720,1550),(750,1800),(800,2000),(850,2350),(880,2600)]),
    ('boundary_north','port_north','northwest',1,SPEED_LOCAL,[]),
    ('boundary_nw','port_nw','northwest',1,SPEED_LOCAL,[]),
    ('boundary_sw','port_sw','rb_w',2,SPEED_LOCAL,[]),
    ('boundary_south','port_south','south_cross',1,SPEED_MAIN,[]),
    ('boundary_se','port_se','south_t',SOUTH_LANES_PER_DIRECTION,SPEED_MAIN,[]),
    ('boundary_ne','port_ne','godeungeo',2,SPEED_MAIN,[]),
    ('boundary_south_west','port_south_west','south_cross',SOUTH_LANES_PER_DIRECTION,SPEED_MAIN,[]),
]
SOLID_CENTERLINE_STREETS = [street[0] for street in STREETS]
ONE_WAY_STREETS['east_right_slip'] = 1
ROAD_WIDTH_OVERRIDES.update({'east_turn_pocket': 3.0, 'east_right_slip': 3.0})
RING_NODES = ['rb_n','rb_w','rb_s','rb_e']
RING_ANGLES = [90,180,270,360]
ORIGINAL_EGO_LOOP = [
    'start_south','school_approach','west_bank','west_lower',
    'ring_0','ring_1','ring_2','eomeong_road','east_sweep','main_sweep','east_main',
    'east_main_north','godeungeo_road','godeungeo_approach','coast_turn','coast_north',
    'north_return','northwest_return','hamabo_return',
]
EGO_LOOP = [
    'start_south','school_approach','west_bank','west_lower',
    'ring_0','ring_1','south_rb_approach','south_straight','south_connector','coast_south_rev','coast_merge_approach_rev',
    'triangle_east_leg_rev','triangle_east_upper_rev','minor_coast_rev','east_turn_pocket_rev','east_right_slip','godeungeo_road',
    'godeungeo_approach','coast_turn','coast_north','north_return','northwest_return','hamabo_return',
]
DEFAULT_EGO_ROUTE = 'outermost'
EGO_START_ANCHOR_NODE = 'northwest'
EGO_START_ANCHOR_EDGE = 'northwest_return'
EGO_START_DISTANCE_M = 100
EGO_ROUTES = {'outermost':EGO_LOOP, 'expanded':EGO_LOOP, 'original':ORIGINAL_EGO_LOOP}
FEATURE_EDGE_GROUPS = {
    'south_straight':['south_straight','south_straight_rev','south_rb_approach','south_rb_approach_rev'],
    'roundabout_south_widening':['south_rb_approach_rev'],
    'south_connector':['south_connector','south_connector_rev'],
    'southern_coast':['coast_south','coast_south_rev','coast_merge_approach','coast_merge_approach_rev'],
    'coast_northbound_narrowing':['coast_merge_approach_rev'],
    'main_curve':['main_sweep','main_sweep_rev','east_main','east_main_rev'],
    'minor_west':['minor_west','minor_west_rev'],
    'minor_coast':['minor_coast','minor_coast_rev','east_turn_pocket','east_turn_pocket_rev','east_signal_approach','east_signal_approach_rev'],
    'east_signal_turn_pocket':['east_turn_pocket_rev','east_signal_approach_rev'],
    'east_left_signal':['east_signal_approach_rev'],
    'east_right_yield':['east_right_slip'],
    'small_merge':[edge + suffix for edge in ['triangle_top','triangle_west_upper','triangle_east_upper','triangle_cross','triangle_west_leg','triangle_east_leg'] for suffix in (('_rev',) if ONE_WAY_STREETS.get(edge) == -1 else ('','_rev'))],
    'merge_diagonal':['triangle_cross','triangle_cross_rev'],
    'merge_top_one_way':['triangle_top_rev'],
    'merge_lower_right_one_way':['triangle_east_leg_rev'],
    'triangle_southbound_two_lanes':['minor_west','triangle_west_upper','triangle_west_leg'],
}
FIELD_POLYGON = [(455,830),(845,650),(728,935),(700,935)]
# Latest close-up photo: broad upper painted triangle, diagonal lane, long lower gore.
# Photo (110,95) corresponds to (690,955); local relative scale is 0.2.
TRAFFIC_ISLAND = [(704,970),(757,979),(714,1010)]
LOWER_TRAFFIC_ISLAND = [(715,1037),(731,1023),(719,1067)]
MERGE_PAVEMENT = [(690,955),(733,969),(754,969),(776,961),
                  (753,998),(734,1037),(725,1070),(720,1130),(707,1030),(695,970)]
LABELS = [
    ('하마보 앞 · 기존 출발 구간',(87,263)),
    ('어린이보호구역 30km/h',(180,570)),
    ('고성리 로터리 · 반시계 2차로',(-70,970)),
    ('일출로 · 연속 왕복 4차로·60km/h',(385,720)),
    ('삼각형 왼쪽 · 남하 2차로 / 북상 1차로',(425,920)),
    ('섭지코지 방면 연결도로',(925,805)),
    ('대각선 연결로 · 상하 두 도류구역',(800,1090)),
    ('상단: 오른쪽→왼쪽 / 우하단: 아래→위 · 편도 1차로',(790,1160)),
    ('성산고등어 쌀밥 방면',(1004,364)),
    ('좌회전 → 북쪽 해안 구간',(817,32)),
    ('로터리–사거리 왕복 2차로 / 로터리 북상 진입만 2차로',(215,1560)),
    ('오른쪽 왕복 4차로 / 삼각형 앞 북상 1차로',(1000,2350)),
    ('로터리 서쪽 왕복 4차로',(-210,790)),
    ('남쪽 사거리',(-170,3200)), ('남쪽 삼거리',(990,3080)),
]


def compact_point(point):
    """Compact the northern loop and the long southern extension."""
    x, y = point
    if y < NORTH_COMPRESSION_ANCHOR_Y:
        y = NORTH_COMPRESSION_ANCHOR_Y + (y-NORTH_COMPRESSION_ANCHOR_Y)*NORTH_LENGTH_FACTOR
    elif y > SOUTH_COMPRESSION_START_Y:
        y = SOUTH_COMPRESSION_START_Y + (y - SOUTH_COMPRESSION_START_Y) * SOUTH_LENGTH_FACTOR
    return x, y


def preserve_junction_size(point, center, length_factor=1):
    """Apply a junction's own physical scale independently of connecting roads."""
    anchor = compact_point(center)
    return tuple(a + (p-c)*length_factor/GLOBAL_LENGTH_FACTOR for p,c,a in zip(point,center,anchor))


def inset_marking(points, factor):
    """Leave clearance for full-width turning lanes in the smaller junction."""
    center = tuple(sum(p[i] for p in points) / len(points) for i in (0, 1))
    return [tuple(c + (p - c) * factor for p, c in zip(point, center)) for point in points]


# Source coordinates above stay editable. Apply layout changes exactly once on import.
MERGE_CENTER = (735, 1030)
MERGE_STREETS = {'triangle_top', 'triangle_west_upper', 'triangle_east_upper',
                 'triangle_cross', 'triangle_west_leg', 'triangle_east_leg'}
MERGE_NODES = {'merge_west', 'merge_ne', 'merge_cross_west', 'merge_cross_east', 'triangle_south'}
EAST_ENTRY_CENTER = (910,620)
EAST_ENTRY_NODES = {'triangle_northeast','east_cross','east_slip_split','east_signal_split'}
EAST_ENTRY_STREETS = {'east_signal_approach','east_turn_pocket','east_right_slip','east_main_north'}
POINTS = {
    name: (preserve_junction_size(point,RING_CENTER) if name in RING_NODES
           else preserve_junction_size(point,RING_CENTER,EAST_ENTRY_REFERENCE_FACTOR) if name=='west_lower'
           else preserve_junction_size(compact_point(point),MERGE_CENTER,EAST_ENTRY_REFERENCE_FACTOR) if name=='coast_lane_change'
           else preserve_junction_size(point,MERGE_CENTER,MERGE_LENGTH_FACTOR) if name in MERGE_NODES
           else preserve_junction_size(point,EAST_ENTRY_CENTER,EAST_ENTRY_REFERENCE_FACTOR) if name in EAST_ENTRY_NODES
           else compact_point(point))
    for name, point in POINTS.items()
}
# Align the entire southern coast road and its bottom T-junction vertically.
POINTS['coast_lane_change'] = (POINTS['triangle_south'][0], POINTS['coast_lane_change'][1])
POINTS['south_t'] = (POINTS['triangle_south'][0], POINTS['south_t'][1])
# Photo-based entry layout, in physical metres after the map compression.
# A straight mainline, a short two-lane fork, and a tangential right-turn slip.
_entry_junction = POINTS['triangle_northeast']
_main_delta = tuple(POINTS['godeungeo'][i]-_entry_junction[i] for i in (0,1))
_main_length = math.hypot(*_main_delta)
_main_axis = tuple(v/_main_length for v in _main_delta)
# A defined mainline mouth lets the side road reach the diagonal stop line,
# instead of SUMO cutting it off at the start of its turning curve.
_junction_outline = [(_entry_junction[0]+(_main_axis[0]*along-_main_axis[1]*across)/SCALE,
                      _entry_junction[1]+(_main_axis[1]*along+_main_axis[0]*across)/SCALE)
                     for along,across in [(-13,-6.4),(9,-6.4),(9,6.4),(-13,6.4)]]
NODE_OPTIONS['triangle_northeast']['shape'] = ' '.join(f'{x*SCALE:.3f},{(929-y)*SCALE:.3f}'
                                                    for x,y in _junction_outline)
def entry_point(along_m=0, south_m=0):
    return (_entry_junction[0]+_main_axis[0]*along_m/SCALE,
            _entry_junction[1]+(_main_axis[1]*along_m+south_m)/SCALE)

POINTS['east_cross'] = entry_point(22)
POINTS['inner_exit'] = tuple(POINTS['east_cross'][i] +
                            (POINTS['godeungeo'][i]-POINTS['east_cross'][i])*2/3 for i in (0,1))
POINTS['east_slip_split'] = entry_point(south_m=25)
EAST_ENTRY_REFUGE_SEED = (_entry_junction[0]+5/SCALE,_entry_junction[1]+9/SCALE)
# Use one axis for the entry road, widening, and turn pocket. Keep a separate
# signal approach beyond the fork so its stop bar and arrow cannot overlap.
entry_start, entry_finish = POINTS['merge_ne'], POINTS['east_slip_split']
POINTS['east_signal_split'] = tuple(entry_start[i]+(entry_finish[i]-entry_start[i])*.46 for i in (0,1))
LANE_TAPERS['east_signal_split']['direction'] = (entry_finish[0]-entry_start[0],
                                               entry_start[1]-entry_finish[1])
STREETS = [
    (eid, source, target, lanes, speed,
     [(preserve_junction_size(p,MERGE_CENTER,MERGE_LENGTH_FACTOR) if eid in MERGE_STREETS
       else preserve_junction_size(p,EAST_ENTRY_CENTER,EAST_ENTRY_REFERENCE_FACTOR) if eid in EAST_ENTRY_STREETS or (eid=='godeungeo_road' and p==(971,502))
       else preserve_junction_size(p,RING_CENTER,EAST_ENTRY_REFERENCE_FACTOR) if eid=='west_lower' and p==(155,784)
       else compact_point(p))
      for p in mids])
    for eid, source, target, lanes, speed, mids in STREETS
]
STREETS = [(eid,source,target,lanes,speed,[] if eid in STRAIGHT_STREETS else mids)
           for eid,source,target,lanes,speed,mids in STREETS]
_pocket_start = POINTS['east_signal_split']
_pocket_finish = POINTS['east_slip_split']
_pocket_delta = tuple(_pocket_finish[i]-_pocket_start[i] for i in (0,1))
_pocket_length = math.hypot(*_pocket_delta)
_pocket_unit = tuple(v/_pocket_length for v in _pocket_delta)
OUTER_ENTRY_STRAIGHT_LENGTH_M = _pocket_length*SCALE+10-ZIPPER_TAPER_LENGTH_M/2
# A single straight axis for the two-lane pocket and outer right-turn lane.
# Offset the one-way reference by one lane, perpendicular to that same axis.
_pocket_right = (-_pocket_unit[1],_pocket_unit[0])
_slip_start = tuple(_pocket_finish[i]+_pocket_right[i]*3/SCALE for i in (0,1))
_slip_finish = (POINTS['east_cross'][0]-_main_axis[1]*3.3/SCALE,
                POINTS['east_cross'][1]+_main_axis[0]*3.3/SCALE)
ROAD_SHAPE_ENDPOINTS = {'east_right_slip': (_slip_start,_slip_finish)}
# Keep most of the slip straight. Round only the approach to a final straight
# parallel to the mainline, so the yield marking lies on a constant-width lane.
_slip_tail = tuple(_slip_finish[i]-_main_axis[i]*8/SCALE for i in (0,1))
_slip_line_end = tuple(_slip_start[i]+(_slip_tail[i]-_slip_start[i])*.65 for i in (0,1))
_slip_curve_control1 = tuple(_slip_line_end[i]+(_slip_tail[i]-_slip_line_end[i])*.4 for i in (0,1))
_slip_curve_control2 = tuple(_slip_tail[i]-_main_axis[i]*3/SCALE for i in (0,1))
_slip_start_join = tuple(_slip_start[i]+_pocket_unit[i]*10/SCALE for i in (0,1))
_slip_start_control1 = tuple(_slip_start_join[i]+_pocket_unit[i]*4/SCALE for i in (0,1))
_slip_start_control2 = tuple(_slip_line_end[i]-(_slip_tail[i]-_slip_line_end[i])*.3 for i in (0,1))
_slip_path = [_slip_start_join,
              *cubic_points(_slip_start_join,_slip_start_control1,_slip_start_control2,_slip_line_end,samples=8)[1:],
              *cubic_points(_slip_line_end,_slip_curve_control1,_slip_curve_control2,_slip_tail,samples=8)[1:],
              tuple((_slip_tail[i]+_slip_finish[i])/2 for i in (0,1))]
_approach_finish = tuple(POINTS['triangle_northeast'][i]-_main_axis[i]*5/SCALE for i in (0,1))
ROAD_SHAPE_ENDPOINTS['east_signal_approach'] = (_approach_finish,_pocket_finish)
_approach_control1 = tuple(_pocket_finish[i]+_pocket_unit[i]*5/SCALE for i in (0,1))
_approach_tail = (_approach_finish[0]-_main_axis[1]*9/SCALE,
                  _approach_finish[1]+_main_axis[0]*9/SCALE)
_approach_control2 = (_approach_tail[0]-_main_axis[1]*4/SCALE,
                      _approach_tail[1]+_main_axis[0]*4/SCALE)
_approach_path = list(reversed(
    [_pocket_finish,*cubic_points(_pocket_finish,_approach_control1,_approach_control2,_approach_tail,samples=20)[1:],
      _approach_finish]))[1:-1]
STREETS = [(eid,source,target,lanes,speed,
            _slip_path
            if eid=='east_right_slip' else
            []
            if eid=='east_turn_pocket' else
            _approach_path
            if eid=='east_signal_approach' else
            [entry_point(-22)] if eid=='east_main' else mids)
           for eid,source,target,lanes,speed,mids in STREETS]
STREETS = smooth_streets(POINTS, STREETS, SCALE, LANE_TAPERS,
                         protected=MERGE_STREETS,
                         already_curved={'east_right_slip','east_turn_pocket','east_signal_approach'} | STRAIGHT_STREETS,
                         handle_m=ROAD_CURVE_HANDLE_M, step_m=ROAD_CURVE_STEP_M)
FIELD_POLYGON = [compact_point(p) for p in FIELD_POLYGON]
TRAFFIC_ISLAND = [preserve_junction_size(p, MERGE_CENTER, MERGE_LENGTH_FACTOR) for p in TRAFFIC_ISLAND]
LOWER_TRAFFIC_ISLAND = [preserve_junction_size(p, MERGE_CENTER, MERGE_LENGTH_FACTOR) for p in LOWER_TRAFFIC_ISLAND]
TRAFFIC_ISLAND = inset_marking(TRAFFIC_ISLAND, MERGE_MARKING_FACTORS['upper'])
LOWER_TRAFFIC_ISLAND = inset_marking(LOWER_TRAFFIC_ISLAND, MERGE_MARKING_FACTORS['lower'])
MERGE_PAVEMENT = [preserve_junction_size(p, MERGE_CENTER, MERGE_LENGTH_FACTOR) for p in MERGE_PAVEMENT]
LABELS = [(label, compact_point(p)) for label, p in LABELS]
south_label_positions = {
    '남쪽 사거리': (POINTS['south_cross'][0] - 290, POINTS['south_cross'][1] + 75),
    '남쪽 삼거리': (POINTS['south_t'][0] + 40, POINTS['south_t'][1] + 70),
}
LABELS = [(label, south_label_positions.get(label, point)) for label, point in LABELS]
RING_RADIUS_PIXELS /= GLOBAL_LENGTH_FACTOR
ISLAND_RADIUS_PIXELS /= GLOBAL_LENGTH_FACTOR
