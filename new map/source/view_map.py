"""성산 고성리 맵 생성 / 미리보기 / 차량 주행 검증.

python view_map.py                 SUMO GUI (▶를 눌러 재생)
python view_map.py --netedit        도로 편집기
python view_map.py --check          완주 및 교통 검증
python view_map.py --build-only     파일 생성만
"""
import argparse
import csv
import json
import math
import subprocess
from collections import Counter

from sumolib import checkBinary
from sumolib.net import readNet
from maps.seongsan.builder import build
from maps.seongsan import config as map_config


def check(config, seconds):
    import traci
    import traci.constants as tc
    traci.start([checkBinary("sumo"), "-c", str(config), "--no-step-log", "true",
                 "--collision.action", "warn", "--collision.check-junctions", "true",
                 "--collision.mingap-factor", "0"])
    departed = set()
    arrived = set()
    visited = []
    ego_route_index = -1
    ego_route_changed = False
    teleports = collisions = 0
    ego_arrival = None
    ego_departure = None
    vehicle_routes = {}
    vehicle_route_edges = {}
    previous_route_index = {}
    departed_routes = Counter()
    arrived_routes = Counter()
    school_roads = {eid + suffix for eid in map_config.SCHOOL_ZONE["edges"] for suffix in ("", "_rev")}
    school_max_speed = 0.0
    school_samples = 0
    red_stopped = {tls: set() for tls in map_config.SIGNAL_CONTROLLERS}
    net = readNet(str(config.with_name('seongsan.net.xml')))
    stop_edges = {edge.getID():node for node in map_config.STOP_JUNCTIONS
                  for edge in net.getNode(node).getIncoming() if not edge.getID().startswith(':')}
    lane_lengths = {lane.getID():lane.getLength() for edge in net.getEdges() for lane in edge.getLanes()}
    stop_line_stopped = {node:set() for node in map_config.STOP_JUNCTIONS}
    approach_max = {}
    school_entry = {}
    ego_profile = []
    school_max_by_edge = {}
    school_samples_by_edge = Counter()
    previous_road = {}
    straight_passages = Counter()
    right_only_passages = Counter()
    movement_limits = {**map_config.STRAIGHT_ONLY_CONNECTIONS, **map_config.RIGHT_ONLY_CONNECTIONS}
    forbidden_turns = []
    feature_vehicles = {name: set() for name in map_config.FEATURE_EDGE_GROUPS}
    feature_groups_by_edge = {}
    for name, edges in map_config.FEATURE_EDGE_GROUPS.items():
        for edge in edges:
            feature_groups_by_edge.setdefault(edge, []).append(name)
    try:
        while traci.simulation.getTime() < seconds:
            traci.simulationStep()
            new_vehicles = traci.simulation.getDepartedIDList()
            departed.update(new_vehicles)
            for vid in new_vehicles:
                vehicle_routes[vid] = traci.vehicle.getRouteID(vid)
                vehicle_route_edges[vid] = traci.vehicle.getRoute(vid)
                departed_routes[vehicle_routes[vid]] += 1
                traci.vehicle.subscribe(vid, [tc.VAR_SPEED, tc.VAR_ROAD_ID, tc.VAR_NEXT_TLS,
                                             tc.VAR_LANEPOSITION, tc.VAR_ROUTE_INDEX, tc.VAR_LANE_ID])
                if vid == 'ego':
                    anchor_lane = net.getEdge(map_config.EGO_START_ANCHOR_EDGE).getLane(0)
                    gap = math.dist(anchor_lane.getShape()[-1],net.getNode(map_config.EGO_START_ANCHOR_NODE).getCoord())
                    distance = traci.vehicle.getDrivingDistance(vid,map_config.EGO_START_ANCHOR_EDGE,anchor_lane.getLength())+gap
                    ego_departure = {'edge':traci.vehicle.getRoadID(vid),'lane':traci.vehicle.getLaneIndex(vid),
                                     'position_m':traci.vehicle.getLanePosition(vid),'distance_to_junction_m':distance}
            for vid, observation in traci.vehicle.getAllSubscriptionResults().items():
                speed = observation[tc.VAR_SPEED]
                road = observation[tc.VAR_ROAD_ID]
                route_index = observation[tc.VAR_ROUTE_INDEX]
                for traversed in vehicle_route_edges[vid][previous_route_index.get(vid, -1)+1:route_index+1]:
                    for feature in feature_groups_by_edge.get(traversed, []):
                        feature_vehicles[feature].add(vid)
                    # A connector can be crossed between two samples after compaction.
                    # Validate each traversed route edge rather than skipping over it.
                    old_road = previous_road.get(vid)
                    if old_road in movement_limits and old_road != traversed:
                        passages = right_only_passages if old_road in map_config.RIGHT_ONLY_CONNECTIONS else straight_passages
                        passages[old_road] += 1
                        if traversed != movement_limits[old_road]:
                            forbidden_turns.append({'vehicle': vid, 'from': old_road, 'to': traversed})
                    previous_road[vid] = traversed
                previous_route_index[vid] = route_index
                for feature in feature_groups_by_edge.get(road, []):
                    feature_vehicles[feature].add(vid)
                if road in map_config.SCHOOL_NORTH_APPROACH_EDGES:
                    approach_max[vid] = max(approach_max.get(vid, 0), speed * 3.6)
                if road in school_roads:
                    school_max_speed = max(school_max_speed, speed)
                    school_samples += 1
                    school_max_by_edge[road] = max(school_max_by_edge.get(road, 0), speed * 3.6)
                    school_samples_by_edge[road] += 1
                    if vid not in school_entry and vid in approach_max:
                        school_entry[vid] = {"vehicle": vid, "approach_peak_kmh": round(approach_max[vid], 3),
                                             "school_entry_kmh": round(speed * 3.6, 3)}
                if vid == "ego":
                    ego_profile.append([round(traci.simulation.getTime(), 1), road,
                                        round(observation[tc.VAR_LANEPOSITION], 2), round(speed * 3.6, 3),
                                        road in school_roads])
                upcoming = observation[tc.VAR_NEXT_TLS]
                if road in stop_edges:
                    remaining = lane_lengths[observation[tc.VAR_LANE_ID]]-observation[tc.VAR_LANEPOSITION]
                    if -.05 <= remaining <= 2.5 and speed < .1:
                        stop_line_stopped[stop_edges[road]].add(vid)
                if upcoming:
                    tls, _, distance, state = upcoming[0]
                    if tls in red_stopped and state == "r" and 0 <= distance <= 20 and speed < 0.1:
                        red_stopped[tls].add(vid)
            arrivals = traci.simulation.getArrivedIDList()
            arrived.update(arrivals)
            for vid in arrivals:
                arrived_routes[vehicle_routes[vid]] += 1
            teleports += traci.simulation.getStartingTeleportNumber()
            collisions += traci.simulation.getCollidingVehiclesNumber()
            if "ego" in traci.vehicle.getIDList():
                # A short edge can be entered and left within one simulation step.
                # Track SUMO's route progress so those traversed edges are included.
                route_index = traci.vehicle.getRouteIndex("ego")
                actual_route = traci.vehicle.getRoute("ego")
                if route_index < ego_route_index:
                    ego_route_changed = True
                visited.extend(actual_route[ego_route_index + 1:route_index + 1])
                ego_route_index = route_index
            if "ego" in arrivals:
                ego_arrival = traci.simulation.getTime()
        manifest = json.loads(config.with_name("manifest.json").read_text(encoding="utf-8"))
        expected = manifest["ego_edges"]
        report = {"seconds": seconds, "ego_arrival_seconds": ego_arrival,
                  "ego_departure": ego_departure,
                  "selected_ego_route": manifest["selected_ego_route"],
                  "ego_followed_requested_loop": visited == expected and not ego_route_changed,
                  "vehicles_departed": len(departed), "vehicles_arrived": len(arrived),
                  "background_arrived": len(arrived - {"ego"}),
                  "teleports": teleports, "collision_vehicle_events": collisions,
                  "traffic_routes_configured": len(manifest["traffic_routes"]),
                  "traffic_routes_departed": len(set(departed_routes) - {"ego_loop"}),
                  "traffic_routes_completed": len(set(arrived_routes) - {"ego_loop"}),
                  "road_speed_limits_enabled": map_config.ROAD_SPEED_LIMITS_ENABLED,
                  "school_zone_limit_kmh": map_config.SCHOOL_ZONE["speed_kmh"] if map_config.ROAD_SPEED_LIMITS_ENABLED else None,
                  "school_zone_control_target_kmh": map_config.SCHOOL_ZONE.get("control_speed_kmh", map_config.SCHOOL_ZONE["speed_kmh"]) if map_config.ROAD_SPEED_LIMITS_ENABLED else None,
                  "school_zone_observed_max_kmh": round(school_max_speed * 3.6, 3),
                  "school_zone_vehicle_step_samples": school_samples,
                  "school_zone_max_kmh_by_edge": {edge: round(speed, 3) for edge, speed in school_max_by_edge.items()},
                  "school_zone_samples_by_edge": dict(school_samples_by_edge),
                  "straight_only_passages": dict(straight_passages),
                  "right_only_passages": dict(right_only_passages),
                  "forbidden_turns_observed": forbidden_turns,
                  "feature_vehicles_observed": {name: len(vehicles) for name, vehicles in feature_vehicles.items()},
                  "vehicles_observed_stopped_at_red": {tls: len(vids) for tls, vids in red_stopped.items()},
                  "vehicles_observed_stopped_at_stop_line": {node:len(vids) for node,vids in stop_line_stopped.items()},
                  "school_north_entries": list(school_entry.values()),
                  "visited_ego_edges": visited}
        config.with_name("check_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        with config.with_name("ego_speed_profile.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", "edge", "lane_position_m", "speed_kmh", "inside_school_zone"])
            writer.writerows(ego_profile)
        print(json.dumps(report, indent=2))
        if (ego_arrival is None or ego_departure is None
                or abs(ego_departure['distance_to_junction_m']-map_config.EGO_START_DISTANCE_M)>.1
                or visited != expected or ego_route_changed or teleports or collisions
                or (map_config.ROAD_SPEED_LIMITS_ENABLED and school_max_speed * 3.6 > map_config.SCHOOL_ZONE["speed_kmh"] + 0.05) or forbidden_turns):
            raise SystemExit("Map check failed; inspect generated/check_report.json")
    finally:
        traci.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--netedit", action="store_true")
    mode.add_argument("--build-only", action="store_true")
    parser.add_argument("--no-traffic", action="store_true")
    parser.add_argument("--background-only", action="store_true", help="빨간 시연 차량 없이 일반 차량만 주행")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--route", choices=list(map_config.EGO_ROUTES), default=map_config.DEFAULT_EGO_ROUTE,
                        help="outermost: 최외곽 폐루프(기본) / expanded: 동일 경로 별칭 / original: 기존 북쪽 루프")
    parser.add_argument("--seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.laps < 1 or args.seconds < 1:
        parser.error("--laps and --seconds must be positive")
    if args.check and args.background_only:
        parser.error("일반 차량 사고율 측정은 python evaluate_traffic.py 를 사용하세요.")
    config = build(laps=args.laps, traffic=not args.no_traffic, include_ego=not args.background_only,
                   route_name=args.route)
    print(f"Map: {config}")
    print(f"Preview: {config.parent.parent / 'preview.svg'}")
    if args.check:
        check(config, args.seconds)
    elif args.netedit:
        subprocess.run([checkBinary("netedit"), str(config.with_name("seongsan.net.xml"))], check=True)
    elif not args.build_only:
        subprocess.run([checkBinary("sumo-gui"), "-c", str(config)], check=True)


if __name__ == "__main__":
    main()
