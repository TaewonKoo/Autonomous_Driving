"""성산 고성리 맵 생성 / 미리보기 / 차량 주행 검증.

python view_map.py                 SUMO GUI (▶를 눌러 재생)
python view_map.py --netedit        도로 편집기
python view_map.py --check          완주 및 교통 검증
python view_map.py --build-only     파일 생성만
"""
import argparse
import csv
import json
import subprocess
from collections import Counter

from sumolib import checkBinary
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
    teleports = collisions = 0
    ego_arrival = None
    vehicle_routes = {}
    departed_routes = Counter()
    arrived_routes = Counter()
    bakery_vehicles = set()
    bakery_reverse_vehicles = set()
    school_roads = {eid + suffix for eid in map_config.SCHOOL_ZONE["edges"] for suffix in ("", "_rev")}
    school_max_speed = 0.0
    school_samples = 0
    red_stopped = {tls: set() for tls in map_config.SIGNAL_CONTROLLERS}
    approach_max = {}
    school_entry = {}
    ego_profile = []
    school_max_by_edge = {}
    school_samples_by_edge = Counter()
    previous_road = {}
    straight_passages = Counter()
    forbidden_turns = []
    try:
        while traci.simulation.getTime() < seconds:
            traci.simulationStep()
            bakery_vehicles.update(traci.edge.getLastStepVehicleIDs("bakery_sagyejeong"))
            bakery_reverse_vehicles.update(traci.edge.getLastStepVehicleIDs("bakery_sagyejeong_rev"))
            new_vehicles = traci.simulation.getDepartedIDList()
            departed.update(new_vehicles)
            for vid in new_vehicles:
                vehicle_routes[vid] = traci.vehicle.getRouteID(vid)
                departed_routes[vehicle_routes[vid]] += 1
                traci.vehicle.subscribe(vid, [tc.VAR_SPEED, tc.VAR_ROAD_ID, tc.VAR_NEXT_TLS,
                                             tc.VAR_LANEPOSITION])
            for vid, observation in traci.vehicle.getAllSubscriptionResults().items():
                speed = observation[tc.VAR_SPEED]
                road = observation[tc.VAR_ROAD_ID]
                if road and not road.startswith(":"):
                    old_road = previous_road.get(vid)
                    if old_road in map_config.STRAIGHT_ONLY_CONNECTIONS and old_road != road:
                        straight_passages[old_road] += 1
                        if road != map_config.STRAIGHT_ONLY_CONNECTIONS[old_road]:
                            forbidden_turns.append({"vehicle": vid, "from": old_road, "to": road})
                    previous_road[vid] = road
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
                edge = traci.vehicle.getRoadID("ego")
                if not edge.startswith(":") and (not visited or visited[-1] != edge):
                    visited.append(edge)
            if "ego" in arrivals:
                ego_arrival = traci.simulation.getTime()
        manifest = json.loads(config.with_name("manifest.json").read_text(encoding="utf-8"))
        expected = manifest["ego_edges"]
        report = {"seconds": seconds, "ego_arrival_seconds": ego_arrival,
                  "ego_followed_requested_loop": visited == expected,
                  "vehicles_departed": len(departed), "vehicles_arrived": len(arrived),
                  "background_arrived": len(arrived - {"ego"}),
                  "teleports": teleports, "collision_vehicle_events": collisions,
                  "traffic_routes_configured": len(manifest["traffic_routes"]),
                  "traffic_routes_departed": len(set(departed_routes) - {"ego_loop"}),
                  "traffic_routes_completed": len(set(arrived_routes) - {"ego_loop"}),
                  "bakery_road_vehicles_toward_village": len(bakery_vehicles),
                  "bakery_road_vehicles_toward_roundabout": len(bakery_reverse_vehicles),
                  "school_zone_limit_kmh": map_config.SCHOOL_ZONE["speed_kmh"],
                  "school_zone_observed_max_kmh": round(school_max_speed * 3.6, 3),
                  "school_zone_vehicle_step_samples": school_samples,
                  "school_zone_max_kmh_by_edge": {edge: round(speed, 3) for edge, speed in school_max_by_edge.items()},
                  "school_zone_samples_by_edge": dict(school_samples_by_edge),
                  "straight_only_passages": dict(straight_passages),
                  "forbidden_turns_observed": forbidden_turns,
                  "vehicles_observed_stopped_at_red": {tls: len(vids) for tls, vids in red_stopped.items()},
                  "school_north_entries": list(school_entry.values()),
                  "visited_ego_edges": visited}
        config.with_name("check_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        with config.with_name("ego_speed_profile.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", "edge", "lane_position_m", "speed_kmh", "inside_school_zone"])
            writer.writerows(ego_profile)
        print(json.dumps(report, indent=2))
        if (ego_arrival is None or visited != expected or teleports or collisions
                or school_max_speed * 3.6 > map_config.SCHOOL_ZONE["speed_kmh"] + 0.05 or forbidden_turns):
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
    parser.add_argument("--seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.laps < 1 or args.seconds < 1:
        parser.error("--laps and --seconds must be positive")
    if args.check and args.background_only:
        parser.error("일반 차량 사고율 측정은 python evaluate_traffic.py 를 사용하세요.")
    config = build(laps=args.laps, traffic=not args.no_traffic, include_ego=not args.background_only)
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
