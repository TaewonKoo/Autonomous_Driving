"""Measure physical collisions with only background traffic, across random seeds.

python evaluate_traffic.py --seeds 10 --seconds 3600
No learned policy, no red demo vehicle, and no changes to driver parameters.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

from sumolib import checkBinary
from maps.seongsan import config as C
from maps.seongsan.builder import build


def run_seed(config, out, seed, seconds, clearance):
    folder = out / f"seed_{seed:02d}"
    folder.mkdir()
    command = [checkBinary("sumo"), "-c", str(config), "--seed", str(seed),
               "--end", str(seconds + clearance), "--no-step-log", "true",
               "--collision.action", "warn", "--collision.mingap-factor", "0",
               "--collision.check-junctions", "true", "--time-to-teleport", "-1",
               "--collision-output", str(folder / "collisions.xml"),
               "--tripinfo-output", str(folder / "trips.xml"),
               "--tripinfo-output.write-unfinished", "true",
               "--summary-output", str(folder / "simulation.xml"),
               "--summary-output.period", "60", "--duration-log.statistics", "true"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    log = result.stdout + result.stderr
    (folder / "sumo.log").write_text(log, encoding="utf-8")
    (folder / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"SUMO seed {seed} failed: {result.stderr}")
    trips = ET.parse(folder / "trips.xml").getroot().findall("tripinfo")
    assert all(t.get("id") != "ego" for t in trips)
    departed = [t for t in trips if float(t.get("depart", "-1")) >= 0]
    arrived = [t for t in departed if float(t.get("arrival", "-1")) >= 0]
    if not departed:
        raise RuntimeError(f"No vehicles in seed {seed}")
    collisions = ET.parse(folder / "collisions.xml").getroot().findall("collision")
    involved = {c.get(key) for c in collisions for key in ("collider", "victim") if c.get(key)}
    steps = ET.parse(folder / "simulation.xml").getroot().findall("step")
    last = steps[-1]
    return {"seed": seed, "departed": len(departed), "arrived": len(arrived),
            "unfinished": len(departed) - len(arrived),
            "waiting_to_insert_at_end": int(last.get("waiting", "0")),
            "collision_records": len(collisions), "collision_involved_vehicles": len(involved),
            "collision_vehicle_rate_percent": 100 * len(involved) / len(departed),
            "collision_vehicles": sorted(involved),
            "emergency_braking_warnings": len(re.findall(r"performs emergency braking", log)),
            "teleport_warnings": len(re.findall(r"Teleporting vehicle", log)),
            "mean_trip_seconds": round(sum(float(t.get("duration")) for t in arrived) / len(arrived), 2)
            if arrived else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--clearance", type=int, default=1800)
    args = parser.parse_args()
    if args.seeds < 1 or args.seconds < 1 or args.clearance < 0:
        parser.error("seeds/seconds must be positive and clearance non-negative")
    root = Path(__file__).resolve().parent
    out = root / "results" / f"traffic_baseline_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    # Isolated generated files keep the user's GUI configuration unchanged.
    original_end = C.TRAFFIC_END
    C.TRAFFIC_END = args.seconds
    try:
        config = build(out / "network" / "generated", include_ego=False)
    finally:
        C.TRAFFIC_END = original_end
    (out / "config_snapshot.py").write_bytes((root / "maps/seongsan/config.py").read_bytes())
    with ThreadPoolExecutor(max_workers=min(4, args.seeds)) as pool:
        futures = [pool.submit(run_seed, config, out, seed, args.seconds, args.clearance)
                   for seed in range(args.seeds)]
        runs = []
        for future in futures:
            run = future.result()
            runs.append(run)
            print(f"seed {run['seed']}: {run['collision_involved_vehicles']}/{run['departed']} "
                  f"collision-involved vehicles, {run['unfinished']} unfinished", flush=True)
    total = sum(r["departed"] for r in runs)
    involved = sum(r["collision_involved_vehicles"] for r in runs)
    report = {"description": "Background vehicles only; no ego or learned policy",
              "collision_definition": "physical body overlap; junction collision detection enabled",
              "rate_definition": "distinct collision-involved vehicles / departed vehicles",
              "demand_vehicles_per_hour": json.loads(config.with_name("manifest.json").read_text(encoding="utf-8"))["traffic_per_hour"],
              "injection_seconds_per_run": args.seconds, "clearance_limit_seconds": args.clearance,
              "sumo_version": subprocess.run([checkBinary("sumo"), "--version"],
                                             capture_output=True, text=True, check=True).stdout.splitlines()[0],
              "network_sha256": hashlib.sha256(config.with_name("seongsan.net.xml").read_bytes()).hexdigest(),
              "runs": runs, "total_departed": total,
              "total_arrived": sum(r["arrived"] for r in runs),
              "total_unfinished": sum(r["unfinished"] for r in runs),
              "total_collision_records": sum(r["collision_records"] for r in runs),
              "total_collision_involved_vehicles": involved,
              "collision_vehicle_rate_percent": 100 * involved / total,
              "runs_with_collisions": sum(r["collision_records"] > 0 for r in runs),
              "total_emergency_braking_warnings": sum(r["emergency_braking_warnings"] for r in runs),
              "total_teleport_warnings": sum(r["teleport_warnings"] for r in runs)}
    report_file = out / "report.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, ensure_ascii=False, indent=2))
    print(f"Report: {report_file}")


if __name__ == "__main__":
    main()
