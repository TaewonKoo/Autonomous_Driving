"""Driving tasks for the paper profiles, with the project's SUMO network."""
from collections import namedtuple
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import traci
from sumolib import checkBinary

from .profiles import TRAFFIC_RANDOMIZATION


Vehicle = namedtuple("Vehicle", "id gap speed accel lateral")
MISSING = Vehicle("", 120., 0., 0., 0.)
MAP = Path(__file__).resolve().parents[1] / "maps" / "seongsan" / "generated"
MERGE_EDGE = "godeungeo_road"


def speed_reward(speeds, target):
    values = np.asarray(speeds)
    return max(0., 1. - float(np.sqrt(np.mean((values - target) ** 2))) / target) if values.size else 0.


def idm(speed, leader_speed, gap, target, accel, brake, tau=1., min_gap=2.):
    desired = min_gap + np.maximum(0., speed * (tau + (speed - leader_speed) /
                                                (2 * np.sqrt(accel * brake))))
    return accel * (1 - (speed / target) ** 4 - (desired / np.maximum(gap, .05)) ** 2)


def bounded_accel(action, settings):
    return float(np.clip(action, -1, 1)) * (settings.get("accel", 2.5) if action >= 0
                                             else -settings.get("brake", -4.5))


class Ring:
    """Vectorized longitudinal IDM on a periodic, single-lane road."""

    def __init__(self, profile, seed, output, evaluation=False):
        self.p = profile
        self.rng = np.random.default_rng(seed)
        self.evaluation = evaluation
        self.flow = profile.scenario == "flow"
        self.n = 22 if self.flow else 30 + profile.cavs
        self.cavs = np.linspace(0, self.n, profile.cavs, endpoint=False, dtype=int)
        self.groups = profile.cavs if not self.flow and profile.cooperation != "global" else 1
        self.obs_dim = 3 if self.flow else {
            "global": 2 * profile.cavs, "shared": 2, "local": 2 * (2 * profile.radius - 1)
        }[profile.cooperation]
        self.action_dim = profile.cavs if self.groups == 1 else 1
        self.discrete = 0
        self.steps = 0

    def reset(self):
        self.length = float(self.rng.uniform(*(self.p.paper_settings["eval_m" if self.evaluation else
                                                                  "train_m"]))) if self.flow else 230.
        self.position = np.arange(self.n) * self.length / self.n
        self.speed = np.zeros(self.n)
        self.steps = 0
        for _ in range(self.p.paper_settings.get("warmup", 0)):
            self.advance()
        return self.observation()

    def advance(self, action=None):
        gap = (np.roll(self.position, -1) - self.position) % self.length - 5.
        accel = idm(self.speed, np.roll(self.speed, -1), gap, 30., 1., 1.5)
        if self.flow:
            accel += self.rng.normal(0, np.sqrt(.2), self.n)
        new_speed = np.maximum(0, self.speed + np.clip(accel, -4.5, 2.5) * self.p.dt)
        if action is not None:
            change = np.clip(np.ravel(action), -1, 1)
            if self.flow:
                change = bounded_accel(change[0], {}) * self.p.dt
            new_speed[self.cavs] = np.maximum(0, self.speed[self.cavs] + change)
        # Propagate a safe speed around the ring, including when a leader brakes.
        safe_speed = np.maximum(0., gap - .05) / self.p.dt
        for _ in range(self.n):
            constrained = np.minimum(new_speed, np.roll(new_speed, -1) + safe_speed)
            if np.array_equal(constrained, new_speed):
                break
            new_speed = constrained
        self.position = (self.position + self.p.dt * new_speed) % self.length
        actual_accel = (new_speed[0] - self.speed[0]) / self.p.dt
        self.speed = new_speed
        return actual_accel

    def observation(self):
        if self.flow:
            return np.array([[self.speed[0] / 30, (self.speed[1] - self.speed[0]) / 30,
                              ((self.position[1] - self.position[0]) % self.length) / 270]],
                            dtype=np.float32)
        positions = self.position[self.cavs]
        speeds = self.speed[self.cavs]
        headway = ((np.roll(positions, -1) - positions) % self.length) / np.maximum(speeds, .1)
        states = np.column_stack((speeds, headway)).astype(np.float32)
        if self.p.cooperation == "global":
            return states.reshape(1, -1)
        if self.p.cooperation == "local":
            return np.concatenate([np.roll(states, offset, axis=0)
                                   for offset in range(1 - self.p.radius, self.p.radius)], axis=1)
        return states

    def step(self, action):
        accel = self.advance(action)
        self.steps += 1
        if self.flow:
            reward = [self.speed.mean() - .1 * abs(accel)]
        else:
            reward = np.maximum(0, 1 - np.abs(25 - self.speed[self.cavs]) / 25)
            if self.groups == 1:
                reward = [reward.max()]
        return self.observation(), reward, False, self.steps >= self.p.horizon, {
            "mean_speed": float(self.speed.mean()), "collision": False, "success": False}

    def close(self):
        pass


class Seongsan:
    def __init__(self, profile, seed, output, evaluation=False):
        self.p = profile
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.scenario = profile.scenario
        self.limit = (profile.evaluation_horizon or profile.horizon) if evaluation else (
            max(profile.horizon, 3600) if self.scenario in ("lane", "transfer") else profile.horizon)
        self.groups = profile.cavs if self.scenario == "intersection" else 1
        self.action_dim = profile.cavs if self.scenario == "benchmark" else 1
        self.obs_dim = {"lane": 21, "benchmark": 5 * profile.cavs, "transfer": 5,
                        "intersection": 6, "merge": 11, "freeway": 10}[self.scenario]
        self.discrete = 6 if self.scenario == "lane" else 0
        self.conn = None
        self.label = str(output.resolve())
        self.route_file = (output / "traffic.rou.xml").resolve()
        self.write_routes()

    def write_routes(self):
        routes = ET.parse(MAP / "seongsan.rou.xml")
        root = routes.getroot()
        ego = root.find("./vehicle[@id='ego']")
        ego_type = root.find("./vType[@id='ego_type']")
        ego.set("depart", "15")
        ego_type.set("emergencyDecel", "4.5")
        ego_type.set("carFollowModel", "IDM")
        ego_type.set("accel", str(self.p.paper_settings.get("accel", 2.5)))
        if self.scenario in ("merge", "freeway"):
            if self.scenario == "merge":
                ego_type.set("length", "5")
                edges = ("minor_coast_rev east_turn_pocket_rev east_right_slip "
                         "godeungeo_road godeungeo_approach")
            else:
                edges = ("south_connector coast_south_rev coast_merge_approach_rev "
                         "triangle_east_leg_rev triangle_east_upper_rev minor_coast_rev")
            root.find("./route[@id='ego_loop']").set("edges", edges)
            ego.set("departPos", "base")
            ego.set("arrivalPos", "max")
            ego.set("arrivalLane", "current")
        if self.scenario in ("lane", "intersection"):
            for vehicle in root.findall(".//vType[@vClass='passenger']"):
                if vehicle is not ego_type:
                    vehicle.set("carFollowModel", "IDM")
                    if self.scenario == "intersection":
                        vehicle.attrib.update(maxSpeed="15", tau="1", minGap="2", delta="4",
                                              accel="1", decel="1.5")
        if self.scenario in ("merge", "freeway"):
            mix = root.find("./vTypeDistribution[@id='background']")
            mix.clear()
            mix.set("id", "background")
            # A seeded population samples all seven intervals; SUMO draws a type for each car.
            for i in range(96):
                attributes = {key: f"{self.rng.uniform(*bounds):.6f}"
                              for key, bounds in TRAFFIC_RANDOMIZATION.items()}
                ET.SubElement(mix, "vType", id=f"paper_idm_{i}", probability="1",
                              carFollowModel="IDM", laneChangeModel="SL2015",
                              speedFactor="1", speedDev="0", length="5", **attributes)
        # SUMO skips a flow if its start time precedes the preceding departure.
        traffic = [element for element in root if element.tag in ("vehicle", "flow")]
        for element in traffic:
            root.remove(element)
        root.extend(sorted(traffic, key=lambda element: float(element.get("depart", element.get("begin", "0")))))
        # Sumolib starts SUMO with a separate working directory on some Windows installs.
        routes.write(self.route_file, encoding="utf-8", xml_declaration=True)

    def reset(self):
        options = ["-n", str(MAP / "seongsan.net.xml"), "-r", str(self.route_file),
                   "--step-length", str(self.p.dt), "--seed", str(self.rng.integers(1, 2**31 - 1)),
                   "--time-to-teleport", "-1", "--collision.action", "warn",
                   "--collision.check-junctions", "true", "--collision.mingap-factor", "0",
                   "--lateral-resolution", "0.8" if self.scenario == "lane" else "-1",
                   "--no-step-log", "true", "--no-warnings", "true"]
        if self.conn is None:
            traci.start([checkBinary("sumo"), *options], label=self.label, doSwitch=False,
                        stdout=subprocess.DEVNULL)
            self.conn = traci.getConnection(self.label)
        else:
            self.conn.load(options)
        self.v = self.conn.vehicle
        for _ in range(round(60 / self.p.dt)):
            self.conn.simulationStep()
            if "ego" in self.conn.simulation.getDepartedIDList():
                break
        else:
            raise RuntimeError("SUMO did not insert ego within 60 seconds")
        self.steps = 0
        self.previous_accel = self.previous_lateral_accel = 0.
        self.origin_lane = None
        self.target_lane = None
        self.merge_seen = False
        self.controlled = ["ego"] + [None] * (self.p.cavs - 1)
        self.lane_goal_met = False
        self.eligible = set()
        self.select_cavs(initial=True)
        return self.observation()

    def select_cavs(self, initial=False):
        if self.p.cavs == 1:
            return
        present = set(self.v.getIDList())
        self.controlled = [vehicle if vehicle in present else None for vehicle in self.controlled]
        self.eligible.intersection_update(present)
        if self.scenario == "intersection":
            candidates = sorted(v for v in present if self.v.getRoadID(v) in (
                "school_approach", "west_bank", "west_lower_rev", "godeungeo_road",
                "godeungeo_approach_rev", "east_right_slip", "minor_coast_rev",
                "triangle_east_upper_rev"))
        else:
            arriving = sorted(present) if initial else self.conn.simulation.getDepartedIDList()
            self.eligible.update(vehicle for vehicle in arriving if self.rng.random() <
                                 self.p.paper_settings["nominal_penetration"])
            candidates = sorted(present & self.eligible)
        available = iter(v for v in candidates if v not in self.controlled and v != "ego")
        for slot in range(1, self.p.cavs):
            if self.controlled[slot] is None:
                self.controlled[slot] = next(available, None)

    def policy_ids(self):
        return tuple(self.controlled) if self.groups > 1 else ("ego",)

    def neighbor(self, raw):
        if not raw or not raw[0]:
            return MISSING
        vid, gap = raw
        return Vehicle(vid, max(0., gap), self.v.getSpeed(vid), self.v.getAcceleration(vid),
                       self.v.getLateralLanePosition(vid) + 3.2 * self.v.getLaneIndex(vid))

    def adjacent(self, vid, direction, leader):
        mode = (2 if direction > 0 else 3) if leader else (0 if direction > 0 else 1)
        raw = self.v.getNeighbors(vid, mode)
        return self.neighbor(min(raw, key=lambda pair: pair[1]) if raw else None)

    def neighbors(self, vid):
        lane = self.v.getLaneIndex(vid)
        road = self.v.getRoadID(vid)
        lanes = self.conn.edge.getLaneNumber(road) if not road.startswith(":") else 1
        direction = 1 if lane + 1 < lanes else -1 if lane > 0 and lanes > 1 else 0
        if self.scenario == "lane":
            if self.target_lane is None or self.target_lane[0] != road:
                lanes_to_route = {entry[0] for entry in self.v.getBestLanes(vid) if entry[4]}
                alternate = lane + direction
                if not direction or f"{road}_{alternate}" not in lanes_to_route:
                    alternate = lane
                self.target_lane = (road, alternate)
                self.origin_lane = None
            direction = int(np.sign(self.target_lane[1] - lane))
        front = self.neighbor(self.v.getLeader(vid, 120))
        rear = self.neighbor(self.v.getFollower(vid, 120))
        return (front, rear, self.adjacent(vid, direction, True) if direction else MISSING,
                self.adjacent(vid, direction, False) if direction else MISSING), direction

    def merge_state(self):
        road = self.v.getRoadID("ego")
        position = self.v.getLanePosition("ego")
        route_distance = self.v.getDrivingDistance("ego", MERGE_EDGE, 0.)
        if road == MERGE_EDGE:
            distance = position
        elif route_distance >= 0:
            distance = -route_distance
        else:
            distance = 0.
        peers = []
        for vid in self.v.getIDList():
            if vid == "ego":
                continue
            edge = self.v.getRoadID(vid)
            if edge == MERGE_EDGE:
                peer_position = self.v.getLanePosition(vid)
            else:
                approach = self.v.getDrivingDistance(vid, MERGE_EDGE, 0.)
                if approach < 0 or approach > 120:
                    continue
                peer_position = -approach
            peers.append((peer_position, self.v.getSpeed(vid), self.v.getAcceleration(vid)))
        ahead = sorted((p for p in peers if p[0] >= distance), key=lambda p: p[0])
        behind = sorted((p for p in peers if p[0] < distance), key=lambda p: -p[0])
        ahead = (ahead + [(distance + 120, 0., 0.)] * 2)[:2]
        behind = (behind + [(distance - 120, 0., 0.)] * 2)[:2]
        return distance, ahead, behind

    def observation(self):
        if self.scenario == "merge":
            distance, ahead, behind = self.merge_state()
            obs = [*ahead[1][:2], *ahead[0][:2], distance, self.v.getSpeed("ego"),
                   self.v.getAcceleration("ego"), *behind[0][:2], *behind[1][:2]]
            scale = [120, 20, 120, 20, 120, 20, 5, 120, 20, 120, 20]
            return np.asarray([obs], np.float32) / np.asarray(scale, np.float32)
        if self.scenario in ("lane", "freeway"):
            (front, rear, next_front, next_rear), _ = self.neighbors("ego")
            speed = self.v.getSpeed("ego")
            accel = self.v.getAcceleration("ego")
            if self.scenario == "freeway":
                cars = (front, rear, next_front, next_rear)
                return np.array([[*(c.gap / 120 for c in cars), *(c.speed / 20 for c in cars),
                                  speed / 20, accel / 5]], dtype=np.float32)
            obs = [self.v.getDistance("ego") / 1100, speed / 20, accel / 5,
                   (self.v.getLateralLanePosition("ego") + 3.2 * self.v.getLaneIndex("ego")) / 6.4,
                   self.v.getLateralSpeed("ego") / 3]
            for car, sign in ((front, 1), (next_front, 1), (rear, -1), (next_rear, -1)):
                obs.extend((sign * car.gap / 120, car.speed / 20, car.accel / 5, car.lateral / 6.4))
            return np.array([obs], dtype=np.float32)
        observations = []
        for vid in self.controlled:
            if vid is None:
                observations.append([0.] * (6 if self.scenario == "intersection" else 5))
                continue
            front = self.neighbor(self.v.getLeader(vid, 120))
            rear = self.neighbor(self.v.getFollower(vid, 120))
            obs = [self.v.getSpeed(vid) / 20, front.speed / 20, front.gap / 120,
                   rear.speed / 20, rear.gap / 120]
            if self.scenario == "intersection":
                obs.insert(0, self.v.getLanePosition(vid) / 120)
            observations.append(obs)
        return np.array(observations, dtype=np.float32).reshape(self.groups, -1)

    def step(self, action):
        old_road = self.v.getRoadID("ego")
        old_lane = self.v.getLaneIndex("ego")
        changed = denied = False
        near = lane_error = old_lat_speed = 0.
        values = np.ravel(action)
        paper = self.p.paper_settings
        if self.scenario in ("lane", "freeway"):
            (front, rear, adjacent_front, adjacent_rear), direction = self.neighbors("ego")
            speed = self.v.getSpeed("ego")
            old_lat_speed = self.v.getLateralSpeed("ego")
            if self.scenario == "lane":
                lateral, follow = divmod(int(values[0]), 2)
                leader = adjacent_front if follow else front
                desired = max(1., self.v.getAllowedSpeed("ego"))
                accel = float(np.clip(idm(speed, leader.speed, leader.gap, desired, 2.5, 4.5), -4.5, 2.5))
                relevant = ((adjacent_front,) if lateral == 0 and not follow else
                            (adjacent_front, adjacent_rear) if lateral == 1 and follow else
                            (front, rear) if lateral == 2 and not follow else ())
                near = min((-1 / (c.gap + .1) for c in relevant if c.id and c.gap < 10), default=0.)
                lane_error = abs(3.2 * direction - self.v.getLateralLanePosition("ego")) if direction else 0.
                requested = lateral == 1 or (lateral == 2 and self.origin_lane is not None)
                if lateral == 2 and self.origin_lane is not None:
                    origin_road, origin_lane = self.origin_lane
                    direction = int(np.sign(origin_lane - old_lane)) if origin_road == old_road else 0
                    if not direction and origin_road == old_road:
                        self.v.changeLane("ego", old_lane, self.p.dt)
                        self.origin_lane = None
            else:
                accel = bounded_accel(values[0], paper)
                requested = int(values[1]) == 1
            if requested and direction:
                next_front = self.adjacent("ego", direction, True)
                next_rear = self.adjacent("ego", direction, False)
                if self.scenario == "lane" and min(next_front.gap, next_rear.gap) < 10:
                    denied = True
                else:
                    self.v.changeLane("ego", old_lane + direction, self.p.dt)
                    if self.scenario == "lane":
                        self.origin_lane = (old_road, old_lane) if lateral == 1 else None
            self.v.setAcceleration("ego", accel, self.p.dt)
        else:
            for vid, value in zip(self.controlled, values):
                if vid is not None:
                    self.v.setAcceleration(vid, bounded_accel(value, paper), self.p.dt)

        self.conn.simulationStep()
        self.steps += 1
        self.select_cavs()
        sim = self.conn.simulation
        collisions = sim.getCollidingVehiclesIDList()
        collision = bool(collisions) if self.scenario == "intersection" else "ego" in collisions
        success = "ego" in sim.getArrivedIDList() and not collision
        missing = "ego" not in self.v.getIDList()
        done = collision or success or missing
        speed = self.v.getSpeed("ego") if not missing else 0.
        accel = self.v.getAcceleration("ego") if not missing else self.previous_accel
        jerk = (accel - self.previous_accel) / self.p.dt
        if not missing:
            road = self.v.getRoadID("ego")
            changed = road == old_road and old_lane != self.v.getLaneIndex("ego")
            self.merge_seen |= road == MERGE_EDGE
            if changed and self.target_lane == (road, self.v.getLaneIndex("ego")):
                self.lane_goal_met = True
        reward = speed / 20 - .01 * abs(jerk) + float(success)
        network_speed = None
        if self.scenario == "lane":
            lateral_accel = ((self.v.getLateralSpeed("ego") - old_lat_speed) / self.p.dt
                             if not missing else 0.)
            lateral_jerk = (lateral_accel - self.previous_lateral_accel) / self.p.dt
            self.previous_lateral_accel = lateral_accel
            target = self.v.getAllowedSpeed("ego") if not missing else 0.
            reward = (-.01 * self.p.dt - .04 * lane_error - .03 * abs(speed - target)
                      - .0005 * (jerk**2 + lateral_jerk**2) + near - float(denied))
            if collision:
                reward -= 100
        elif self.scenario in ("benchmark", "intersection"):
            speeds = [self.v.getSpeed(vid) for vid in self.v.getIDList()]
            network_speed = float(np.mean(speeds)) if speeds else 0.
            reward = 0. if collision else speed_reward(speeds, 15.)
            if self.scenario == "benchmark":
                reward -= .1 * sum(max(0., 1. - self.neighbor(self.v.getLeader(vid, 120)).gap /
                                       max(self.v.getSpeed(vid), .1)) for vid in self.controlled
                                   if vid is not None)
        elif self.scenario == "merge":
            reward = -.015 * abs(jerk) / 3
            if not done:
                distance, ahead, behind = self.merge_state()
                p, f = ahead[0], behind[0]
                if f[0] > distance - 120:
                    reward -= .015 * max(0., -f[2]) / 4.5
                    if self.merge_seen and p[0] < distance + 120:
                        imbalance = (abs(p[0] - distance - 5) - abs(distance - f[0] - 5))
                        reward -= .015 * (abs(imbalance) / max(abs(p[0] - f[0] - 10), .1)
                                          + abs((p[1] + f[1]) / 2 - speed) / 5)
                if speed < .1:
                    reward -= .5
            reward += float(success) - float(collision)
        elif self.scenario == "freeway":
            gap = self.neighbor(self.v.getLeader("ego", 120)).gap if not missing else 120.
            reward = (-5. if gap < 25 else -2.) if changed and gap != 25 else 0.
            reward -= 10 * max(0., 25 - gap) / 25 + .005 * abs(jerk)
            if gap < 27.5:
                if speed < 8.89:
                    reward -= .5 * abs(speed - 8.89) / 8.89
                elif speed < 16.89:
                    reward += abs(speed - 8.89) / 16.89
                else:
                    reward -= .5 * abs(speed - 16.89) / 16.89
            if collision:
                reward -= 200
        elif collision:
            reward -= 100
        self.previous_accel = accel
        state = np.zeros((self.groups, self.obs_dim), dtype=np.float32) if done else self.observation()
        return state, [reward] * self.groups, done, self.steps >= self.limit and not done, {
            "mean_speed": float(speed) if network_speed is None else network_speed,
            "collision": collision, "success": success, "lane_goal_met": self.lane_goal_met}

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None


def make_environment(profile, seed, output, evaluation=False):
    environment = Ring if profile.scenario in ("flow", "cooperative") else Seongsan
    return environment(profile, seed, output, evaluation)
