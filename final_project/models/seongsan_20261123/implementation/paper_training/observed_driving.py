"""Observation-driven SUMO control with explicit paper and transfer provenance."""
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .environments import MAP
from .profiles import Profile


ROOT = Path(__file__).resolve().parents[1]


def load_sumo():
    """Use the already installed in-process SUMO runtime when available."""
    try:
        import libsumo
    except ImportError:
        dependencies = ROOT / "results/paper_reference_sources_20261120/deps"
        if not (dependencies / "libsumo").is_dir():
            raise RuntimeError("The existing paper reference libsumo installation is required.")
        sys.path.insert(0, str(dependencies))
        import libsumo
    return libsumo


@dataclass(frozen=True)
class DrivingSettings:
    dt: float = .2
    accel: float = 3.
    brake: float = 4.
    speed: float = 50 / 3.6
    length: float = 5.
    front_range: float = 50.
    rear_range: float = 10.
    communication_range: float = 300.
    cycle_scale: float = 90.
    lane_duration: float = 3.
    stage_steps: int = 100
    full_seconds: float = 500.
    collision_reward: float = -10.
    success_reward: float = 1.
    time_objective: float = -.001
    time_weight: float = 5.
    red_base: float = -.25
    red_speed: float = -.02
    overspeed: float = -.002


SETTINGS = DrivingSettings()
PPO_PROFILE = Profile(
    paper=4, source="https://arxiv.org/pdf/1707.06347", algorithm="ppo",
    scenario="transfer", horizon=2048, steps=1_000_000,
    actor=(64, 64), critic=(64, 64), lr=3e-4, batch=64, epochs=10,
    gamma=.99, gae=.95, clip=.2,
)


class EgoInsertionError(RuntimeError):
    """A sampled starting state is blocked by existing traffic."""


def discounted_failure_time_cost(time_cost, gamma, remaining_steps):
    """Discounted future time costs of a finite-horizon absorbing failure.

    This is a Seongsan experimental termination convention, not a paper reward.
    The first omitted cost is one step after the actual failure transition.
    """
    remaining_steps = max(0, int(remaining_steps))
    if gamma == 1.:
        return float(time_cost * remaining_steps)
    return float(time_cost * gamma * (1. - gamma ** remaining_steps) / (1. - gamma))


def provenance(numeric_correction=False, failure_time_tail=False):
    result = {
        "status": "SUMO transfer combining published components; not an exact paper reproduction",
        "ppo": {"values": {key: getattr(PPO_PROFILE, key) for key in
                           ("algorithm", "horizon", "actor", "critic", "activation", "lr",
                            "batch", "epochs", "gamma", "gae", "clip", "anneal")},
                "source": "Schulman 2017, section 6.1, Tables 1 and 3",
                "transfer": "Categorical joint acceleration/lane actions replace Gaussian robotics actions. The original T=2048 is used as a total update batch, split into 512-step segments across four workers; it is not 2048 steps per worker."},
        "dynamics": {"values": asdict(SETTINGS),
                     "source": "Jiang 2023 Table 3: acceleration +3/-4, 50 km/h, length 5 m, front/back 50/10 m, lane duration 3 s; section 5.1 V2I 300 m, cycle 90 s. Isele 2018: dt .2 s, horizon 100 steps, rewards +1/-10. Zhu 2021 Table III: time objective -0.001, time weight 5, red base -0.25, red speed -0.02, overspeed -0.002.",
                     "full_seconds": "Existing course episode budget, not a paper driving parameter."},
        "observations": {
            "neighbors": "Ye 2020 section IV-B, 21 features for ego and four neighbors; relative positions/scales adapted to the local road.",
            "cross_traffic": "Isele 2018 sequential-action grid, 5x11, forward 0..20 m, lateral +/-90 m, heading/speed/occupancy. Ego-relative frame is the transfer choice.",
            "signals": "Jiang 2023 section 3.2 and Zhu 2021 Table I. Route-specific light, distance and time to green. Assumes V2I in 300 m. Explicit presence and yellow coding are transfer choices.",
            "route": "Course connectivity plus selected route, stop/yield sign and own stopping history. These are Seongsan additions, not claimed as verbatim paper inputs."},
        "reward": {"version": "published_components_v2",
                   "source": "Isele 2018 section III: success +1 and collision -10. Zhu 2021 equations (24),(30): time component c_obj*c_time=-0.001*5 per second; red violation -0.25-0.02*v[m/s].",
                   "transfer": "The time component is integrated over dt. Red crossing terminates with Zhu's red-violation term, not the collision penalty. Other unsuccessful terminations receive no success bonus or extra penalty. Combining these components, terminal traffic-rule definitions and route task boundaries are transfer choices.",
                   "full_route": "Success +1 is issued after each cleared junction, reusing the single-intersection task on a route. This repetition is a transfer choice. Overspeed uses Zhu equation (25), coefficient -0.002 times excess m/s.",
                   "excluded_candidates": "Zhu's fuel, battery and missed-green branches are omitted; this is not its full HEV reward. No additional hand-tuned coefficients are introduced."},
        "actions": "Isele sequential accelerate/coast/brake combined with Jiang left/keep/right. Cartesian 9-action discretization is local; numerical acceleration bounds and duration are Jiang values.",
        "lane_duration_semantics": "Jiang's 3 s value is applied to the TraCI changeLane request duration (request persistence). SUMO's default lateral motion model is retained; this is not a reproduction of Jiang's 3 s lane-changing trajectory.",
        "control": "speedMode=96, laneChangeMode=0; setSpeed(v+a*dt), physical bounds only; no automatic longitudinal safety filter. Lane commands only constrained by lane existence/junction geometry.",
        "implementation_choices": "Adam and framework initialization follow the existing paper runner. Seed, training budget, reset sampling and scenario selection are recorded per run. Floating-point tolerances are implementation details.",
        "framework_defaults": {"source": "Existing runner / PyTorch defaults, not verified paper values",
                               "adam_betas": [.9, .999], "adam_epsilon": 1e-8,
                               "adam_weight_decay": 0.,
                               "initialization": "torch.nn.Linear.reset_parameters (default Kaiming-uniform weights and uniform biases)",
                               "advantage": "Per-update normalization; denominator floor 1e-6",
                               "critic": "Separate Adam optimizer, mean squared error; no value clipping",
                               "gradient_clipping": False,
                               "resume": "Weights-only warm start; Adam state is reset at each training stage"},
        "full_route": "Original ego_loop edge sequence, departPos and arrivalPos 48.90476674674972 m, and arrivalLane 0 are loaded from seongsan.rou.xml. Starting lane is sampled among lanes connected to the next route edge. The 500 s budget is a course setting.",
        "signal_curriculum": "Ego starts 5..50 m before the signal and finishes one vehicle length (5 m) along the outgoing edge. Static red/green episodes have Isele's 20 s horizon. Initial speed follows Zhu's U(0,v_limit), restricted to physically stoppable state pairs. Reset/goal and static-phase choices are local curriculum design, not the original experiments.",
        "junction_curriculum": "Sample existing route junctions with the same inputs, start 5..50 m before them and finish 5 m on the outgoing edge. Natural signals, stop/merge rules, and background traffic. Warm-up U(150,300) s comes from Jiang section 5.1. Sampling route junctions is a local curriculum choice.",
        "stop_curriculum": "In training only, alternate moving starts with Isele-style stopped starts one vehicle length before mandatory stop lines. Own stopping history includes the observed initial stop. This is a declared local curriculum; reward coefficients are unchanged. Junction training reuses live background until the route-file flow horizon; evaluation resets independently.",
        "stop_subgoal": "First commanded full stop within one vehicle length of a mandatory stop line is a successful subtask (+1, same Isele success coefficient). It is rewarded once per node, requires requested speed zero, and is not awarded for blocked routes. Defining this additional subtask is Seongsan design, not a verbatim reward from Isele.",
        "stop_preview": "Map control points are queried along the selected route up to the existing 50 m observation range, including signs beyond short connectors. Rule-violation detection still uses the actual node being crossed. This lookahead definition is Seongsan design; it supplies observations without controlling the car.",
        "mixed_curriculum": "After full-route-only training showed signal regression, one of four workers is assigned to each of signal, mandatory-stop, local-junction and full-route tasks. Each contributes 512 transitions per 2048-step update. This equal task sampling and additional 500,000-step budget are Seongsan choices, not paper-prescribed parameters. Numerical reward coefficients and PPO settings remain fixed.",
    }
    if numeric_correction:
        result["status"] = "Numeric correction of the existing Seongsan PPO; architecture, observations, actions and map retained. Not a whole-paper replication."
        result["ppo"]["transfer"] = "Existing 200-input, 9-action categorical PPO retained. T=2048 is collected per worker; update batch is N*T as in PPO Algorithm1."
        result["reward"]["version"] = "existing_ppo_numeric_correction_v1"
        result["reward"]["transfer"] = "Existing published-component combination retained. Zhu's time term -0.001*5 is applied per step without the previously added dt rescaling. No junction-crossing or stop subtask bonus."
        result["reward"]["full_route"] = "Isele +1 only at final task success, -10 for collision; existing Zhu red/overspeed coefficients unchanged."
        result["stop_subgoal"] = "Removed from reward; stopping history remains an unchanged observation/rule-evaluation feature."
        result["mixed_curriculum"] = "Not used. Existing full-route task only, no task mixture or distance curriculum."
        result["lane_duration_semantics"] = "The existing 3 s value is applied to SUMO lanechange.duration as well as TraCI request duration."
        result["framework_defaults"].update(source="PPO author Baselines ppo1 reference numerical settings, ported to the existing PyTorch PPO implementation",
                                            adam_epsilon=1e-5,
                                            initialization="normc: hidden/value output norm1.0, policy output norm0.01, zero bias")
        result["scope"] = "User explicitly requested correction of prior PPO numbers, not replacement by SAC or a new environment. Prior Seongsan input/action mapping and preprocessing are retained."
    if failure_time_tail:
        result["failure_time_tail"] = {
            "status": "Experimental Seongsan design; not prescribed by a driving paper",
            "definition": "On failure, add sum(gamma**j * existing_time_cost, j=1..H-k); H is the existing episode limit and k the failure step.",
            "purpose": "Remove the time-cost saving from early failure, while retaining the existing PPO, dynamics, observations, actions and other rewards.",
            "new_independent_reward_coefficients": False,
            "unchanged": "Success and timeout handling; stopping and junction bonuses; true failure remains terminal with no critic bootstrap.",
            "limitations": "This only makes the discounted time-cost component independent of failure time. It does not prove optimal safe driving, resolve the unobserved finite-horizon timer, or reproduce DAC/Pardo algorithms.",
        }
    return result


class ObservedDriving:
    """One learned policy commands all ego acceleration and lane decisions."""
    groups = 1
    discrete = 9
    action_dim = 1

    def __init__(self, output, seed=23, stage="signal", traffic=0., gui=False, reuse_traffic=False,
                 numeric_correction=False, failure_time_tail=False):
        import sumolib
        self.s = SETTINGS
        self.p = PPO_PROFILE
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.stage, self.traffic, self.gui = stage, float(traffic), gui
        self.reuse_traffic = reuse_traffic
        self.numeric_correction = numeric_correction
        self.failure_time_tail = failure_time_tail
        if numeric_correction and failure_time_tail:
            raise ValueError("Do not combine numeric correction and the single-factor failure-time experiment.")
        self.curriculum_distance = self.s.front_range
        self.episode_number = 0
        self.ego_known = False
        self.net = sumolib.net.readNet(str(MAP / "seongsan.net.xml"), withInternal=True)
        original = ET.parse(MAP / "seongsan.rou.xml").getroot()
        original_ego = original.find("./vehicle[@id='ego']")
        self.full_start = float(original_ego.get("departPos", "0"))
        self.full_arrival = original_ego.get("arrivalPos", "max")
        self.full_arrival_lane = original_ego.get("arrivalLane", "current")
        self.full_route = original.find("./route[@id='ego_loop']").get("edges").split()
        self.route = (["south_straight", "south_connector"] if stage == "signal" else self.full_route)
        self.route_file = self.output / "traffic.rou.xml"
        self._write_routes(original)
        self.sim = None
        self.insertion_failures = 0
        self.names = self._feature_names()
        self.obs_dim = len(self.names)
        self.maximum_steps = (self.s.stage_steps if stage in ("signal", "junction", "stop")
                              else round(self.s.full_seconds / self.s.dt))

    @staticmethod
    def _feature_names():
        names = ["ego_distance", "ego_speed", "ego_accel", "ego_lateral_position", "ego_lateral_speed"]
        names += [f"{car}_{field}" for car in ("front", "rear", "adjacent_front", "adjacent_rear")
                  for field in ("distance", "speed", "accel", "lateral_position")]
        names += [f"cross_grid_{forward}_{lateral}_{field}" for forward in range(5)
                  for lateral in range(11) for field in ("heading", "speed", "occupied")]
        names += ["signal_present", "signal_red", "signal_yellow", "signal_green",
                  "signal_distance", "time_to_green"]
        names += ["route_right", "route_current", "route_left", "distance_to_control",
                  "mandatory_stop", "yield_connection", "stop_completed"]
        names += ["ego_speed_limit"]
        return names

    def _write_routes(self, original):
        root = ET.Element("routes")
        for element in original:
            if element.tag in ("vType", "vTypeDistribution", "route"):
                root.append(element)
        ego_type = root.find("./vType[@id='ego_type']")
        ego_type.attrib.update(accel=str(self.s.accel), decel=str(self.s.brake),
                               emergencyDecel=str(self.s.brake), maxSpeed=str(self.s.speed),
                               length=str(self.s.length), speedFactor="1", speedDev="0")
        root.find("./route[@id='ego_loop']").set("edges", " ".join(self.route))
        if self.traffic:
            for flow in original.findall("flow"):
                if "vehsPerHour" in flow.attrib:
                    flow.set("vehsPerHour", str(float(flow.get("vehsPerHour")) * self.traffic))
                root.append(flow)
        # Ego is inserted explicitly after initialization, with its controls set before moving.
        ET.ElementTree(root).write(self.route_file, encoding="utf-8", xml_declaration=True)

    def reset(self, *, seed=None, light=None, distance=None, initial_speed=None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        for _ in range(self.s.stage_steps):
            try:
                return self._reset_once(light, distance, initial_speed, seed is not None)
            except EgoInsertionError:
                self.insertion_failures += 1
        raise EgoInsertionError("Repeatedly blocked ego insertion; no usable reset was found.")

    def _reset_once(self, light, distance, initial_speed, force_reload):
        reuse_scene = bool(self.sim is not None and self.stage in ("junction", "stop", "full") and self.reuse_traffic
                           and not force_reload and self.sim.simulation.getTime() < 3600 - self.s.stage_steps * self.s.dt)
        if self.stage in ("junction", "stop"):
            eligible = [i for i, edge in enumerate(self.full_route[:-1])
                        if (self.stage == "stop" and self.net.getEdge(edge).getToNode().getType() == "allway_stop")
                        or (self.stage != "stop" and self.net.getEdge(edge).getLength() >= 2 * self.s.length)]
            index = int(self.rng.choice(eligible))
            first = index
            if self.stage == "stop" and (self.episode_number % 2 or self.net.getEdge(self.full_route[index]).getLength() < 2 * self.s.length):
                first = max(0, index - 1)
            self.route = self.full_route[first:index + 2]
            if not reuse_scene:
                self._write_routes(ET.parse(MAP / "seongsan.rou.xml").getroot())
        options = ["-n", str(MAP / "seongsan.net.xml"), "-r", str(self.route_file),
                   "--step-length", str(self.s.dt), "--seed", str(self.rng.integers(1, 2**30)),
                   "--time-to-teleport", "-1", "--collision.action", "warn",
                   "--collision.check-junctions", "true", "--collision.mingap-factor", "0",
                   "--no-step-log", "true", "--no-warnings", "true", "--start"]
        if self.numeric_correction:
            options.extend(["--lanechange.duration", str(self.s.lane_duration)])
        if self.sim is None:
            if self.gui:
                import traci
                from sumolib import checkBinary
                traci.start([checkBinary("sumo-gui"), *options], doSwitch=False, label=str(self.output))
                self.sim = traci.getConnection(str(self.output))
            else:
                self.sim = load_sumo()
                self.sim.start(["sumo", *options])
        elif not reuse_scene:
            self.sim.load(options)
        if not reuse_scene:
            self.ego_known = False
        self.v = self.sim.vehicle
        if self.traffic and not reuse_scene:
            warmup_steps = round(self.rng.uniform(150., 300.) / self.s.dt)
            for _ in range(warmup_steps):
                self.sim.simulationStep()
        initial_distance = (float(distance) if distance is not None else
                            self.rng.uniform(self.s.length, self.curriculum_distance))
        short_task = self.stage in ("signal", "junction", "stop")
        position = max(0., self.net.getEdge(self.route[0]).getLength() - initial_distance) if short_task else self.full_start
        initial_distance = self.net.getEdge(self.route[0]).getLength() - position
        stopped_training_start = bool(self.stage in ("junction", "stop") and self.reuse_traffic
                                     and self.episode_number % 2 == 0
                                     and self.net.getEdge(self.route[0]).getToNode().getType() == "allway_stop")
        if stopped_training_start:
            initial_distance = self.s.length
            position = self.net.getEdge(self.route[0]).getLength() - initial_distance
        departure_limit = min(self.s.speed, self.net.getEdge(self.route[0]).getSpeed())
        start_speed = (float(initial_speed) if initial_speed is not None else
                       float(self.rng.uniform(0, departure_limit)) if short_task else 0.)
        if stopped_training_start:
            start_speed = 0.
        if initial_speed is None and short_task:
            while start_speed**2 / (2 * self.s.brake) + start_speed * self.s.dt >= initial_distance:
                start_speed = float(self.rng.uniform(0, departure_limit))
        edge = self.net.getEdge(self.route[0])
        allowed_lanes = [lane.getIndex() for lane in edge.getLanes()
                         if any(c.getTo().getID() == self.route[1] for c in lane.getOutgoing())]
        if not allowed_lanes:
            raise RuntimeError(f"No connection for reset route {self.route[:2]}")
        departure_lane = str(self.rng.choice(allowed_lanes))
        route_id = "ego_loop"
        if reuse_scene:
            if self.ego_known:
                self.v.remove("ego")
            self.episode_number += 1
            route_id = f"ego_episode_{self.episode_number}"
            self.sim.route.add(route_id, self.route)
        self.v.add("ego", route_id, typeID="ego_type", departLane=departure_lane, departPos=str(position), departSpeed=str(start_speed),
                   arrivalPos=str(self.s.length) if short_task else self.full_arrival,
                   arrivalLane="current" if short_task else self.full_arrival_lane)
        self.ego_known = True
        self.v.setSpeedMode("ego", 96)
        self.v.setLaneChangeMode("ego", 0)
        self.v.setSpeed("ego", start_speed)
        self.sim.simulationStep()
        for _ in range(self.s.stage_steps):
            if "ego" in self.v.getIDList():
                break
            self.sim.simulationStep()
        else:
            raise EgoInsertionError(f"Ego insertion blocked on {self.route[0]}; no transition collected.")
        self.v.setSpeedMode("ego", 96)
        self.v.setLaneChangeMode("ego", 0)
        self.steps = 0
        self.start_distance = self.v.getDistance("ego")
        self.last_distance = self.start_distance
        self.stop_completed = set()
        initial_junction = self.control_ahead()
        if initial_junction["stop"] and initial_junction["distance"] <= self.s.length and self.v.getSpeed("ego") <= 1e-6:
            self.stop_completed.add(initial_junction["node"])
        self.forced_light = light
        if self.stage == "signal":
            self.force_light(light or self.rng.choice(["red", "green"]))
        self.initial_light = self.signal()["state"]
        self.previous_accel = 0.
        return self.observation()

    def force_light(self, color):
        state = self.sim.trafficlight.getRedYellowGreenState("signal_south")
        self.sim.trafficlight.setRedYellowGreenState("signal_south", ("r" if color == "red" else "G") * len(state))

    def signal(self):
        tls = self.v.getNextTLS("ego")
        if not tls or tls[0][2] > self.s.communication_range:
            return {"present": False, "distance": self.s.communication_range, "state": "", "to_green": 0.}
        ident, link, distance, color = tls[0]
        remaining = max(0., self.sim.trafficlight.getNextSwitch(ident) - self.sim.simulation.getTime())
        # Frozen phase experiments provide only current color; no fictitious change time.
        to_green = 0. if color in "gG" else -self.s.cycle_scale
        if color not in "gG" and self.forced_light is None and self.stage != "signal":
            logic = self.sim.trafficlight.getAllProgramLogics(ident)[0]
            index = self.sim.trafficlight.getPhase(ident)
            to_green = remaining
            for offset in range(1, len(logic.phases) + 1):
                phase = logic.phases[(index + offset) % len(logic.phases)]
                if phase.state[link] in "gG":
                    break
                to_green += phase.duration
        return {"present": True, "distance": max(0., distance), "state": color, "to_green": to_green}

    def junction(self):
        lane_id = self.v.getLaneID("ego")
        lane = self.net.getLane(lane_id)
        if lane.getEdge().getID().startswith(":"):
            return {"distance": self.s.front_range, "node": "", "stop": False, "yield": False}
        links = self.v.getNextLinks("ego")
        state = links[0][5] if links else ""
        node = lane.getEdge().getToNode()
        return {"distance": max(0., lane.getLength() - self.v.getLanePosition("ego")),
                "node": node.getID(), "stop": node.getType() == "allway_stop", "yield": state in ("m", "s")}

    def lane_reaches_next_route_edge(self, lane_id):
        next_index = self.v.getRouteIndex("ego") + 1
        if next_index >= len(self.route):
            return (self.stage != "full" or self.full_arrival_lane == "current"
                    or self.net.getLane(lane_id).getIndex() == int(self.full_arrival_lane))
        return any(c.getTo().getID() == self.route[next_index]
                   for c in self.net.getLane(lane_id).getOutgoing())

    def control_ahead(self):
        """A stop sign can be beyond a short connector, within the same 50 m view."""
        immediate = self.junction()
        for edge_id in self.route[max(0, self.v.getRouteIndex("ego")):]:
            edge = self.net.getEdge(edge_id)
            node = edge.getToNode()
            if node.getType() != "allway_stop":
                continue
            distance = self.v.getDrivingDistance("ego", edge_id, edge.getLength())
            if 0 <= distance <= self.s.front_range:
                return {"distance": distance, "node": node.getID(), "stop": True, "yield": immediate["yield"]}
        return immediate

    def observation(self):
        v, s = self.v, self.s
        speed = v.getSpeed("ego")
        lane_index = v.getLaneIndex("ego")
        road = v.getRoadID("ego")
        ids = v.getIDList()
        obs = [(v.getDistance("ego") - self.start_distance) / max(s.front_range, sum(self.net.getEdge(e).getLength() for e in self.route)),
               speed / s.speed, v.getAcceleration("ego") / s.brake,
               v.getLateralLanePosition("ego") / self.net.getLane(v.getLaneID("ego")).getWidth(),
               v.getLateralSpeed("ego") / s.speed]
        front, rear = v.getLeader("ego", s.front_range), v.getFollower("ego", s.rear_range)
        adjacent_front = adjacent_rear = None
        if not road.startswith(":"):
            edge = self.net.getEdge(road)
            adjacent = lane_index + (1 if lane_index + 1 < edge.getLaneNumber() else -1)
            if 0 <= adjacent < edge.getLaneNumber():
                at = v.getLanePosition("ego")
                candidates = [(v.getLanePosition(i) - at, i) for i in ids
                              if v.getRoadID(i) == road and v.getLaneIndex(i) == adjacent]
                ahead = sorted((d, i) for d, i in candidates if 0 <= d <= s.front_range)
                behind = sorted((-d, i) for d, i in candidates if -s.rear_range <= d < 0)
                if ahead:
                    d, i = ahead[0]
                    adjacent_front = (i, max(0., d - v.getLength(i)))
                if behind:
                    d, i = behind[0]
                    adjacent_rear = (i, max(0., d - s.length))
        for item, sign in ((front, 1), (rear, -1), (adjacent_front, 1), (adjacent_rear, -1)):
            if item and item[0]:
                i, distance = item
                obs += [sign * distance / (s.front_range if sign > 0 else s.rear_range),
                        v.getSpeed(i) / s.speed, v.getAcceleration(i) / s.brake,
                        float(v.getLaneIndex(i) - lane_index)]
            else:
                obs += [float(sign), 0., 0., 0.]
        grid = np.zeros((5, 11, 3), dtype=np.float32)
        occupied_distance = np.full((5, 11), np.inf)
        x, y = v.getPosition("ego")
        heading = math.radians(v.getAngle("ego"))
        for i in ids:
            if i == "ego":
                continue
            px, py = v.getPosition(i)
            dx, dy = px - x, py - y
            forward = dx * math.sin(heading) + dy * math.cos(heading)
            lateral = dx * math.cos(heading) - dy * math.sin(heading)
            if 0 <= forward < 20 and -90 <= lateral < 90:
                row, col = min(4, int(forward / 4)), min(10, int((lateral + 90) / (180 / 11)))
                distance = math.hypot(dx, dy)
                if distance < occupied_distance[row, col]:
                    occupied_distance[row, col] = distance
                    angle = ((v.getAngle(i) - v.getAngle("ego") + 180) % 360 - 180) / 180
                    grid[row, col] = [angle, v.getSpeed(i) / 20., 1.]
        obs += grid.ravel().tolist()
        signal = self.signal()
        color = signal["state"]
        obs += [float(signal["present"]), float(color in ("r", "R")), float(color in ("y", "Y")),
                float(color in ("g", "G")), signal["distance"] / s.communication_range,
                signal["to_green"] / s.cycle_scale]
        junction = self.control_ahead()
        for direction in (-1, 0, 1):
            candidate = lane_index + direction
            usable = False
            if not road.startswith(":"):
                edge = self.net.getEdge(road)
                next_index = v.getRouteIndex("ego") + 1
                target = self.route[next_index] if next_index < len(self.route) else None
                if 0 <= candidate < edge.getLaneNumber():
                    usable = self.lane_reaches_next_route_edge(edge.getLane(candidate).getID())
            obs.append(float(usable))
        obs += [min(junction["distance"] / s.front_range, 1.), float(junction["stop"]),
                float(junction["yield"]), float(junction["node"] in self.stop_completed)]
        obs += [self.net.getLane(v.getLaneID("ego")).getSpeed() / s.speed]
        out = np.asarray(obs, dtype=np.float32)
        if out.shape != (self.obs_dim,) or not np.isfinite(out).all():
            raise RuntimeError(f"Invalid observation shape or value: {out.shape}")
        return np.clip(out, -1., 1.)[None, :]

    @staticmethod
    def decode(action):
        index = int(np.ravel(action)[0])
        if not 0 <= index < 9:
            raise ValueError(f"Unknown action index {index}")
        longitudinal, lateral = divmod(index, 3)
        return (-SETTINGS.brake, 0., SETTINGS.accel)[longitudinal], (-1, 0, 1)[lateral]

    def step(self, action):
        acceleration, direction = self.decode(action)
        speed = self.v.getSpeed("ego")
        before_signal, before_junction = self.signal(), self.junction()
        road = self.v.getRoadID("ego")
        lane = self.v.getLaneIndex("ego")
        before_control = self.control_ahead()
        if before_control["stop"] and before_control["distance"] <= self.s.length and speed <= 1e-6:
            self.stop_completed.add(before_control["node"])
        limit = self.net.getLane(self.v.getLaneID("ego")).getSpeed()
        requested = max(0., min(self.s.speed, speed + acceleration * self.s.dt))
        self.v.setSpeed("ego", requested)
        lane_applied = False
        if not road.startswith(":"):
            lanes = self.net.getEdge(road).getLaneNumber()
            target = lane + direction
            if 0 <= target < lanes:
                self.v.changeLane("ego", target, self.s.lane_duration)
                lane_applied = direction != 0
        self.sim.simulationStep()
        self.steps += 1
        present = "ego" in self.v.getIDList()
        arrived = "ego" in self.sim.simulation.getArrivedIDList()
        if not present:
            self.ego_known = False
        collision = "ego" in self.sim.simulation.getCollidingVehiclesIDList()
        actual = self.v.getSpeed("ego") if present else None
        distance = self.v.getDistance("ego") if present else self.last_distance if hasattr(self, "last_distance") else 0.
        # Distance traveled in this step includes junction entry and identifies stop-line crossing.
        travelled = distance - self.last_distance if hasattr(self, "last_distance") else actual * self.s.dt
        red_violation = bool(before_signal["present"] and before_signal["state"] in ("r", "R")
                             and travelled > before_signal["distance"] + 1e-6)
        crossed_junction = present and self.v.getRoadID("ego") != road
        cleared_junction = bool(crossed_junction and not self.v.getRoadID("ego").startswith(":"))
        stop_violation = bool(before_junction["stop"] and crossed_junction
                              and before_junction["node"] not in self.stop_completed)
        self.last_distance = distance
        blocked_route = bool(present and self.v.getRoadID("ego") == road and actual <= 1e-6
                             and self.junction()["distance"] <= 1e-6
                             and not self.lane_reaches_next_route_edge(self.v.getLaneID("ego")))
        bad_route = (not present and not arrived and not collision) or blocked_route
        failure = collision or red_violation or stop_violation or bad_route
        successful_stop = False
        if present and not failure and actual <= 1e-6 and requested <= 1e-6:
            after_junction = self.control_ahead()
            if (after_junction["stop"] and after_junction["distance"] <= self.s.length
                    and after_junction["node"] not in self.stop_completed):
                self.stop_completed.add(after_junction["node"])
                successful_stop = True
        outcome = ("collision" if collision else "red_violation" if red_violation else
                   "stop_violation" if stop_violation else "route_failure" if bad_route else
                   "complete" if arrived else "timeout" if self.steps >= self.maximum_steps else "running")
        done = failure or arrived
        truncated = self.steps >= self.maximum_steps and not done
        reward = self.s.time_objective * self.s.time_weight * (1. if self.numeric_correction else self.s.dt)
        if collision:
            reward += self.s.collision_reward
        elif red_violation:
            reward += self.s.red_base + self.s.red_speed * actual
        elif not failure and (arrived or (self.stage == "full" and cleared_junction and not self.numeric_correction)):
            reward += self.s.success_reward
        if successful_stop and not self.numeric_correction:
            reward += self.s.success_reward
        if self.stage == "full" and actual is not None and not failure:
            reward += self.s.overspeed * max(0., actual - limit)
        base_reward = reward
        tail_cost = (discounted_failure_time_cost(
            self.s.time_objective * self.s.time_weight * self.s.dt,
            self.p.gamma, self.maximum_steps - self.steps)
            if self.failure_time_tail and failure else 0.)
        reward += tail_cost
        obs = np.zeros((1, self.obs_dim), dtype=np.float32) if done else self.observation()
        info = {"outcome": outcome, "collision": bool(collision), "success": bool(arrived and not failure),
                "red_violation": red_violation, "stop_violation": stop_violation,
                "base_reward": base_reward, "failure_time_tail_cost": tail_cost,
                "blocked_route": blocked_route,
                "successful_stop_subgoal": successful_stop,
                "requested_accel": acceleration, "requested_speed": requested, "actual_speed": actual,
                "speed_error": abs(actual - requested) if actual is not None else None,
                "speed_mode": self.v.getSpeedMode("ego") if present else 96,
                "lane_change_requested": direction, "lane_change_issued": lane_applied,
                "lane_change_applied": bool(present and self.v.getRoadID("ego") == road
                                            and self.v.getLaneIndex("ego") != lane),
                "signal": before_signal, "junction": before_junction, "mean_speed": actual or 0.,
                "steps": self.steps, "initial_light": self.initial_light}
        if info["speed_error"] is not None and info["speed_error"] > 1e-4:
            import json
            diagnostic = {**info, "road_before": road,
                          "road_after": self.v.getRoadID("ego") if present else None,
                          "lane_before": lane, "lane_after": self.v.getLaneIndex("ego") if present else None,
                          "route": self.route}
            with (self.output / "speed_mismatches.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(diagnostic) + "\n")
        return obs, [reward], done, truncated, info

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
