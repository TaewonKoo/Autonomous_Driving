"""Training-task distribution for traffic-rule exposure; observations, actions and reward coefficients unchanged.

Adds two task variants on top of ObservedDriving:
* signal: every other episode runs the real signal cycle from a random point instead of a frozen
  colour, so red states carry a positive time-to-green exactly as on the full route.
* speed: short two-edge tasks that start on, or just before, an edge whose limit is below the ego
  maximum speed. The existing Zhu overspeed term (-0.002 per m/s over the limit) is applied, which
  the base environment only applies on the full route.
These are Seongsan task-sampling choices, not settings taken from a paper.
"""
from pathlib import Path

import numpy as np

from . import observed_driving
from .observed_driving import ObservedDriving


MAPS = {
    "original": observed_driving.MAP,  # whatever maps/seongsan/generated currently holds
    # Same nodes, edges, connections and signal program; every lane 50 km/h, no curvature turn-speed limit.
    "nolimit": Path(observed_driving.MAP).parent / "generated_nolimit",
    # Frozen copy of GPT's +1-lane map (all lanes 60 km/h), applied 2026-11-24 01:39; see SNAPSHOT.json.
    "pluslane": Path(observed_driving.MAP).parent / "generated_pluslane_20261124",
    # Frozen copy of the +1-lane map after Claude's marking/geometry fix (2026-11-24 02:40).
    "pluslane_fixed": Path(observed_driving.MAP).parent / "generated_pluslane_fixed_20261124",
    # + smoothed northern coast road and cleaned partial stop lines (2026-11-24).
    "pluslane_v3": Path(observed_driving.MAP).parent / "generated_pluslane_v3_20261124",
    # + road priorities restored (v3 had main/side priorities inverted by the +1-lane edit).
    "pluslane_v4": Path(observed_driving.MAP).parent / "generated_pluslane_v4_20261124",
}


def use_map(name):
    """Point ObservedDriving at a map variant (per process; call in every worker)."""
    observed_driving.MAP = MAPS[name]
    return MAPS[name]


OLD_ROUTE_NAMES = ("route_right", "route_current", "route_left")
# Coarse routing information allowed by the course TA (2026-11-23): whether the current lane ends for
# the route within 50 m, and distance/direction to the next fork. Looks ahead along the route.
COARSE_ROUTE_NAMES = ("route_lane_usable_distance", "route_needed_direction", "route_next_fork_distance")
ROUTE_LOOKAHEAD = 50.


CONFLICT_SLOTS, CONFLICT_RANGE, TTA_SCALE, REAR_RANGE = 3, 60., 10., 40.
EXTRA_NAMES = ([f"conflict_{k}_{f}" for k in range(CONFLICT_SLOTS) for f in ("distance", "speed", "time_to_arrival", "has_priority", "direction")]
               + ["ego_has_priority"]
               + [f"other_adjacent_{car}_{f}" for car in ("front", "rear") for f in ("distance", "speed", "accel", "lateral_position")]
               + [f"rear40_{lane}_{f}" for lane in ("own", "left", "right") for f in ("distance", "speed")])
# Junction foes (own design, extra_obs=2): the nearest vehicles whose junction connection conflicts with the
# ego's next connection according to SUMO's junction foe matrix, including vehicles already standing inside
# the junction (the conflict inputs above give those only distance 0, whatever their path).
FOE_SLOTS = 3
FOE_NAMES = [f"foe_{k}_{f}" for k in range(FOE_SLOTS) for f in ("distance", "speed", "inside_junction", "ego_must_yield")]
# extra_obs=3: where the two paths cross (centre lines of the SUMO internal lanes): the ego's distance along its path
# to that point and the foe's time to reach it at its current speed.
CONFLICT_POINT_NAMES = [f"foe_{k}_{f}" for k in range(FOE_SLOTS) for f in ("ego_to_crossing", "time_to_crossing")]


def history_lags_from_names(names):
    return tuple(sorted({int(n.rsplit("@-", 1)[1]) for n in names if "@-" in n}))


def env_options_from_names(names):
    """Observation options that reproduce a checkpoint's input list."""
    return {"route_obs": "coarse" if "route_lane_usable_distance" in names else "lane",
            "signal_obs": "color" if "time_to_green_unused" in names else "color_time",
            "extra_obs": 3 if "foe_0_ego_to_crossing" in names else 2 if "foe_0_distance" in names
            else int("conflict_0_distance" in names),
            "history_lags": history_lags_from_names(names),
            "input_names": list(names)}


class _FogView:
    """Vehicle-domain view used while building the observation in fog (own design): every other vehicle whose
    centre is farther than `visibility` metres from the ego centre is invisible to the ego's sensors."""

    def __init__(self, real, visible):
        self._real, self._visible = real, visible

    def __getattr__(self, name):
        return getattr(self._real, name)

    def getIDList(self):
        return [i for i in self._real.getIDList() if i in self._visible]

    def getLeader(self, vid, dist=0.):
        r = self._real.getLeader(vid, dist)
        return r if (r and r[0] and r[0] in self._visible) else None

    def getFollower(self, vid, dist=0.):
        r = self._real.getFollower(vid, dist)
        return r if (r and r[0] and r[0] in self._visible) else ("", -1.)


class _FirstChoice:
    """Delegate to a numpy Generator, but answer the first choice() call from an allowed subset."""

    def __init__(self, rng, allowed):
        self.rng, self.allowed, self.used = rng, set(allowed), False

    def choice(self, options, *args, **kwargs):
        if not self.used:
            self.used = True
            keep = [x for x in options if int(x) in self.allowed]
            if not keep:
                raise RuntimeError("No eligible low-speed task edge")
            return self.rng.choice(keep)
        return self.rng.choice(options, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.rng, name)


class _AnyLane:
    """For junction/stop tasks the 2nd choice() in ObservedDriving._reset_once picks the departure
    lane among lanes that continue on the route; answer it from all lanes of the start edge."""

    def __init__(self, rng, lanes):
        self.rng, self.lanes, self.calls = rng, lanes, 0

    def choice(self, options, *args, **kwargs):
        self.calls += 1
        if self.calls == 2:
            return self.rng.choice(list(self.lanes()))
        return self.rng.choice(options, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.rng, name)


class _Focus:
    """1st choice(): the task edge, restricted to `allowed` route indices; 2nd choice(): optionally any lane."""

    def __init__(self, rng, allowed, lanes=None):
        self.rng, self.allowed, self.lanes, self.calls = rng, set(allowed), lanes, 0

    def choice(self, options, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            keep = [x for x in options if int(x) in self.allowed]
            if keep:
                return self.rng.choice(keep)
        if self.calls == 2 and self.lanes is not None:
            return self.rng.choice(list(self.lanes()))
        return self.rng.choice(options, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.rng, name)


class RuleCurriculumDriving(ObservedDriving):
    tasks = ("signal", "stop", "speed", "full", "junction")

    # Driver Dojo Eq. (1), arXiv 2207.11432 p.8: +5 sub-goal, +10 goal, -10 crash/off-route/non-road, else v/v_max.
    # Seongsan mapping (design choice): red-light and STOP violations are failures (-10); a cleared
    # junction (first entry into a new route edge) is a sub-goal; reaching the task end is the goal.
    DOJO = {"subgoal": 5., "goal": 10., "failure": -10.}

    def __init__(self, output, seed=23, task="signal", traffic=0., signal_seconds=None, reward="base", lane_change_penalty=0., roach_max_speed=None, route_obs="lane", signal_obs="color_time",
                 route_lookahead=ROUTE_LOOKAHEAD, random_start_lane=False, extra_obs=False,
                 yield_failure=False, roach_hazard_scale=1.0, stop_zone=None, blocked_seconds=None, junction_hazard=False, desired_speed="roach", speed_reward="roach", history_lags=(), signal_aware=False, unsafe_lane_change_penalty=0., idle_penalty=0., visibility=None, input_names=None, signal_next_only=False, yield_approach_speed=0., fog_speed_cap=False, fog_yield_scale=False, stop_reward_floor=0., collision_penalty=0., **kwargs):
        # "stop@edgeA+edgeB" / "junction@edge": practise only the junction(s) at the end of these route edges.
        task, _, focus = task.partition("@")
        self.focus_edges = set(focus.split("+")) if focus else set()
        if task not in self.tasks:
            raise ValueError(f"Unknown task {task}")
        if reward not in ("base", "dojo", "dojo_roach"):
            raise ValueError(f"Unknown reward {reward}")
        super().__init__(output, seed, "junction" if task == "speed" else task, traffic, **kwargs)
        self.task = task
        self.reward_mode = reward
        # Liu et al. 2018 (arXiv 1802.00332) Table 1: lane switching -0.4 vs collision -1.0; scaled to -10 -> -4.
        self.lane_change_penalty = float(lane_change_penalty)
        # Roach (Zhang et al. ICCV 2021) author code valeo_action.py: maximum speed 6.0 m/s.
        self.roach_max_speed = float(roach_max_speed) if roach_max_speed else 6.0
        if route_obs not in ("lane", "coarse"):
            raise ValueError(f"Unknown route_obs {route_obs}")
        self.route_obs = route_obs
        self.route_idx = [self.names.index(n) for n in OLD_ROUTE_NAMES]
        if route_obs == "coarse":
            self.names = list(self.names)
            for k, name in zip(self.route_idx, COARSE_ROUTE_NAMES):
                self.names[k] = name
        self._last_route = [1., 0., 1.]
        if signal_obs not in ("color_time", "color"):
            raise ValueError(f"Unknown signal_obs {signal_obs}")
        # color: the policy sees only the light colour (and distance), as a camera would; the
        # remaining time to green (V2I SPaT) is held at 0 in an unused, renamed input slot.
        self.signal_obs = signal_obs
        self.route_lookahead = float(route_lookahead)
        # Junction/stop practice normally starts in a lane that already continues on the route;
        # optionally start in any lane so the policy also practises choosing the lane.
        self.random_start_lane = bool(random_start_lane) and task in ("junction", "stop")
        # Extra perception (appended after the original 200 inputs): conflicting vehicles at the next
        # junction with distance / speed / time-to-arrival / right-of-way (cf. Li & Czarnecki 2019),
        # own right-of-way, the other-side adjacent lane, and followers within 40 m.
        self.extra_obs = int(extra_obs)
        # Failure to yield (own design, cf. Li & Czarnecki 2019 'failure to yield right-of-way'): after
        # entering a junction on a yielding link (m/w/s/g), a vehicle approaching that junction from
        # another road (or already in it) brakes harder than SUMO's default 4.5 m/s^2 -> failure (-10).
        self.yield_failure = bool(yield_failure)
        # Junction hazard in the Roach desired speed (own design): before entering a junction, treat a
        # conflicting vehicle (SUMO junction foe matrix) that is already crossing inside the junction
        # (> 1 m/s, within 30 m) or that has right of way and arrives within 3 s (not at all-way stops,
        # where it must stop) like a red light: the desired speed ramps to 0 five metres before the line.
        # Reward only; the policy's speed request is never altered.
        # junction_hazard=2 (own design): the same idea at the crossing point of the two paths, also while the ego is
        # already inside the junction. A conflicting vehicle (ego must yield to it, or it is already in the junction)
        # is a hazard if it reaches the crossing before the ego can clear it (accelerating at 2 m/s^2) plus 1.5 s;
        # the target is then to stop at the line (outside) or 3 m before the crossing (inside).
        self.junction_hazard = int(junction_hazard)
        # Desired-speed profile for the speed-matching reward.
        #  "roach": Roach hazard ramps (9.5 m / 45 deg cone, 8 m vehicle offset, 5 m line offset).
        #  "idm"  : own design with IDM coefficients (Treiber, Hennecke & Helbing 2000: minimum gap s0 = 2 m,
        #           time headway T = 1.5 s, comfortable deceleration b = 2 m/s^2). Vehicles count only if they
        #           are the lane leader or inside the ego's path corridor (not cars waiting in the next lane);
        #           desired = (gap - s0) / T. Red/yellow lights, uncompleted STOP lines and junction hazards
        #           use the speed from which the ego can stop with b one metre before the line.
        #  "idm_krauss": as "idm", and the vehicle term is also capped by the Krauss (1998) safe speed
        #           v_safe = -b*tau + sqrt((b*tau)^2 + v_l^2 + 2*b*(gap - s0)), tau = 1 s (SUMO default), so the ego
        #           can still stop behind a slow or stopped leader (the time-gap term alone ignores closing speed).
        self.desired_speed = desired_speed
        # Speed reward. "roach": 1 - |v - v_des| / v_max (Roach). "relative" (own design): the error is divided by
        # max(v_des, 2 m/s) instead, so creeping on while the desired speed is ~0 (red light, STOP, junction hazard)
        # costs as much as being far off at cruise; identical to Roach when v_des = v_max. Clipped at -1.
        self.speed_reward = speed_reward
        # Signal-aware conflicts (own design): a vehicle that still has to stop (red light, STOP / all-way stop) is
        # not a conflict until it is within 3 m of its line and moving; at a traffic light, who yields follows the
        # current signal (G has priority over g) instead of SUMO's static junction table.
        self.signal_aware = bool(signal_aware)
        self.signal_next_only = bool(signal_next_only)
        self.yield_approach_speed = float(yield_approach_speed or 0.)
        # Unsafe lane change (MOBIL safety criterion, Kesting, Treiber & Helbing 2007, b_safe = 4 m/s^2): after an
        # applied lane change, if the new follower would need to brake harder than b_safe to keep s0, or the ego would
        # need that against its new leader, add this penalty (reward only; the action is never blocked).
        self.unsafe_lane_change_penalty = float(unsafe_lane_change_penalty)
        # Idle penalty (own design): standing still although the desired speed says go (no hazard, no red light,
        # no close leader, STOP already served) costs this per step, so hesitating on a clear green is not free.
        self.idle_penalty = float(idle_penalty)
        # Fog (own design): perception range limited to `visibility` m for other vehicles and the signal colour;
        # map knowledge (route, STOP/yield control, speed limit) and the ego's own state are unaffected.
        # Outcomes and rewards always use the true state.
        self.visibility = float(visibility) if visibility else None
        # Fog speed (own design, reward only): see idm_desired_speed.
        self.fog_speed_cap = bool(fog_speed_cap)
        self.fog_yield_scale = bool(fog_yield_scale)
        # Own design (reward only): when the desired speed is ~0 (STOP line, queue), divide the speed error by this
        # floor instead of 2 m/s, so creeping at 0.3-2 m/s is no longer scored almost like a full stop.
        self.stop_reward_floor = float(stop_reward_floor or 0.)
        # Collision penalty (own design, reward only): a collision costs this instead of the Driver Dojo failure -10
        # (other failures keep -10); 0 = off.
        self.collision_penalty = float(collision_penalty or 0.)
        self._fog_obs = False
        # Roach's hazard distances were tuned for its 6 m/s cruise. With a faster cruise, scale them by
        # v_max/6 so the following gap keeps Roach's time headway (stop-line offsets stay 5 m).
        self.roach_hazard_scale = float(roach_hazard_scale)
        # A full stop within `stop_zone` m of a STOP line completes the stop (base env: one car length).
        self.stop_zone = stop_zone
        # Roach 'Blocked' criterion (carla-roach criteria/blocked.py): speed < 0.1 m/s for more than
        # 90 s ends the episode as a failure (Roach terminal -1 -> Driver Dojo failure -10).
        self.blocked_seconds = blocked_seconds
        self._blocked_steps = 0
        self._yield_node = None
        self._base_dim = self.obs_dim
        if self.extra_obs:
            self.names = (list(self.names) + EXTRA_NAMES + (FOE_NAMES if self.extra_obs >= 2 else [])
                          + (CONFLICT_POINT_NAMES if self.extra_obs >= 3 else []))
            self.obs_dim = len(self.names)
        # Observation history (frame stacking, Mnih et al. 2015): the full input vector of `lags` steps ago is
        # appended (e.g. 2,4 -> 0.4 s and 0.8 s ago) so the memoryless MLP can see accelerations and who moved first.
        if isinstance(history_lags, str):
            history_lags = tuple(int(x) for x in history_lags.split(",") if x)
        self.history_lags = tuple(history_lags or ())
        self._history = None
        if self.history_lags:
            current = list(self.names)
            self.names = current + [f"{n}@-{lag}" for lag in self.history_lags for n in current]
            self.obs_dim = len(self.names)
        self.time_idx = self.names.index("time_to_green") if "time_to_green" in self.names else self.names.index("time_to_green_unused")
        if signal_obs == "color":
            self.names = list(self.names); self.names[self.time_idx] = "time_to_green_unused"
        self.visited_route_index = None
        if task == "signal" and signal_seconds is not None:
            # Longer signal episodes let the policy experience a full red wait (up to 50 s here).
            self.maximum_steps = round(signal_seconds / self.s.dt)
        self.natural_signal = False
        self.signal_episodes = 0
        # Index i starts a two-edge task [i, i+1]; keep those where either edge is below the ego maximum.
        limit = lambda edge: min(lane.getSpeed() for lane in self.net.getEdge(edge).getLanes())
        self.speed_edges = [i for i in range(len(self.full_route) - 1)
                            if min(limit(self.full_route[i]), limit(self.full_route[i + 1])) < self.s.speed - 1e-6]
        # Reduced input set (weight-preserving pruning): compute the full vector, give the policy only these inputs.
        self._input_idx = None
        # Fog model input (own design): "visibility" = current visibility (min(vis, 200 m) / 200 m, 1 = clear),
        # appended after the selected inputs; it is not part of the full observation vector.
        self.visibility_input = bool(input_names is not None and "visibility" in input_names)
        if self.visibility_input:
            input_names = [n for n in input_names if n != "visibility"]
        if input_names is not None and list(input_names) != list(self.names):
            full = {n: i for i, n in enumerate(self.names)}
            self._input_idx = [full[n] for n in input_names]
            self.names = list(input_names)
            self.obs_dim = len(self.names)
        if self.visibility_input:
            self.names = list(self.names) + ["visibility"]
            self.obs_dim = len(self.names)

    def _reset_once(self, light, distance, initial_speed, force_reload):
        self._history = None
        if self.focus_edges and self.task in ("junction", "stop"):
            if self.sim is not None and self.ego_known and "ego" not in self.sim.vehicle.getLoadedIDList():
                self.ego_known = False
            allowed = [i for i, e in enumerate(self.full_route[:-1]) if e in self.focus_edges]
            rng = self.rng
            lanes = (lambda: range(self.net.getEdge(self.route[0]).getLaneNumber())) if self.random_start_lane else None
            self.rng = _Focus(rng, allowed, lanes)
            try:
                return ObservedDriving._reset_once(self, light, distance, initial_speed, force_reload)
            finally:
                self.rng = rng
        # SUMO silently discards an ego that cannot depart (e.g. a start lane needing too many lane
        # changes in dense traffic); forget it so the base reset does not try to remove it again.
        if self.sim is not None and self.ego_known and "ego" not in self.sim.vehicle.getLoadedIDList():
            self.ego_known = False
        if self.random_start_lane:
            rng = self.rng
            self.rng = _AnyLane(rng, lambda: range(self.net.getEdge(self.route[0]).getLaneNumber()))
            try:
                return super()._reset_once(light, distance, initial_speed, force_reload)
            finally:
                self.rng = rng
        if self.task != "speed":
            return super()._reset_once(light, distance, initial_speed, force_reload)
        rng = self.rng
        self.rng = _FirstChoice(rng, self.speed_edges)
        try:
            return super()._reset_once(light, distance, initial_speed, force_reload)
        finally:
            self.rng = rng

    def force_light(self, color):
        self.signal_episodes += 1
        self.natural_signal = self.task == "signal" and self.forced_light is None and self.signal_episodes % 2 == 0
        if not self.natural_signal:
            return super().force_light(color)
        tls = self.sim.trafficlight
        logic = tls.getAllProgramLogics("signal_south")[0]
        durations = [phase.duration for phase in logic.phases]
        t = float(self.rng.uniform(0, sum(durations)))
        for index, duration in enumerate(durations):
            if t < duration:
                break
            t -= duration
        tls.setPhase("signal_south", index)
        tls.setPhaseDuration("signal_south", duration - t)

    def signal(self):
        result = self._signal_true()
        if getattr(self, "signal_next_only", False) and result["present"]:
            # Own design (observation): only the NEXT traffic control is shown; a signal behind an all-way STOP is hidden.
            control = self.control_ahead()
            if control["stop"] and control["distance"] < result["distance"]:
                return {"present": False, "distance": self.s.communication_range, "state": "", "to_green": 0.}
        if getattr(self, "_fog_obs", False) and result["present"] and result["distance"] > self.visibility:
            return {"present": False, "distance": self.s.communication_range, "state": "", "to_green": 0.}
        return result

    def _signal_true(self):
        result = super().signal()
        if not (self.natural_signal and result["present"] and result["state"] not in "gG"):
            return result
        ident, link, _, color = self.v.getNextTLS("ego")[0]
        logic = self.sim.trafficlight.getAllProgramLogics(ident)[0]
        index = self.sim.trafficlight.getPhase(ident)
        to_green = max(0., self.sim.trafficlight.getNextSwitch(ident) - self.sim.simulation.getTime())
        for offset in range(1, len(logic.phases) + 1):
            phase = logic.phases[(index + offset) % len(logic.phases)]
            if phase.state[link] in "gG":
                break
            to_green += phase.duration
        return {**result, "to_green": to_green}

    def _reset_episode_state(self):
        self.visited_route_index = None
        self._blocked_steps = 0

    def reset(self, **kwargs):
        self._reset_episode_state()
        from paper_training.observed_driving import EgoInsertionError
        for attempt in range(4):
            try:
                return super().reset(**kwargs)
            except EgoInsertionError:
                # The reused traffic scene is jammed at every start point; reload a fresh scene with a new seed.
                if attempt == 3:
                    raise
                kwargs["seed"] = int(self.rng.integers(1 << 31))
                self.insertion_failures += 1

    def _foe_net(self):
        if getattr(self, "_foes", None) is None:
            import sumolib
            from . import observed_driving as od
            self._foes = sumolib.net.readNet(str(Path(od.MAP) / "seongsan.net.xml"), withInternal=True, withFoes=True)
            self._via = {}
            for node in self._foes.getNodes():
                for edge in node.getIncoming():
                    if edge.getID().startswith(":"):
                        continue  # internal-to-internal links are mapped to their road connection below
                    for lane in edge.getLanes():
                        for c in lane.getOutgoing():
                            if c.getViaLaneID():
                                self._via[c.getViaLaneID()] = c
            # later parts of a split internal lane (e.g. the waiting position of a left turn) map to the same connection
            changed = True
            while changed:
                changed = False
                for lane_id, c in list(self._via.items()):
                    for nxt in self._foes.getLane(lane_id).getOutgoing():
                        via = nxt.getViaLaneID()
                        if via and via not in self._via:
                            self._via[via] = c; changed = True
        return self._foes

    def _connection(self, vid):
        """SUMO connection a vehicle is using (inside a junction) or about to use (approaching)."""
        net = self._foe_net()
        lane_id = self.v.getLaneID(vid)
        if lane_id.startswith(":"):
            return self._via.get(lane_id)
        links = self.v.getNextLinks(vid)
        if not links:
            return None
        return next((c for c in net.getLane(lane_id).getOutgoing() if c.getToLane().getID() == links[0][0]), None)

    def _yield_check_before(self):
        road = self.v.getRoadID("ego")
        self._road_before = road
        # junction_hazard>=3 (own design, off by default: SUMO's own driver is flagged at 3.4% of entries): entering while a conflicting vehicle reaches the crossing before the
        # ego could clear it (no margin) is also a failure to yield, checked at the moment of entry.
        self._entry_conflict = (self.junction_hazard >= 3 and not road.startswith(":")
                                and self._crossing_hazard_distance(margin=0.) is not None)
        if road.startswith(":"):
            return
        links = self.v.getNextLinks("ego")
        conn = self._connection("ego") if links and links[0][5] in "mwsg" else None
        self._pending_yield = (self.net.getEdge(road).getToNode().getID(), road, conn) if conn else None

    def _yield_violation_after(self):
        """Failure to yield: a vehicle whose connection has right-of-way over the ego's (SUMO junction
        request table) brakes harder than 4.5 m/s^2 while the ego is inside that junction."""
        if "ego" not in self.v.getIDList():
            self._yield_node = None
            return False
        road = self.v.getRoadID("ego")
        if road.startswith(":") and getattr(self, "_entry_conflict", False) and not getattr(self, "_road_before", ":").startswith(":"):
            return True
        if road.startswith(":"):
            if self._yield_node is None and getattr(self, "_pending_yield", None):
                self._yield_node = self._pending_yield
        else:
            self._yield_node = None
            return False
        if self._yield_node is None:
            return False
        node_id, entry, ego_conn = self._yield_node
        node = self._foe_net().getNode(node_id)
        for vid in self.v.getIDList():
            if vid == "ego" or self.v.getAcceleration(vid) >= -4.5:
                continue
            r = self.v.getRoadID(vid)
            inside = r.startswith(":") and self._node_of_internal(r) == node_id
            approaching = (not r.startswith(":") and r != entry and self.net.getEdge(r).getToNode().getID() == node_id
                           and self.net.getLane(self.v.getLaneID(vid)).getLength() - self.v.getLanePosition(vid) < 30)
            if not (inside or approaching):
                continue
            other = self._connection(vid)
            if other is not None and node.forbids(other, ego_conn):
                return True
        return False

    def step(self, action):
        if self.yield_failure and "ego" in self.v.getIDList():
            self._yield_check_before()
        limit = None
        present_before = "ego" in self.v.getIDList()
        if self.task == "speed" and present_before:
            limit = self.net.getLane(self.v.getLaneID("ego")).getSpeed()
        if present_before and self.visited_route_index is None:
            self.visited_route_index = self.v.getRouteIndex("ego")
        obs, reward, done, truncated, info = super().step(action)
        if self.stop_zone and not done and "ego" in self.v.getIDList() and self.v.getSpeed("ego") < .1:
            control = self.control_ahead()
            if control["stop"] and control["distance"] <= self.stop_zone:
                self.stop_completed.add(control["node"])
        failure = info["collision"] or info["red_violation"] or info["stop_violation"] or (
            info["outcome"] == "route_failure")
        if self.blocked_seconds and not done and not failure and info["actual_speed"] is not None:
            self._blocked_steps = self._blocked_steps + 1 if info["actual_speed"] < .1 else 0
            if self._blocked_steps * self.s.dt > self.blocked_seconds:
                failure, done = True, True
                info.update(outcome="blocked", blocked=True)
                obs = np.zeros((1, self.obs_dim), dtype=np.float32)
        if self.yield_failure and not done and not failure and self._yield_violation_after():
            failure, done = True, True
            info.update(outcome="yield_violation", yield_violation=True)
            obs = np.zeros((1, self.obs_dim), dtype=np.float32)
        if self.reward_mode in ("dojo", "dojo_roach"):
            info["base_reward_unused"] = reward[0]
            subgoal = False
            if not failure and "ego" in self.v.getIDList() and not self.v.getRoadID("ego").startswith(":"):
                index = self.v.getRouteIndex("ego")
                subgoal = index > self.visited_route_index
                self.visited_route_index = max(index, self.visited_route_index)
            if failure:
                value = self.DOJO["failure"]
                if info["collision"] and getattr(self, "collision_penalty", 0.):
                    value = self.collision_penalty
            elif info["success"]:
                value = self.DOJO["goal"]
            elif subgoal:
                value = self.DOJO["subgoal"]
            else:
                value = ((info["actual_speed"] or 0.) / self.s.speed if self.reward_mode == "dojo"
                         else self.roach_speed_reward(info))
            if not failure and info["lane_change_applied"]:
                value += self.lane_change_penalty
                if self.unsafe_lane_change_penalty and self._unsafe_lane_change():
                    value += self.unsafe_lane_change_penalty
                    info["unsafe_lane_change"] = True
            info.update(task=self.task, natural_signal=self.natural_signal, dojo_subgoal=subgoal)
            return obs, [value], done, truncated, info
        if limit is not None and info["actual_speed"] is not None and not failure:
            penalty = self.s.overspeed * max(0., info["actual_speed"] - limit)
            reward = [reward[0] + penalty]
            info["speed_task_overspeed_penalty"] = penalty
        info["task"] = self.task
        info["natural_signal"] = self.natural_signal
        return obs, reward, done, truncated, info

    # ---- Roach speed-matching term (carla-roach valeo_action.py / hazard_actor.py) ----
    def _center_and_heading(self, vid):
        import math
        x, y = self.v.getPosition(vid)
        angle = math.radians(self.v.getAngle(vid))  # SUMO: degrees clockwise from north
        fx, fy = math.sin(angle), math.cos(angle)
        half = self.v.getLength(vid) / 2.
        return (x - fx * half, y - fy * half), (fx, fy), self.v.getAngle(vid)

    IDM_S0, IDM_T, IDM_B, LINE_MARGIN, KRAUSS_TAU = 2.0, 1.5, 2.0, 1.0, 1.0
    MOBIL_B_SAFE = 4.0

    def _required_brake(self, gap, v_back, v_front):
        """Constant deceleration the rear vehicle needs to stay s0 behind the front one."""
        room = gap - self.IDM_S0
        if room <= 0.:
            return float("inf")
        closing = v_back - v_front
        return closing * closing / (2. * room) if closing > 0. else 0.

    def _unsafe_lane_change(self):
        v = self.v
        if "ego" not in v.getIDList():
            return False
        ve = v.getSpeed("ego")
        follower = v.getFollower("ego", 50.)
        if follower and follower[0] and follower[0] != "ego" and follower[1] >= -1e-6:
            if self._required_brake(follower[1], v.getSpeed(follower[0]), ve) > self.MOBIL_B_SAFE:
                return True
        leader = v.getLeader("ego", 50.)
        if leader and leader[0]:
            if self._required_brake(leader[1], ve, v.getSpeed(leader[0])) > self.MOBIL_B_SAFE:
                return True
        return False

    def _vehicle_desired(self, gap, leader_speed):
        free = max(gap - self.IDM_S0, 0.)
        desired = free / self.IDM_T
        if self.desired_speed == "idm_krauss":
            bt = self.IDM_B * self.KRAUSS_TAU
            desired = min(desired, -bt + (bt * bt + leader_speed * leader_speed + 2. * self.IDM_B * free) ** .5)
        return max(desired, 0.)

    def idm_desired_speed(self):
        import math
        v = self.v
        vmax = self.roach_max_speed
        stop_at = lambda d: min(vmax, math.sqrt(2. * self.IDM_B * max(d - self.LINE_MARGIN, 0.)))
        desired = vmax
        leader = v.getLeader("ego", 80.)
        if leader and leader[0]:
            gap = leader[1] + v.getMinGap("ego")  # TraCI leader distance excludes the ego's minGap
            desired = min(desired, self._vehicle_desired(gap, v.getSpeed(leader[0])))
        (ex, ey), (fx, fy), ego_angle = self._center_and_heading("ego")
        ego_front = (ex + fx * v.getLength("ego") / 2., ey + fy * v.getLength("ego") / 2.)
        for vid in v.getIDList():
            if vid == "ego" or (leader and vid == leader[0]):
                continue
            (x, y), _, angle = self._center_and_heading(vid)
            dx, dy = x - ex, y - ey
            forward, left = dx * fx + dy * fy, -dx * fy + dy * fx
            if forward <= 0 or forward > 40.:
                continue
            if abs(left) > (v.getWidth("ego") + v.getWidth(vid)) / 2. + .3:
                continue  # not in the ego's path corridor (e.g. waiting in the adjacent lane)
            gap = forward - (v.getLength("ego") + v.getLength(vid)) / 2.
            yaw = math.radians((angle - ego_angle + 180) % 360 - 180)
            along = max(v.getSpeed(vid) * math.cos(yaw), 0.)  # its speed along the ego's heading
            desired = min(desired, self._vehicle_desired(gap, along))
        light = self.signal()
        if light["present"] and light["state"] in ("r", "R", "y", "Y"):
            desired = min(desired, stop_at(light["distance"]))
        control = self.control_ahead()
        if control["stop"] and control["node"] not in self.stop_completed:
            desired = min(desired, stop_at(control["distance"]))
        if self.junction_hazard:
            line = self._junction_hazard_distance()
            if line is not None:
                desired = min(desired, stop_at(line))
        if self.yield_approach_speed and not v.getRoadID("ego").startswith(":"):
            # Own design (reward only): approach a junction where the ego must yield slowly enough to stop for a late foe.
            links = v.getNextLinks("ego")
            if links and links[0][5] in ("m", "g"):
                d = self.net.getLane(v.getLaneID("ego")).getLength() - v.getLanePosition("ego")
                va = self.yield_approach_speed
                if getattr(self, "fog_yield_scale", False) and getattr(self, "visibility", None):
                    # Own design (reward only): in fog, reach the yield line at half the speed that can stop within the visibility.
                    va = min(va, .5 * math.sqrt(2. * self.IDM_B * max(self.visibility - self.LINE_MARGIN, 0.)))
                desired = min(desired, math.sqrt(va * va + 2. * self.IDM_B * max(d - self.LINE_MARGIN, 0.)))
        if getattr(self, "fog_speed_cap", False) and getattr(self, "visibility", None):
            # Own design (reward only): in fog, 50 % of the posted limit when visibility <= 100 m (Korean Road Traffic
            # Act enforcement rule art. 19(2)), and never faster than stopping within the visible distance (IDM b).
            if self.visibility <= 100.:
                desired = min(desired, 0.5 * self.net.getLane(v.getLaneID("ego")).getSpeed())
            desired = min(desired, math.sqrt(2. * self.IDM_B * max(self.visibility - self.LINE_MARGIN, 0.)))
        return desired

    def roach_desired_speed(self):
        if self.desired_speed in ("idm", "idm_krauss"):
            return self.idm_desired_speed()
        import math
        vmax = self.roach_max_speed
        k = self.roach_hazard_scale
        desired = vmax
        (ex, ey), (fx, fy), ego_angle = self._center_and_heading("ego")
        nearest = None
        for vid in self.v.getIDList():
            if vid == "ego":
                continue
            (x, y), _, angle = self._center_and_heading(vid)
            dx, dy = x - ex, y - ey
            dist = math.hypot(dx, dy)
            if dist > 9.5 * k:
                continue
            forward, left = dx * fx + dy * fy, -dx * fy + dy * fx
            yaw = (angle - ego_angle + 180) % 360 - 180
            ahead = dist < 1e-3 or abs(math.degrees(math.atan2(left, forward))) < 45
            if abs(yaw) <= 150 and ahead and (nearest is None or dist < nearest):
                nearest = dist
        if nearest is not None:
            desired = min(desired, vmax * min(max(nearest - 8. * k, 0.), 5. * k) / (5. * k))
        light = self.signal()
        if light["present"] and light["state"] in ("r", "R", "y", "Y") and light["distance"] <= 18. * k:
            desired = min(desired, vmax * min(max(light["distance"] - 5., 0.), 5. * k) / (5. * k))
        control = self.control_ahead()
        if control["stop"] and control["node"] not in self.stop_completed:
            desired = min(desired, vmax * min(max(control["distance"] - 5., 0.), 5. * k) / (5. * k))
        if self.junction_hazard:
            line = self._junction_hazard_distance()
            if line is not None and line <= 18. * k:
                desired = min(desired, vmax * min(max(line - 5., 0.), 5. * k) / (5. * k))
        return desired

    def _junction_hazard_distance(self):
        if self.junction_hazard >= 2:
            return self._crossing_hazard_distance()
        return self._entry_hazard_distance()

    @staticmethod
    def _time_to_travel(d, v, a, vmax):
        if d <= 0:
            return 0.
        t1 = max(vmax - v, 0.) / a
        d1 = v * t1 + .5 * a * t1 * t1
        if d <= d1:
            return (-v + (v * v + 2 * a * d) ** .5) / a
        return t1 + (d - d1) / vmax

    def _crossing_hazard_distance(self, margin=1.5):
        v = self.v
        if "ego" not in v.getIDList():
            return None
        road = v.getRoadID("ego")
        inside_ego = road.startswith(":")
        line = None if inside_ego else self.net.getLane(v.getLaneID("ego")).getLength() - v.getLanePosition("ego")
        speed, length = v.getSpeed("ego"), v.getLength("ego")
        best = None
        for vid, _, f_speed, inside, must_yield, _, d_e, d_f in self._conflicts():
            if d_e is None or not (must_yield or inside):
                continue
            if d_e < 3. or d_f < -v.getLength(vid):
                continue  # ego committed / already past, or foe already past the crossing
            occupying = d_f <= 2.
            if not occupying and f_speed < .5:
                continue  # waiting foe
            t_f = 0. if occupying else d_f / f_speed
            t_clear = self._time_to_travel(d_e + length + 2., speed, 2., self.roach_max_speed)
            if t_f < t_clear + margin:
                stop = line if not inside_ego else d_e - 3.
                best = stop if best is None else min(best, stop)
        return best

    def _entry_hazard_distance(self):
        """Distance to the junction entry if a conflicting vehicle makes entering unsafe, else None."""
        road = self.v.getRoadID("ego")
        if road.startswith(":") or "ego" not in self.v.getIDList():
            return None
        line = self.net.getLane(self.v.getLaneID("ego")).getLength() - self.v.getLanePosition("ego")
        if line > CONFLICT_RANGE:
            return None
        allway = self.net.getEdge(road).getToNode().getType() == "allway_stop"
        for _, dist, speed, inside, must_yield, to_node, _, _ in self._conflicts():
            if inside and speed > 1. and dist < 30.:
                return line
            if not inside and not allway and must_yield and speed > 1. and to_node / speed < 3.:
                return line
        return None

    def roach_speed_reward(self, info):
        if info["actual_speed"] is None or "ego" not in self.v.getIDList():
            return 0.
        desired = self.roach_desired_speed()
        info["roach_desired_speed"] = desired
        idle = self.idle_penalty if (self.idle_penalty and info["actual_speed"] < .1 and desired >= 2.) else 0.
        if idle:
            info["idle_penalty"] = idle
        if self.speed_reward == "relative":
            floor = self.stop_reward_floor if (getattr(self, "stop_reward_floor", 0.) and desired < .1) else 2.
            return max(-1., 1. - abs(info["actual_speed"] - desired) / max(desired, floor)) + idle
        return 1. - abs(info["actual_speed"] - desired) / self.roach_max_speed

    # ---- coarse routing observation ----
    def observation(self):
        if not getattr(self, "visibility", None) or "ego" not in self.v.getIDList():
            return self._observation_true()
        real = self.v
        ex, ey = real.getPosition("ego")
        r2 = self.visibility ** 2
        visible = {"ego"}
        for vid in real.getIDList():
            x, y = real.getPosition(vid)
            if (x - ex) ** 2 + (y - ey) ** 2 <= r2:
                visible.add(vid)
        self.v, self._fog_obs = _FogView(real, visible), True
        try:
            return self._observation_true()
        finally:
            self.v, self._fog_obs = real, False

    def _observation_true(self):
        dim, self.obs_dim = self.obs_dim, self._base_dim
        try:
            out = super().observation()
        finally:
            self.obs_dim = dim
        if self.route_obs == "coarse":
            out[0, self.route_idx] = self.coarse_route_features()
        if self.signal_obs == "color":
            out[0, self.time_idx] = 0.
        if self.extra_obs:
            extra = self.extra_features() + (self.foe_features() if self.extra_obs >= 2 else [])
            if self.extra_obs >= 3:
                extra += self._crossing_features
            out = np.concatenate([out, np.clip(np.asarray(extra, np.float32), -1, 1)[None, :]], axis=1)
        if self.history_lags:
            from collections import deque
            if self._history is None:
                self._history = deque([out] * (max(self.history_lags) + 1), maxlen=max(self.history_lags) + 1)
            else:
                self._history.append(out)
            out = np.concatenate([out] + [self._history[-1 - lag] for lag in self.history_lags], axis=1)
        if getattr(self, "_input_idx", None) is not None:
            out = out[:, self._input_idx]
        if getattr(self, "visibility_input", False):
            vis = min(self.visibility, 200.) / 200. if self.visibility else 1.
            out = np.concatenate([out, np.full((out.shape[0], 1), vis, np.float32)], axis=1)
        return out

    def coarse_route_features(self):
        lane_id = self.v.getLaneID("ego")
        if lane_id.startswith(":"):
            return self._last_route
        cur = self.net.getLane(lane_id)
        route = self.v.getRoute("ego")
        k = self.v.getRouteIndex("ego")
        dist = cur.getLength() - self.v.getLanePosition("ego")  # to the end of the current edge
        usable, direction, fork = None, 0., None
        for j in range(k, len(route) - 1):
            if dist > self.route_lookahead:
                break
            nxt = route[j + 1]
            edge = self.net.getEdge(route[j])
            connecting = [l for l in edge.getLanes() if any(c.getTo().getID() == nxt for c in l.getOutgoing())]
            if fork is None and len(connecting) < edge.getLaneNumber():
                fork = dist
            onward = [c.getToLane() for c in cur.getOutgoing() if c.getTo().getID() == nxt]
            if not onward:
                usable = dist
                if connecting:
                    target = min(connecting, key=lambda l: abs(l.getIndex() - cur.getIndex()))
                    direction = float(np.sign(target.getIndex() - cur.getIndex()))
                break
            cur = min(onward, key=lambda l: abs(l.getIndex() - cur.getIndex()))
            dist += cur.getLength()
        L = self.route_lookahead
        cap = lambda d: min(L if d is None else d, L) / L
        self._last_route = [cap(usable), direction, cap(fork)]
        return self._last_route

    # ---- extra perception ----
    @staticmethod
    def _links_conflict(node, a, b):
        try:
            return node.areFoes(a, b) or node.areFoes(b, a)
        except (KeyError, IndexError):  # junction without a stored foe row for these links
            return False

    @staticmethod
    def _must_yield(node, conn, ego_conn):
        try:
            return float(node.forbids(conn, ego_conn))
        except (KeyError, IndexError):
            return 0.

    def _conflicts(self):
        """Vehicles whose junction connection conflicts with the ego's next one (SUMO junction foe matrix):
        (id, euclidean distance, speed, inside junction, ego must yield, distance to the junction)."""
        v = self.v
        road = v.getRoadID("ego")
        ego_conn = self._connection("ego")
        if ego_conn is None:
            return []
        if road.startswith(":"):
            node_id = self._node_of_internal(road)
        else:
            node_id = self.net.getEdge(road).getToNode().getID()
            if self.net.getLane(v.getLaneID("ego")).getLength() - v.getLanePosition("ego") > CONFLICT_RANGE:
                return []
        node = self._foe_net().getNode(node_id)
        ego_idx = node.getLinkIndex(ego_conn)
        if ego_idx < 0:
            return []
        ex, ey = v.getPosition("ego")
        signal = self.signal_aware and self.net.getNode(node_id).getType().startswith("traffic_light")
        ego_state = None
        if signal:
            if road.startswith(":"):
                ego_state = getattr(self, "_ego_signal_state", None)
            else:
                links = v.getNextLinks("ego")
                ego_state = links[0][5] if links else None
                self._ego_signal_state = ego_state
        found = []
        for vid in v.getIDList():
            if vid == "ego":
                continue
            r = v.getRoadID(vid)
            inside = r.startswith(":")
            if inside:
                if self._node_of_internal(r) != node_id:
                    continue
                to_node = 0.
            else:
                if r == road or self.net.getEdge(r).getToNode().getID() != node_id:
                    continue
                to_node = self.net.getLane(v.getLaneID(vid)).getLength() - v.getLanePosition(vid)
                if to_node > CONFLICT_RANGE:
                    continue
            conn = self._connection(vid)
            idx = node.getLinkIndex(conn) if conn is not None else -1
            if idx < 0 or not self._links_conflict(node, idx, ego_idx):
                continue
            must_yield = self._must_yield(node, conn, ego_conn)
            if self.signal_aware:
                state = None
                if not inside:
                    links = v.getNextLinks(vid)
                    state = links[0][5] if links else None
                    if state in ("r", "R", "s", "w") and (to_node > 3. or v.getSpeed(vid) < .5):
                        continue  # it has to stop first
                if signal:
                    mine = ego_state or ""
                    if state is not None and mine:
                        must_yield = float(state == "G" and mine == "g") if state != mine else must_yield
            x, y = v.getPosition(vid)
            d_e = d_f = None
            cross = self._crossing(ego_conn, conn)
            if cross is not None:
                d_e = cross[0] - self._path_position("ego", ego_conn)
                d_f = cross[1] - self._path_position(vid, conn)
            found.append((vid, ((x - ex) ** 2 + (y - ey) ** 2) ** .5, v.getSpeed(vid), inside,
                          must_yield, to_node, d_e, d_f))
        found.sort(key=lambda f: f[1])
        return found

    def _path(self, conn):
        """Centre line of a connection's internal lanes: (points, cumulative lengths, {lane id: start offset})."""
        cache = self.__dict__.setdefault("_path_cache", {})
        key = (conn.getFromLane().getID(), conn.getToLane().getID())
        if key not in cache:
            net = self._foe_net()
            points, offsets, total = [], {}, 0.
            lane_id = conn.getViaLaneID()
            while lane_id and lane_id.startswith(":") and lane_id not in offsets:
                lane = net.getLane(lane_id)
                offsets[lane_id] = total
                shape = lane.getShape()
                points += shape if not points else shape[1:]
                total += lane.getLength()
                lane_id = next((c.getViaLaneID() for c in lane.getOutgoing() if c.getViaLaneID()), None)
            cum = [0.]
            for a, b in zip(points, points[1:]):
                cum.append(cum[-1] + ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** .5)
            cache[key] = (points, cum, offsets)
        return cache[key]

    def _path_position(self, vid, conn):
        _, _, offsets = self._path(conn)
        lane_id = self.v.getLaneID(vid)
        if lane_id in offsets:
            return offsets[lane_id] + self.v.getLanePosition(vid)
        if lane_id.startswith(":"):
            return 0.
        return -(self.net.getLane(lane_id).getLength() - self.v.getLanePosition(vid))

    def _crossing(self, conn_a, conn_b):
        """Arc lengths (along a, along b) of the first point where the two internal paths cross, or None."""
        cache = self.__dict__.setdefault("_cross_cache", {})
        key = (conn_a.getFromLane().getID(), conn_a.getToLane().getID(), conn_b.getFromLane().getID(), conn_b.getToLane().getID())
        if key in cache:
            return cache[key]
        pa, ca, _ = self._path(conn_a)
        pb, cb, _ = self._path(conn_b)
        result = None
        if conn_a.getToLane().getID() == conn_b.getToLane().getID() and pa and pb:
            result = (ca[-1], cb[-1])  # merging into the same lane
        for i in range(len(pa) - 1):
            if result is not None and ca[i] >= result[0]:
                break
            (x1, y1), (x2, y2) = pa[i], pa[i + 1]
            for j in range(len(pb) - 1):
                (x3, y3), (x4, y4) = pb[j], pb[j + 1]
                den = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
                if abs(den) < 1e-9:
                    continue
                s = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / den
                u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / den
                if 0 <= s <= 1 and 0 <= u <= 1:
                    sa = ca[i] + s * (ca[i + 1] - ca[i])
                    if result is None or sa < result[0]:
                        result = (sa, cb[j] + u * (cb[j + 1] - cb[j]))
        cache[key] = result
        return result

    def foe_features(self):
        found = self._conflicts()
        self._foe_ids = [f[0] for f in found[:FOE_SLOTS]]  # for diagnostics only
        out = []
        for k in range(FOE_SLOTS):
            if k < len(found):
                _, dist, speed, inside, must_yield, _, _, _ = found[k]
                out += [dist / CONFLICT_RANGE, speed / self.s.speed, float(inside), must_yield]
            else:
                out += [1., 0., 0., 0.]
        cross = []
        for k in range(FOE_SLOTS):
            if k < len(found) and found[k][6] is not None:
                d_e, d_f, f_speed = found[k][6], found[k][7], found[k][2]
                t_f = 0. if d_f <= 0 else min(d_f / max(f_speed, .1), TTA_SCALE)
                cross += [d_e / CONFLICT_RANGE, t_f / TTA_SCALE]
            else:
                cross += [1., 1.]
        self._crossing_features = cross
        return out

    @staticmethod
    def _node_of_internal(road):
        return road[1:road.rfind("_")]

    def extra_features(self):
        v, s = self.v, self.s
        road = v.getRoadID("ego")
        ego_angle = v.getAngle("ego")
        if road.startswith(":"):
            node, ego_dist = self._node_of_internal(road), 0.
        else:
            node = self.net.getEdge(road).getToNode().getID()
            ego_dist = self.net.getLane(v.getLaneID("ego")).getLength() - v.getLanePosition("ego")
        links = v.getNextLinks("ego")
        ego_priority = 1. if road.startswith(":") or not links else float(links[0][2])
        candidates = []
        if ego_dist <= CONFLICT_RANGE:
            for vid in v.getIDList():
                if vid == "ego":
                    continue
                r = v.getRoadID(vid)
                if r.startswith(":"):
                    if self._node_of_internal(r) != node:
                        continue
                    dist, prio = 0., 1.
                else:
                    if r == road or self.net.getEdge(r).getToNode().getID() != node:
                        continue
                    dist = self.net.getLane(v.getLaneID(vid)).getLength() - v.getLanePosition(vid)
                    if dist > CONFLICT_RANGE:
                        continue
                    other = v.getNextLinks(vid)
                    prio = float(other[0][2]) if other else 1.
                speed = v.getSpeed(vid)
                tta = min(dist / max(speed, .1), TTA_SCALE)
                direction = ((v.getAngle(vid) - ego_angle + 180) % 360 - 180) / 180
                candidates.append((tta, [dist / CONFLICT_RANGE, speed / s.speed, tta / TTA_SCALE, prio, direction]))
        candidates.sort(key=lambda c: c[0])
        out = []
        for k in range(CONFLICT_SLOTS):
            out += candidates[k][1] if k < len(candidates) else [1., 0., 1., 0., 0.]
        out.append(ego_priority)
        # other-side adjacent lane (the original inputs cover only one side)
        lane_index = v.getLaneIndex("ego")
        other_lane = None
        if not road.startswith(":"):
            lanes = self.net.getEdge(road).getLaneNumber()
            if lane_index + 1 < lanes and lane_index - 1 >= 0:
                other_lane = lane_index - 1
        same_edge = [(vid, v.getLanePosition(vid), v.getLaneIndex(vid)) for vid in v.getIDList()
                     if vid != "ego" and not road.startswith(":") and v.getRoadID(vid) == road]
        at = v.getLanePosition("ego") if not road.startswith(":") else 0.
        def nearest(lane, ahead, limit):
            cands = [((p - at) if ahead else (at - p), vid) for vid, p, l in same_edge if l == lane]
            cands = [(d, vid) for d, vid in cands if 0 <= d <= limit]
            return min(cands) if cands else None
        for ahead, limit, sign in ((True, s.front_range, 1), (False, s.rear_range, -1)):
            hit = nearest(other_lane, ahead, limit) if other_lane is not None else None
            if hit:
                d, vid = hit
                gap = max(0., d - (v.getLength(vid) if ahead else s.length))
                out += [sign * gap / limit, v.getSpeed(vid) / s.speed, v.getAcceleration(vid) / s.brake,
                        float(v.getLaneIndex(vid) - lane_index)]
            else:
                out += [float(sign), 0., 0., 0.]
        # followers within 40 m on own / left (+1) / right (-1) lanes
        for lane in (lane_index, lane_index + 1, lane_index - 1):
            hit = nearest(lane, False, REAR_RANGE) if same_edge else None
            if hit:
                d, vid = hit
                out += [max(0., d - s.length) / REAR_RANGE, v.getSpeed(vid) / s.speed]
            else:
                out += [1., 0.]
        return out
