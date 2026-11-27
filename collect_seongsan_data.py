"""Seongsan driving-data collector: POMDP transitions of the trained PPO ego, with the same columns as the
course's collect_pomdp_data.py (6_Project_Evaluation.md, "기본 데이터 수집 항목") plus a few Seongsan columns.

One transition per 0.2 s step:  (o_t, a_t, r_t, o_{t+1}, terminated, truncated, s_t, s_{t+1})

How it runs (GPU policy, CPU simulation):
- `--workers` processes each run one SUMO (libsumo, one per process) and only simulate + record.
- The main process gathers the observations of whichever workers are ready and runs ONE batched
  forward pass of the policy on `--device` (default cuda), then sends each worker its action.
- Each worker writes its own shards every `--shard-episodes` episodes, so memory stays bounded
  and the output can be uploaded to Hugging Face as many files:
      data/<name>/jsonl/w{worker}_{k}.jsonl.gz   full record (privileged state, detected vehicles, events)
      data/<name>/npz/w{worker}_{k}.npz          fixed-size arrays for training (course NPZ keys)
      data/<name>/meta.json                      checkpoint, reward/env settings, observation names, columns

Reward = the reward the policy was trained with (settings read from the run's config.json), as the course asks
("데이터 수집 코드도 train.py의 REWARD를 사용"). Episode termination = evaluation rules (collision, red light,
STOP violation, route failure, arrival, time limit); the training-only yield-failure / blocked-time terminations are off.

Columns that differ from the course env because the Seongsan policy is different:
- observation: the policy's 63 inputs (names in meta.json), not the course's 19.
- action_raw = [accel_raw, lane_change_raw] decoded from the 9 discrete actions (index = 3*lon + lat):
  accel_raw in {-1 brake, 0 coast, +1 accelerate}, lane_change_raw in {-1 right, 0 keep, +1 left}.
  The index itself is stored as action.index, the commanded acceleration (m/s^2) as action.accel_mps2.
- state.vehicles: vehicles within `--state-radius` m of the ego (the course env stores every vehicle of its
  short highway; on the Seongsan network that would be hundreds of vehicles per step).
- detected_vehicles: vehicles within `--detect-radius` m (= the policy's 50 m front sensing range), or within the fog
  visibility when it is shorter.
- visibility (extra column): fog visibility in m, 0 = clear. In fog the observation is degraded while state is the
  true simulator state, which is exactly the POMDP gap the course format separates.

Examples:
    python collect_seongsan_data.py --checkpoint results/W28_20261125/model.pt --episodes 20 --name smoke
    python collect_seongsan_data.py --checkpoint <ck> --transitions 10000000 --workers 10 --name seongsan_final

Reading:
    from collect_seongsan_data import iter_transitions
    for rec in iter_transitions("data/seongsan_final/jsonl/w0_0.jsonl.gz"): ...
"""
import argparse
import glob
import gzip
import json
import math
import multiprocessing as mp
import os
import time
from multiprocessing.connection import wait

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- helpers
def compact(obj, ndigits):
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), ndigits)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return compact(obj.tolist(), ndigits)
    if isinstance(obj, dict):
        return {k: compact(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [compact(v, ndigits) for v in obj]
    return obj


def iter_transitions(jsonl_path):
    """Yield complete transitions, filling the omitted next_observation / next_state from the next record."""
    opener = gzip.open if jsonl_path.endswith(".gz") else open
    with opener(jsonl_path, "rt", encoding="utf-8") as f:
        prev = None
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if prev is not None:
                prev.setdefault("next_observation", rec["observation"])
                prev.setdefault("next_state", rec["state"])
                yield prev
            prev = rec
        if prev is not None:
            yield prev


def privileged_state(v, radius):
    """SUMO-side state (not a policy input): ego + vehicles within `radius` m."""
    if "ego" not in v.getIDList():
        return {"ego_alive": False}
    ex, ey = v.getPosition("ego")
    r2 = radius * radius
    vehicles = []
    for vid in v.getIDList():
        if vid == "ego":
            continue
        x, y = v.getPosition(vid)
        if (x - ex) ** 2 + (y - ey) ** 2 > r2:
            continue
        vehicles.append({"id": vid, "x": x, "y": y, "speed": v.getSpeed(vid), "angle": v.getAngle(vid),
                         "lane": v.getLaneIndex(vid), "lane_id": v.getLaneID(vid),
                         "road_id": v.getRoadID(vid), "lane_pos": v.getLanePosition(vid)})
    return {"ego_alive": True, "ego_x": ex, "ego_y": ey, "ego_speed": v.getSpeed("ego"),
            "ego_accel": v.getAcceleration("ego"), "ego_angle": v.getAngle("ego"),
            "ego_lane": v.getLaneIndex("ego"), "ego_lane_id": v.getLaneID("ego"),
            "ego_road_id": v.getRoadID("ego"), "ego_lane_pos": v.getLanePosition("ego"),
            "vehicles": vehicles}


def detected_vehicles(state, radius):
    """Vehicles within `radius` m of the ego, with ego-frame offsets (longitudinal +ahead, lateral +left)."""
    if not state.get("ego_alive"):
        return []
    ex, ey = state["ego_x"], state["ego_y"]
    a = math.radians(state["ego_angle"])  # SUMO: 0 = north, clockwise
    fx, fy = math.sin(a), math.cos(a)
    out = []
    for veh in state["vehicles"]:
        dx, dy = veh["x"] - ex, veh["y"] - ey
        d = math.hypot(dx, dy)
        if d > radius:
            continue
        out.append({"id": veh["id"], "x": veh["x"], "y": veh["y"], "distance": d,
                    "rel_pos": dx * fx + dy * fy, "lateral_offset": -dx * fy + dy * fx, "speed": veh["speed"]})
    return out


def env_kwargs(ck, cfg):
    """Observation settings from the checkpoint, reward settings from the training run's config.json."""
    from paper_training.rule_curriculum import history_lags_from_names
    names = ck["observation_names"]
    return dict(
        route_obs="coarse" if "route_lane_usable_distance" in names else "lane",
        signal_obs="color" if "time_to_green_unused" in names else "color_time",
        route_lookahead=float(ck.get("route_lookahead", cfg.get("route_lookahead_m", 50.))),
        extra_obs=3 if "foe_0_ego_to_crossing" in names else 2 if "foe_0_distance" in names else int("conflict_0_distance" in names),
        history_lags=history_lags_from_names(names), signal_aware=bool(ck.get("signal_aware", False)),
        stop_zone=ck.get("stop_zone"), input_names=names, signal_next_only=bool(ck.get("signal_next_only", False)),
        reward=cfg.get("reward", "dojo"), lane_change_penalty=cfg.get("lane_change_penalty", 0.),
        roach_max_speed=cfg.get("roach_max_speed", 13.8889), roach_hazard_scale=cfg.get("roach_hazard_scale", 1.),
        junction_hazard=cfg.get("junction_hazard", False), desired_speed=cfg.get("desired_speed", "roach"),
        speed_reward=cfg.get("speed_reward", "roach"),
        unsafe_lane_change_penalty=cfg.get("unsafe_lane_change_penalty", 0.), idle_penalty=cfg.get("idle_penalty", 0.),
        yield_approach_speed=cfg.get("yield_approach_speed", 0.) or 0.,
        fog_speed_cap=bool(cfg.get("fog_speed_cap", False)), fog_yield_scale=bool(cfg.get("fog_yield_scale", False)),
        stop_reward_floor=cfg.get("stop_reward_floor", 0.) or 0.,
        yield_failure=False, blocked_seconds=None)  # evaluation terminations


# ---------------------------------------------------------------- worker (CPU: SUMO + recording)
def worker_main(conn, w, a):
    import torch
    torch.set_num_threads(1)
    import sys
    sys.path.insert(0, ROOT)
    from paper_training.rule_curriculum import use_map, RuleCurriculumDriving
    from paper_training.profiles import Profile
    ck = torch.load(a["checkpoint"], map_location="cpu", weights_only=False)
    cfg = a["cfg"]
    use_map(cfg["map"]["name"] if isinstance(cfg.get("map"), dict) else a["map"])
    traffic, visibility = a["conditions"][w % len(a["conditions"])]
    env = RuleCurriculumDriving(os.path.join(ROOT, "results", "collect_tmp", f"{a['name']}_w{w}"), a["seed_base"] + w,
                                "full", traffic, visibility=visibility or None, **env_kwargs(ck, cfg))
    detect_radius = min(a["detect_radius"], visibility) if visibility else a["detect_radius"]
    env.numeric_correction |= ck.get("numeric_correction", False)
    env.p = Profile(**ck["profile"])
    nd, out = a["ndigits"], a["out"]
    shard, total, eps_in_shard = 0, 0, 0
    jf = None
    buf = {k: [] for k in ("observation", "next_observation", "action_raw", "action_index", "lane_change", "reward",
                           "terminated", "truncated", "episode", "t", "traffic", "visibility")}
    xy, offsets = [], [0]

    def open_shard():
        return gzip.open(os.path.join(out, "jsonl", f"w{w}_{shard}.jsonl.gz"), "wt", encoding="utf-8", compresslevel=6)

    def flush():
        nonlocal xy, offsets
        n = len(buf["reward"])
        if n:
            np.savez_compressed(
                os.path.join(out, "npz", f"w{w}_{shard}.npz"),
                observation=np.asarray(buf["observation"], np.float32).reshape(n, -1),
                action_raw=np.asarray(buf["action_raw"], np.float32).reshape(n, 2),
                action_index=np.asarray(buf["action_index"], np.int8),
                lane_change=np.asarray(buf["lane_change"], np.int8),
                reward=np.asarray(buf["reward"], np.float32),
                next_observation=np.asarray(buf["next_observation"], np.float32).reshape(n, -1),
                terminated=np.asarray(buf["terminated"], np.bool_), truncated=np.asarray(buf["truncated"], np.bool_),
                episode=np.asarray(buf["episode"], np.int32), t=np.asarray(buf["t"], np.int32),
                traffic=np.asarray(buf["traffic"], np.float32), visibility=np.asarray(buf["visibility"], np.float32),
                detected_xy=np.asarray(xy, np.float32).reshape(-1, 2), detected_offsets=np.asarray(offsets, np.int64))
        for k in buf:
            buf[k].clear()
        xy, offsets = [], [0]

    ep = t = None
    obs = state = None
    outcomes = {}
    while True:
        cmd, arg = conn.recv()
        if cmd == "reset":
            if jf is None:
                jf = open_shard()
            ep, t = arg, 0
            obs = env.reset(seed=a["seed_base"] + ep)
            state = privileged_state(env.v, a["state_radius"])
            conn.send(np.asarray(obs, np.float32).reshape(-1))
        elif cmd == "step":
            idx = int(arg)
            lon, lat = divmod(idx, 3)
            detected = detected_vehicles(state, detect_radius)
            lane_before = env.v.getLaneIndex("ego") if "ego" in env.v.getIDList() else None
            nobs, r, dn, tr, info = env.step(idx)
            reward = float(np.asarray(r, dtype=np.float64).ravel()[0])
            nstate = privileged_state(env.v, a["state_radius"])
            done = bool(dn or tr or t + 1 >= env.maximum_steps)
            truncated = bool((tr or t + 1 >= env.maximum_steps) and not dn)
            o = np.asarray(obs, np.float32).reshape(-1)
            no = np.asarray(nobs, np.float32).reshape(-1)
            issued = bool(info.get("lane_change_issued", False))
            lc_raw = (-1, 0, 1)[lat]
            rec = {"episode": ep, "t": t, "traffic": traffic, "visibility": visibility, "observation": compact(o, nd),
                   "action_raw": [float((-1, 0, 1)[lon]), float(lc_raw)],
                   "action": {"index": idx, "accel_raw": float((-1, 0, 1)[lon]), "accel_mps2": float(info.get("requested_accel", 0.)),
                              "lane_change_raw": float(lc_raw), "lane_change": lc_raw if issued else 0,
                              "lane_change_applied": bool(info.get("lane_change_applied", False))},
                   "reward": reward, "terminated": bool(dn), "truncated": truncated,
                   "state": compact(state, nd), "detected_vehicles": compact(detected, nd),
                   "event": {"collided": bool(info.get("collision", False)), "arrived": info.get("outcome") == "complete",
                             "lane_before": lane_before,
                             "lane_after": env.v.getLaneIndex("ego") if "ego" in env.v.getIDList() else None,
                             "outcome": info.get("outcome")}}
            if done:
                rec["next_observation"] = compact(no, nd)
                rec["next_state"] = compact(nstate, nd)
            jf.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            buf["observation"].append(o); buf["next_observation"].append(no)
            buf["action_raw"].append(rec["action_raw"]); buf["action_index"].append(idx)
            buf["lane_change"].append(rec["action"]["lane_change"]); buf["reward"].append(reward)
            buf["terminated"].append(bool(dn)); buf["truncated"].append(truncated)
            buf["episode"].append(ep); buf["t"].append(t); buf["traffic"].append(traffic); buf["visibility"].append(visibility)
            xy.extend((d["x"], d["y"]) for d in detected); offsets.append(len(xy))
            total += 1
            t += 1
            obs, state = nobs, nstate
            if done:
                outcomes[info.get("outcome")] = outcomes.get(info.get("outcome"), 0) + 1
                eps_in_shard += 1
                if eps_in_shard >= a["shard_episodes"]:
                    jf.close(); jf = None
                    flush(); shard += 1; eps_in_shard = 0
            conn.send((np.asarray(nobs, np.float32).reshape(-1), done))
        elif cmd == "close":
            if jf is not None:
                jf.close()
                flush()
            env.close()
            conn.send({"worker": w, "traffic": traffic, "visibility": visibility, "transitions": total, "outcomes": outcomes})
            return


# ---------------------------------------------------------------- main (GPU: batched policy)
def main():
    p = argparse.ArgumentParser(description="Seongsan POMDP data collector (GPU policy, parallel SUMO)")
    p.add_argument("--checkpoint", default=os.path.join(ROOT, "models", "final_ppo", "model.pt"))
    p.add_argument("--config", help="training config.json (default: config.json or training_config.json next to the checkpoint)")
    p.add_argument("--episodes", type=int, default=0, help="stop after this many episodes (0 = use --transitions)")
    p.add_argument("--transitions", type=int, default=0, help="stop starting new episodes once this many transitions exist")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--traffic", default="1,1.5,2", help="traffic multipliers")
    p.add_argument("--visibility", default="0", help="fog visibilities in m (0 = clear); every traffic x visibility pair is one worker condition, "
                   "assigned to workers round-robin (use --workers = number of pairs to cover all)")
    p.add_argument("--seed-base", type=int, default=3_000_000)
    p.add_argument("--state-radius", type=float, default=150.)
    p.add_argument("--detect-radius", type=float, default=50.)
    p.add_argument("--shard-episodes", type=int, default=200)
    p.add_argument("--ndigits", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    p.add_argument("--map", default="pluslane_v4")
    p.add_argument("--out-dir", default=os.path.join(ROOT, "data"))
    p.add_argument("--name", required=True)
    args = p.parse_args()
    if not args.episodes and not args.transitions:
        p.error("give --episodes or --transitions")

    import sys
    import torch
    sys.path.insert(0, ROOT)
    from types import SimpleNamespace
    from paper_training.agents import OnPolicy
    from paper_training.profiles import Profile

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_path = args.config or next((c for c in (os.path.join(os.path.dirname(args.checkpoint), n) for n in ("config.json", "training_config.json"))
                                    if os.path.exists(c)), "")
    cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else {}
    names = ck["observation_names"]
    n_out = ck["model"]["actor.4.weight"].shape[0]
    stub = SimpleNamespace(discrete=n_out, action_dim=2, obs_dim=len(names))
    agent = OnPolicy(Profile(**ck["profile"]), stub, device)
    agent.model.load_state_dict(ck["model"])
    agent.model.eval()

    out = os.path.join(args.out_dir, args.name)
    os.makedirs(os.path.join(out, "jsonl"), exist_ok=True)
    os.makedirs(os.path.join(out, "npz"), exist_ok=True)
    traffic = [float(x) for x in args.traffic.split(",")]
    conditions = [(tr, float(vi)) for vi in args.visibility.split(",") for tr in traffic]
    wa = {"checkpoint": os.path.abspath(args.checkpoint), "cfg": cfg, "map": args.map, "conditions": conditions,
          "seed_base": args.seed_base, "state_radius": args.state_radius, "detect_radius": args.detect_radius,
          "shard_episodes": args.shard_episodes, "ndigits": args.ndigits, "out": out, "name": args.name}
    json.dump({"checkpoint": args.checkpoint, "config": cfg_path, "observation_names": names,
               "worker_conditions_traffic_visibility": [conditions[w % len(conditions)] for w in range(args.workers)],
               "env_settings": {k: v for k, v in env_kwargs(ck, cfg).items() if k != "input_names"},
               "dt_seconds": 0.2, "state_radius_m": args.state_radius, "detect_radius_m": args.detect_radius,
               "action_index": "index = 3*lon + lat; lon 0/1/2 = brake/coast/accelerate, lat 0/1/2 = right/keep/left",
               "seed_base": args.seed_base, "policy": "stochastic" if args.stochastic else "deterministic argmax",
               "columns_course": ["episode", "t", "observation", "action_raw", "action.accel_raw", "action.lane_change_raw",
                                  "action.lane_change", "action.lane_change_applied", "reward", "terminated", "truncated",
                                  "state", "detected_vehicles", "event.collided", "event.arrived", "event.lane_before",
                                  "event.lane_after", "next_observation", "next_state"],
               "columns_extra": ["traffic", "visibility", "action.index", "action.accel_mps2", "event.outcome"],
               "fog": "visibility > 0: the observation (policy input) omits vehicles and signal colours farther than `visibility` m; "
                      "state / next_state stay the true simulator state; detected_vehicles uses min(detect radius, visibility)."},
              open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    ctx = mp.get_context("spawn")
    conns, procs = [], []
    for w in range(args.workers):
        parent, child = ctx.Pipe()
        pr = ctx.Process(target=worker_main, args=(child, w, wa), daemon=True)
        pr.start()
        conns.append(parent); procs.append(pr)

    t0 = time.time()
    next_ep, started, finished, transitions = 0, 0, 0, 0
    pending = {}   # conn -> latest observation awaiting an action

    def want_more():
        if args.episodes:
            return started < args.episodes
        return transitions < args.transitions

    waiting, closing = set(), []  # workers whose reply we are waiting for / told to close
    for c in conns:
        if want_more():
            c.send(("reset", next_ep)); next_ep += 1; started += 1
            waiting.add(c)
        else:
            c.send(("close", None)); closing.append(c)
    last_print = t0
    while waiting:
        ready = wait(list(waiting))
        batch = []
        for c in ready:
            msg = c.recv()
            waiting.discard(c)
            if isinstance(msg, tuple):
                obs, done = msg
                transitions += 1
                if done:
                    finished += 1
                    if want_more():
                        c.send(("reset", next_ep)); next_ep += 1; started += 1
                        waiting.add(c)
                    else:
                        c.send(("close", None)); closing.append(c)
                    continue
                batch.append((c, obs))
            elif isinstance(msg, np.ndarray):
                batch.append((c, msg))
        if batch:
            x = np.stack([o for _, o in batch])
            with torch.no_grad():
                acts, _ = agent.action(x, deterministic=not args.stochastic)
            for (c, _), act in zip(batch, np.asarray(acts).reshape(-1)):
                c.send(("step", int(act))); waiting.add(c)
        now = time.time()
        if now - last_print > 30:
            last_print = now
            print(json.dumps({"elapsed_s": round(now - t0), "episodes_done": finished, "transitions": transitions,
                              "steps_per_s": round(transitions / (now - t0))}), flush=True)
    stats = [c.recv() for c in closing]
    for pr in procs:
        pr.join(timeout=60)
    summary = {"transitions": transitions, "episodes": finished, "seconds": round(time.time() - t0, 1),
               "device": device, "workers": stats}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
