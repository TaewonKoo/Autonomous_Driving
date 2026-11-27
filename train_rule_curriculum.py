"""Continue a Seongsan PPO checkpoint with a traffic-rule task mixture (see paper_training/rule_curriculum.py).

Only the training-task distribution changes. Observations, actions, reward coefficients and the
checkpoint's PPO settings are reused unchanged.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

import train_observed_driving as base
from paper_training.agents import Policy
from paper_training.observed_driving import SETTINGS
from paper_training.profiles import Profile
from paper_training.rule_curriculum import MAPS, RuleCurriculumDriving, use_map


ROOT = Path(__file__).resolve().parent
DEFAULT_TASKS = ("full", "full", "full", "full", "full", "signal", "signal", "stop")


def safe_rollout(arguments):
    """Worker exceptions from libsumo are not picklable; re-raise them as plain text."""
    try:
        return base.sample_rollout(arguments)
    except Exception:
        import traceback
        raise RuntimeError(traceback.format_exc()) from None


def initialize_worker(directory, seed, tasks, traffic, profile_data, map_name, signal_seconds, reward, lane_change_penalty, roach_max_speed, route_obs, signal_obs, route_lookahead, random_start_lane, worker_traffic=None, extra_obs=False, yield_failure=False, roach_hazard_scale=1.0, stop_zone=None, blocked_seconds=None, junction_hazard=False, desired_speed="roach", speed_reward="roach", history_lags=(), signal_aware=False, unsafe_lane_change_penalty=0., idle_penalty=0., input_names=None, signal_next_only=False, yield_approach_speed=0., worker_visibility=None, fog_speed_cap=False, fog_yield_scale=False, stop_reward_floor=0., collision_penalty=0.):
    import atexit
    use_map(map_name)
    torch.set_num_threads(1)
    process = multiprocessing.current_process()._identity[-1]
    np.random.seed(seed + process)
    torch.manual_seed(seed + process)
    task = tasks[(process - 1) % len(tasks)]
    if worker_traffic:
        traffic = worker_traffic[(process - 1) % len(worker_traffic)]
    visibility = worker_visibility[(process - 1) % len(worker_visibility)] if worker_visibility else 0.
    env = RuleCurriculumDriving(Path(directory) / f"worker_{process}_{task}", seed + process, task,
                                0. if task.split("@")[0] == "signal" else traffic,
                                reuse_traffic=task.split("@")[0] in ("junction", "stop", "full", "speed"),
                                signal_seconds=signal_seconds, reward=reward,
                                lane_change_penalty=lane_change_penalty, roach_max_speed=roach_max_speed,
                                route_obs=route_obs, signal_obs=signal_obs,
                                route_lookahead=route_lookahead, random_start_lane=random_start_lane,
                                extra_obs=extra_obs, yield_failure=yield_failure,
                                roach_hazard_scale=roach_hazard_scale, stop_zone=stop_zone,
                                blocked_seconds=blocked_seconds, junction_hazard=junction_hazard,
                                desired_speed=desired_speed, speed_reward=speed_reward,
                                history_lags=history_lags, signal_aware=signal_aware,
                                unsafe_lane_change_penalty=unsafe_lane_change_penalty, idle_penalty=idle_penalty,
                                input_names=input_names, signal_next_only=signal_next_only,
                                yield_approach_speed=yield_approach_speed,
                                visibility=visibility or None, fog_speed_cap=fog_speed_cap, fog_yield_scale=fog_yield_scale, stop_reward_floor=stop_reward_floor, collision_penalty=collision_penalty)
    if task.split("@")[0] == "stop":
        env.curriculum_distance = SETTINGS.length
    profile = Profile(**profile_data)
    env.p = profile
    base.WORKER_ENV = env
    base.WORKER_POLICY = Policy(profile, env).eval()
    base.WORKER_STATE = env.reset()
    atexit.register(env.close)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", default="models/seongsan_ppo_best_search.pt")
    p.add_argument("--output", default="results/rule_curriculum_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--steps", type=int, default=1_000_000, help="additional transitions")
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    p.add_argument("--traffic", type=float, default=1.)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--device", default="cpu")
    p.add_argument("--map", choices=tuple(MAPS), default="pluslane_v4")
    p.add_argument("--signal-seconds", type=float, help="signal-task episode length (default: base 20 s)")
    p.add_argument("--reward", choices=("base", "dojo", "dojo_roach"), default="base")
    p.add_argument("--route-obs", choices=("lane", "coarse"), default="lane",
                   help="coarse: TA-allowed lookahead routing features replace the per-lane route flags")
    p.add_argument("--lr", type=float, help="override learning rate")
    p.add_argument("--anneal", action="store_true", help="linearly anneal the learning rate to 0")
    p.add_argument("--widen", help="e.g. 256,256: distil the resumed policy into a wider network first")
    p.add_argument("--widen-activation", default="relu")
    p.add_argument("--widen-episodes", type=int, default=30)
    p.add_argument("--widen-epochs", type=int, default=20)
    p.add_argument("--save-every", type=int, help="also keep model_<trained steps>.pt every N updates")
    p.add_argument("--blocked-seconds", type=float, help="Roach Blocked: stopped this long -> failure (Roach: 90)")
    p.add_argument("--stop-zone", type=float, help="full stop within this many metres of a STOP line completes it")
    p.add_argument("--roach-hazard-scale", type=float, default=1.0, help="scale Roach hazard distances (use v_max/6)")
    p.add_argument("--yield-failure", action="store_true", help="forcing a priority vehicle to brake >4.5 m/s^2 is a failure")
    p.add_argument("--worker-visibility", help="comma list per task worker: fog visibility in m (0 = clear)")
    p.add_argument("--fog-speed-cap", action="store_true", help="reward: fog speed (50%% of limit at <=100 m, stop within visibility)")
    p.add_argument("--fog-yield-scale", action="store_true", help="reward: in fog, yield-junction approach speed <= half the stop-within-visibility speed")
    p.add_argument("--collision-penalty", type=float, default=0., help="reward: collision value instead of -10 (0 = off)")
    p.add_argument("--stop-reward-floor", type=float, default=0., help="reward: speed-error divisor when desired ~0 (default off = 2 m/s)")
    p.add_argument("--input-names", help="JSON list: give the policy only these inputs (pruned checkpoint)")
    p.add_argument("--idle-penalty", type=float, default=0., help="extra reward per step when standing although the desired speed is >= 2 m/s, e.g. -1")
    p.add_argument("--unsafe-lane-change-penalty", type=float, default=0., help="extra reward when an applied lane change violates the MOBIL safety criterion (b_safe 4 m/s^2), e.g. -6")
    p.add_argument("--yield-approach-speed", type=float, default=0., help="reward: approach speed (m/s) at junctions where the ego must yield (0 = off)")
    p.add_argument("--signal-next-only", action="store_true", help="show a traffic signal only when no all-way STOP comes before it")
    p.add_argument("--signal-aware", action="store_true", help="conflict inputs/hazard follow the current signal and STOP obligations")
    p.add_argument("--history-lags", default="", help="append the inputs of these past steps, e.g. 2,4 (frame stacking)")
    p.add_argument("--horizon", type=int, help="PPO transitions per update (all workers), overrides the checkpoint profile")
    p.add_argument("--batch", type=int, help="PPO minibatch size, overrides the checkpoint profile")
    p.add_argument("--speed-reward", choices=("roach", "relative"), default="roach", help="speed-matching error normalisation")
    p.add_argument("--desired-speed", choices=("roach", "idm", "idm_krauss"), default="roach", help="desired-speed profile of the speed reward")
    p.add_argument("--junction-hazard", nargs="?", type=int, const=1, default=0,
                   help="desired speed yields to conflicting junction traffic (own design): 1 before entry, 2 at the path crossing, also inside")
    p.add_argument("--extra-obs", nargs="?", type=int, const=1, default=0, help="1: append conflict-vehicle / both-side / rear-40 m inputs; 2: also junction-foe inputs")
    p.add_argument("--worker-traffic", help="comma list of traffic multipliers, one per task worker (overrides --traffic)")
    p.add_argument("--route-lookahead", type=float, default=50., help="coarse routing look-ahead (m)")
    p.add_argument("--random-start-lane", action="store_true", help="junction/stop practice starts in any lane")
    p.add_argument("--signal-obs", choices=("color_time", "color"), default="color_time",
                   help="color: drop the time-to-green input (light colour only)")
    p.add_argument("--roach-max-speed", type=float, help="Roach cruise target in m/s (author code: 6.0)")
    p.add_argument("--lane-change-penalty", type=float, default=0., help="per executed lane change (dojo reward only)")
    p.add_argument("--reset-critic", action="store_true", help="re-initialize the value network after loading")
    args = p.parse_args()
    use_map(args.map)
    if args.lane_change_penalty and args.reward == "base":
        p.error("--lane-change-penalty applies to the dojo reward")
    tasks = tuple(args.tasks.split(","))
    worker_traffic = tuple(float(x) for x in args.worker_traffic.split(",")) if args.worker_traffic else None
    worker_visibility = tuple(float(x) for x in args.worker_visibility.split(",")) if args.worker_visibility else None
    if worker_visibility and len(worker_visibility) != len(tasks):
        raise ValueError("--worker-visibility needs one value per task")
    if worker_traffic and len(worker_traffic) != len(tasks):
        p.error("--worker-traffic needs one value per task")
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    resume = (ROOT / args.resume).resolve()
    input_names = json.loads(Path(args.input_names).read_text(encoding="utf-8")) if args.input_names else None
    description = RuleCurriculumDriving(output / "schema", args.seed, "full", args.traffic, route_obs=args.route_obs,
                                        signal_obs=args.signal_obs, route_lookahead=args.route_lookahead,
                                        extra_obs=args.extra_obs, history_lags=args.history_lags, input_names=input_names)
    checkpoint_names = torch.load(resume, map_location="cpu", weights_only=False)["observation_names"]
    input_change = None
    if len(checkpoint_names) < len(description.names) and description.names[:len(checkpoint_names)] == checkpoint_names:
        # Appended inputs: keep every existing weight, start the new input columns at zero.
        previous = torch.load(resume, map_location=args.device, weights_only=False)
        profile = Profile(**previous["profile"]); description.p = profile
        agent = base.make_agent(description, args.device, profile)
        state = dict(previous["model"]); added = len(description.names) - len(checkpoint_names)
        for key in ("actor.0.weight", "critic.0.weight"):
            state[key] = torch.nn.functional.pad(state[key], (0, added))
        agent.model.load_state_dict(state)
        input_change = {"appended": description.names[len(checkpoint_names):],
                        "first_layer_weights_for_these_inputs": "zero-initialised; all existing weights kept"}
    elif checkpoint_names != description.names:
        new_names, description.names = description.names, checkpoint_names
        agent, previous = base.load_agent(resume, description, args.device)
        description.names = new_names
        with torch.no_grad():
            for net in (agent.model.actor, agent.model.critic):
                changed = [k for k, (a, b) in enumerate(zip(checkpoint_names, new_names)) if a != b]
                net[0].weight[:, changed] = 0.
        input_change = {"replaced": [checkpoint_names[k] for k in changed],
                        "with": [new_names[k] for k in changed],
                        "first_layer_weights_for_these_inputs": "zeroed before continuing"}
    else:
        agent, previous = base.load_agent(resume, description, args.device)
    profile = description.p
    widen_info = None
    if args.lr:
        from dataclasses import replace as _replace
        profile = _replace(profile, lr=args.lr, anneal=args.anneal)
        description.p = profile; agent.p = profile
    if args.horizon or args.batch:
        from dataclasses import replace as _replace
        profile = _replace(profile, horizon=args.horizon or profile.horizon, batch=args.batch or profile.batch)
        description.p = profile; agent.p = profile
    if args.widen:
        # Bigger network (Driver Dojo Table 4: 256x256): distil the resumed policy into it, then PPO.
        from dataclasses import replace
        widths = tuple(int(x) for x in args.widen.split(","))
        teacher = agent
        profile = replace(profile, actor=widths, critic=widths, activation=args.widen_activation)
        description.p = profile
        agent = base.make_agent(description, args.device, profile)
        states = []
        for traffic in (3., 1.):
            env_d = RuleCurriculumDriving(output / f"distil_{int(traffic)}", args.seed, "full", traffic, route_obs=args.route_obs,
                                          signal_obs=args.signal_obs, route_lookahead=args.route_lookahead,
                                          extra_obs=args.extra_obs, stop_zone=args.stop_zone)
            for e in range(args.widen_episodes):
                s = env_d.reset(seed=500_000 + 1000 * int(traffic) + e)
                for _ in range(env_d.maximum_steps):
                    states.append(s[0].copy())
                    a_ = teacher.action(s, deterministic=False)[0]   # sample: wider state coverage
                    s, _, d_, tr_, _ = env_d.step(a_)
                    if d_ or tr_:
                        break
            env_d.close()
        x = torch.as_tensor(np.asarray(states, np.float32), device=args.device)
        with torch.no_grad():
            target = teacher.model.distribution(x, None)[0].probs
        optimizer = torch.optim.Adam(agent.model.actor.parameters(), lr=3e-4)
        for _ in range(args.widen_epochs):
            perm = torch.randperm(len(x))
            for i in range(0, len(x), 512):
                idx = perm[i:i + 512]
                probs = agent.model.distribution(x[idx], None)[0].probs
                loss = torch.nn.functional.kl_div(torch.log(probs + 1e-8), target[idx], reduction="batchmean")
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            agree = float((agent.model.distribution(x, None)[0].probs.argmax(-1) == target.argmax(-1)).float().mean())
        for opt in (agent.actor_optimizer, agent.value_optimizer):
            opt.state.clear()
        widen_info = {"widths": widths, "activation": args.widen_activation, "distilled_states": len(x),
                      "argmax_agreement": agree, "source": "Driver Dojo Table 4 PPO hidden_sizes [256,256]"}
        print(json.dumps({"widen": widen_info}), flush=True)
    if args.reset_critic:
        for layer in agent.model.critic:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    count = profile.horizon // len(tasks)
    if count * len(tasks) != profile.horizon:
        raise ValueError("Task count must divide the 2048-transition update")
    start_steps = steps = previous["steps"]
    base.write_json(output / "config.json", {
        "resume": str(resume), "resume_sha256": hashlib.sha256(resume.read_bytes()).hexdigest(),
        "resume_steps": start_steps, "additional_steps": args.steps, "tasks": tasks,
        "transitions_per_task_worker": count, "profile": asdict(profile), "traffic": args.traffic,
        "seed": args.seed, "observation_names": description.names,
        "map": {"name": args.map, "path": str(MAPS[args.map])},
        "signal_task_seconds": args.signal_seconds or SETTINGS.stage_steps * SETTINGS.dt,
        "reward": args.reward, "reward_definition": (RuleCurriculumDriving.DOJO if args.reward == "dojo" else "resumed checkpoint reward"),
        "reward_source": ("Driver Dojo Eq.(1) arXiv 2207.11432 p.8; red/STOP violations mapped to failure (Seongsan choice, cf. Liu 2018 red=-1=collision)"
                          if args.reward == "dojo" else None),
        "critic_reset": args.reset_critic,
        "widen": widen_info,
        "route_obs": args.route_obs, "signal_obs": args.signal_obs, "input_change": input_change,
        "yield_failure": args.yield_failure, "roach_hazard_scale": args.roach_hazard_scale, "stop_zone": args.stop_zone, "blocked_seconds": args.blocked_seconds, "junction_hazard": args.junction_hazard, "desired_speed": args.desired_speed, "speed_reward": args.speed_reward, "history_lags": args.history_lags, "signal_aware": args.signal_aware, "unsafe_lane_change_penalty": args.unsafe_lane_change_penalty, "idle_penalty": args.idle_penalty, "input_names": args.input_names, "signal_next_only": args.signal_next_only, "yield_approach_speed": args.yield_approach_speed,
        "worker_visibility": dict(zip([f"{i+1}:{t}" for i, t in enumerate(tasks)], worker_visibility)) if worker_visibility else None, "fog_speed_cap": args.fog_speed_cap, "fog_yield_scale": args.fog_yield_scale, "stop_reward_floor": args.stop_reward_floor, "collision_penalty": args.collision_penalty,
        "route_lookahead_m": args.route_lookahead, "random_start_lane": args.random_start_lane,
        "worker_traffic": dict(zip([f"{i+1}:{t}" for i, t in enumerate(tasks)], worker_traffic)) if worker_traffic else None,
        "lane_change_penalty": args.lane_change_penalty,
        "roach_max_speed": (args.roach_max_speed or 6.0) if args.reward == "dojo_roach" else None,
        "roach_source": ("carla-roach valeo_action.py r_speed=1-|v-v_des|/v_max; hazards: vehicle 9.5 m/45 deg/heading<=150, offset 8; red/yellow within 18 m and STOP, offset 5; ramp 5 m" if args.reward == "dojo_roach" else None),
        "lane_change_penalty_source": ("Liu et al. 2018 Table 1 lane switching -0.4 relative to collision -1.0, scaled to failure -10" if args.lane_change_penalty else None),
        "unchanged": "observations, 9 actions, reward coefficients, PPO settings of the resumed checkpoint",
        "changed": {
            "task_mixture": {t: f"{tasks.count(t)}/{len(tasks)}" for t in dict.fromkeys(tasks)},
            "task_mixture_source": "Seongsan choice, not a paper setting",
            "signal": "every other signal episode uses the real cycle from a random time instead of a frozen colour",
            "speed": "two-edge tasks starting where the current or next edge limit is below 50 km/h; "
                     "existing Zhu overspeed coefficient -0.002 applied as on the full route",
        },
        "previous_provenance": previous["provenance"], "torch": torch.__version__})
    description.close()

    start = time.monotonic()
    ctx = multiprocessing.get_context("spawn")
    with (ProcessPoolExecutor(max_workers=len(tasks), mp_context=ctx, initializer=initialize_worker,
                              initargs=(str(output), args.seed, tasks, args.traffic, asdict(profile), args.map, args.signal_seconds, args.reward, args.lane_change_penalty, args.roach_max_speed, args.route_obs, args.signal_obs, args.route_lookahead, args.random_start_lane, worker_traffic, args.extra_obs, args.yield_failure, args.roach_hazard_scale, args.stop_zone, args.blocked_seconds, args.junction_hazard, args.desired_speed, args.speed_reward, args.history_lags, args.signal_aware, args.unsafe_lane_change_penalty, args.idle_penalty, input_names, args.signal_next_only, args.yield_approach_speed, worker_visibility, args.fog_speed_cap, args.fog_yield_scale, args.stop_reward_floor, args.collision_penalty)) as pool,
          (output / "metrics.jsonl").open("a", encoding="utf-8") as log):
        while steps < start_steps + args.steps:
            weights = {k: v.detach().cpu() for k, v in agent.model.state_dict().items()}
            progress = (steps - start_steps) / args.steps
            batch = list(pool.map(safe_rollout, [(weights, count, progress)] * len(tasks)))
            stats = agent.update([item[0] for item in batch], progress)
            if not all(torch.isfinite(v).all() for v in agent.model.state_dict().values()):
                # Rare PPO blow-up: drop this update, keep the previous weights, reset optimizer moments.
                agent.model.load_state_dict(weights)
                for opt in (agent.actor_optimizer, agent.value_optimizer):
                    opt.state.clear()
                stats = {"skipped_non_finite_update": True}
            stats = {k: (v if not isinstance(v, float) or v == v and abs(v) != float('inf') else None) for k, v in stats.items()}
            steps += count * len(tasks)
            by_stage = {}
            for item in batch:
                by_stage.setdefault(item[5], Counter()).update(item[1])
            elapsed = time.monotonic() - start
            metrics = {"steps": steps, "elapsed_seconds": round(elapsed, 1),
                       "steps_per_second": round((steps - start_steps) / elapsed),
                       "stage_outcomes": {k: dict(v) for k, v in by_stage.items()}, **stats}
            log.write(json.dumps(metrics, allow_nan=False) + "\n")
            log.flush()
            torch.save({"model": {k: v.detach().cpu() for k, v in agent.model.state_dict().items()},
                        "steps": steps, "observation_names": description.names, "profile": asdict(profile),
                        "stage": "rule_curriculum", "provenance": previous["provenance"],
                        "rule_curriculum": json.loads((output / "config.json").read_text(encoding="utf-8"))["changed"],
                        "training_seed": args.seed, "numeric_correction": False, "failure_time_tail": False,
                        "route_lookahead": args.route_lookahead, "stop_zone": args.stop_zone, "signal_aware": args.signal_aware, "signal_next_only": args.signal_next_only, "fog_speed_cap": args.fog_speed_cap},
                       output / "model.tmp")
            base.replace_with_retry(output / "model.tmp", output / "model.pt")
            if args.save_every and (steps - start_steps) // (count * len(tasks)) % args.save_every == 0:
                import shutil
                shutil.copy2(output / "model.pt", output / f"model_{steps - start_steps:08d}.pt")
            if (steps - start_steps) % (count * len(tasks) * 25) == 0:
                print(json.dumps({k: metrics[k] for k in ("steps", "steps_per_second", "stage_outcomes")}), flush=True)


if __name__ == "__main__":
    main()
