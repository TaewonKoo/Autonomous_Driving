"""Train and audit an ego policy that receives observations and commands SUMO."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch

from paper_training.agents import OnPolicy, Policy
from paper_training.profiles import Profile
from paper_training.candidate_settings import resolve_candidate
from paper_training.observed_driving import ObservedDriving, PPO_PROFILE, SETTINGS, provenance


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    replace_with_retry(temporary, path)


def replace_with_retry(temporary, path):
    # Windows readers may briefly prevent atomic replacement. Retry the I/O
    # operation without changing any rollout, optimizer, or model state.
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.05 * (attempt + 1))


WORKER_ENV = WORKER_POLICY = WORKER_STATE = None
MIXED_STAGES = ("signal", "stop", "junction", "full")


def initialize_worker(directory, seed, stage, traffic, numeric_correction=False, failure_time_tail=False, profile_data=None):
    global WORKER_ENV, WORKER_POLICY, WORKER_STATE
    import atexit
    torch.set_num_threads(1)
    process = multiprocessing.current_process()._identity[-1]
    np.random.seed(seed + process)
    torch.manual_seed(seed + process)
    if stage == "mixed":
        stage = MIXED_STAGES[(process - 1) % len(MIXED_STAGES)]
        traffic = 0. if stage == "signal" else traffic
    WORKER_ENV = ObservedDriving(Path(directory) / f"worker_{process}", seed + process, stage, traffic,
                                reuse_traffic=stage in ("junction", "stop", "full"),
                                numeric_correction=numeric_correction, failure_time_tail=failure_time_tail)
    if stage == "stop":
        WORKER_ENV.curriculum_distance = SETTINGS.length
    worker_profile = Profile(**profile_data) if profile_data is not None else PPO_PROFILE
    WORKER_ENV.p = worker_profile
    WORKER_POLICY = Policy(worker_profile, WORKER_ENV).eval()
    WORKER_STATE = WORKER_ENV.reset()
    atexit.register(WORKER_ENV.close)


def sample_rollout(arguments):
    global WORKER_STATE
    weights, count, progress = arguments
    WORKER_POLICY.load_state_dict(weights)
    if WORKER_ENV.stage == "stop":
        WORKER_ENV.curriculum_distance = SETTINGS.length + (SETTINGS.front_range - SETTINGS.length) * progress
    observations, actions, rewards, continuations = [], [], [], []
    episodes = Counter()
    max_error = 0.
    deceleration_commands = 0
    failures_before = WORKER_ENV.insertion_failures
    with torch.inference_mode():
        for _ in range(count):
            distribution, _ = WORKER_POLICY.distribution(torch.from_numpy(WORKER_STATE))
            action = distribution.sample().numpy()
            state, reward, done, truncated, info = WORKER_ENV.step(action)
            observations.append(WORKER_STATE)
            actions.append(action)
            rewards.append(reward)
            continuations.append([not (done or truncated)])
            max_error = max(max_error, info["speed_error"] or 0.)
            deceleration_commands += info["requested_accel"] < 0
            if done or truncated:
                episodes[info["outcome"]] += 1
                WORKER_STATE = WORKER_ENV.reset()
            else:
                WORKER_STATE = state
    arrays = (np.asarray(observations, np.float32), np.asarray(actions), np.asarray(rewards, np.float32),
              WORKER_STATE, np.ones((count, 1), bool), np.asarray(continuations, bool))
    return (arrays, dict(episodes), max_error, deceleration_commands,
            WORKER_ENV.insertion_failures - failures_before, WORKER_ENV.stage)


def make_agent(env, device="cpu", profile=None):
    agent = OnPolicy(profile or PPO_PROFILE, env, device)
    if env.numeric_correction:
        # Baselines ppo1 mlp_policy.py / tf_util.normc_initializer.
        # TensorFlow output columns correspond to PyTorch Linear weight rows.
        with torch.no_grad():
            for network, final_norm in ((agent.model.actor, .01), (agent.model.critic, 1.)):
                layers = [layer for layer in network if isinstance(layer, torch.nn.Linear)]
                for layer in layers:
                    weight = np.random.randn(layer.out_features, layer.in_features).astype(np.float32)
                    norm = final_norm if layer is layers[-1] else 1.
                    weight *= norm / np.sqrt(np.square(weight).sum(axis=1, keepdims=True))
                    layer.weight.copy_(torch.as_tensor(weight, device=device))
                    layer.bias.zero_()
        for optimizer in (agent.actor_optimizer, agent.value_optimizer):
            optimizer.param_groups[0]["eps"] = 1e-5
    return agent


def load_agent(checkpoint, env, device="cpu", profile=None):
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    env.numeric_correction |= data.get("numeric_correction", False)
    profile = profile or Profile(**data["profile"])
    env.p = profile
    agent = make_agent(env, device, profile)
    if "distance_to_junction" in data["observation_names"]:
        data["observation_names"] = ["distance_to_control" if name == "distance_to_junction" else name
                                     for name in data["observation_names"]]
        data["observation_adapter"] = "Stop-sign preview extends across short road connectors, bounded by the same 50 m range."
    if data["observation_names"] != env.names:
        if data["observation_names"] != env.names[:-1] or env.names[-1] != "ego_speed_limit":
            raise ValueError("Checkpoint observations differ from this environment.")
        # Older first-stage checkpoints predate the appended speed-limit input.
        # A zero column preserves all their outputs exactly until further learning.
        for key in ("actor.0.weight", "critic.0.weight"):
            weight = data["model"][key]
            data["model"][key] = torch.nn.functional.pad(weight, (0, 1))
        data["warm_start_added_inputs"] = ["ego_speed_limit (initially zero weights)"]
    agent.model.load_state_dict(data["model"])
    return agent, data


def evaluate(checkpoint, directory, stage="signal", traffic=0., episodes=100, trace=True, policy_only=False, seed_base=100_000,
             numeric_correction=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    env = ObservedDriving(directory, 100_000, stage, traffic, numeric_correction=numeric_correction)
    agent, data = load_agent(checkpoint, env)
    results = []
    transitions = {key: [] for key in ("observation", "action", "reward", "next_observation",
                                       "terminated", "truncated", "seed", "step", "episode", "light")}
    fields = ["controller", "seed", "light", "step", "action", "requested_accel", "requested_speed",
              "actual_speed", "speed_error", "signal_state", "signal_distance", "outcome"]
    stream = (directory / "transitions.csv").open("w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    controllers = ("policy",) if policy_only else ("policy", "constant_max_acceleration", "constant_zero_acceleration")
    try:
        for controller in controllers:
            for i in range(episodes):
                seed = seed_base + (i // 2 if stage == "signal" else i)
                light = ("red" if i % 2 == 0 else "green") if stage == "signal" else None
                state = env.reset(seed=seed, light=light)
                initial_background = len(env.v.getIDList()) - 1
                requests, errors, minimum_distance, stopped = [], [], SETTINGS.communication_range, False
                actual_speeds = []
                stopped_before_line = False
                initial_speed = env.v.getSpeed("ego")
                first_action = None
                successful_stops = 0
                unblocked_errors = []
                for step in range(1, env.maximum_steps + 1):
                    action = agent.action(state, deterministic=True)[0] if controller == "policy" else np.array([7 if controller == "constant_max_acceleration" else 4])
                    if first_action is None:
                        first_action = int(action[0])
                    before_state = state
                    state, reward, done, truncated, info = env.step(action)
                    if trace and controller == "policy":
                        for key, value in (("observation", before_state[0]), ("action", int(action[0])),
                                           ("reward", reward[0]), ("next_observation", state[0]),
                                           ("terminated", done), ("truncated", truncated), ("seed", seed),
                                           ("step", step), ("episode", i), ("light", light or "natural")):
                            transitions[key].append(value)
                    requests.append(info["requested_accel"])
                    if info["actual_speed"] is not None:
                        actual_speeds.append(info["actual_speed"])
                    if info["speed_error"] is not None:
                        errors.append(info["speed_error"])
                        if not info["blocked_route"]:
                            unblocked_errors.append(info["speed_error"])
                    successful_stops += info["successful_stop_subgoal"]
                    if info["signal"]["present"]:
                        minimum_distance = min(minimum_distance, info["signal"]["distance"])
                        stopped |= info["actual_speed"] is not None and info["actual_speed"] <= 1e-6 and info["signal"]["distance"] <= SETTINGS.length
                        stopped_before_line |= info["actual_speed"] is not None and info["actual_speed"] <= 1e-6 and not info["red_violation"]
                    if trace:
                        writer.writerow(dict(controller=controller, seed=seed, light=light, step=step,
                                             action=int(action[0]), requested_accel=info["requested_accel"],
                                             requested_speed=info["requested_speed"], actual_speed=info["actual_speed"],
                                             speed_error=info["speed_error"], signal_state=info["signal"]["state"],
                                             signal_distance=info["signal"]["distance"], outcome=info["outcome"]))
                    if done or truncated:
                        break
                results.append({"controller": controller, "seed": seed, "light": light, "outcome": info["outcome"],
                                "initial_speed": initial_speed, "first_action": first_action,
                                "initial_background_vehicles": initial_background,
                                "steps": step, "braking_commands": sum(a < 0 for a in requests),
                                "simulation_seconds": step * SETTINGS.dt,
                                "mean_actual_speed_m_s": float(np.mean(actual_speeds)) if actual_speeds else 0.,
                                "stationary_seconds_including_legal_waiting": sum(v <= 1e-6 for v in actual_speeds) * SETTINGS.dt,
                                "maximum_speed_error": max(errors), "stopped_near_red_line": bool(stopped),
                                "maximum_unblocked_speed_error": max(unblocked_errors, default=0.),
                                "successful_stop_subgoals": successful_stops,
                                "stopped_before_red_line": bool(stopped_before_line),
                                "minimum_signal_distance": minimum_distance})
        summary = {}
        for controller in controllers:
            subset = [r for r in results if r["controller"] == controller]
            red = [r for r in subset if r["light"] == "red"]
            green = [r for r in subset if r["light"] == "green"]
            summary[controller] = {"episodes": len(subset), "outcomes": dict(Counter(r["outcome"] for r in subset)),
                                   "red_approaches": len(red) if stage == "signal" else None,
                                   "red_safe_stops": sum(r["stopped_before_red_line"] and r["outcome"] != "red_violation" for r in red) if stage == "signal" else None,
                                   "red_stops_within_5m": sum(r["stopped_near_red_line"] and r["outcome"] != "red_violation" for r in red) if stage == "signal" else None,
                                   "green_approaches": len(green) if stage == "signal" else None,
                                   "green_completions": sum(r["outcome"] == "complete" for r in green) if stage == "signal" else None,
                                   "maximum_speed_error": max(r["maximum_speed_error"] for r in subset),
                                   "maximum_unblocked_speed_error": max(r["maximum_unblocked_speed_error"] for r in subset),
                                   "successful_stop_subgoals": sum(r["successful_stop_subgoals"] for r in subset)}
        report = {"checkpoint": str(Path(checkpoint).resolve()), "training_steps": data["steps"],
                  "candidate_id": data.get("candidate_id"), "training_seed": data.get("training_seed"),
                  "training_profile": data["profile"],
                  "numeric_correction": env.numeric_correction,
                  "stage": stage, "traffic": traffic, "speed_mode": 96, "lane_change_mode": 0,
                  "evaluation_seed_base": seed_base,
                  "scope": {"signal": "Static red/green signal response probe",
                            "junction": "Two-edge local junction tasks",
                            "stop": "Local mandatory-stop tasks, including short connecting roads",
                            "full": "Complete outermost route"}[stage],
                  "summary": summary, "episodes": results}
        write_json(directory / "evaluation.json", report)
        if trace and transitions["action"]:
            arrays = {key: np.asarray(value) for key, value in transitions.items()}
            arrays["observation_names"] = np.asarray(env.names)
            np.savez_compressed(directory / "policy_transitions.npz", **arrays)
        return report
    finally:
        stream.close()
        env.close()


def audit(directory, numeric_correction=False):
    directory = Path(directory)
    env = ObservedDriving(directory, numeric_correction=numeric_correction)
    records = []
    try:
        for light in ("red", "green"):
            state = env.reset(seed=12345, light=light, distance=SETTINGS.rear_range, initial_speed=0.)
            initial = state.tolist()
            errors = []
            for step in range(100):
                state, reward, done, truncated, info = env.step([7])
                if info["speed_error"] is not None:
                    errors.append(info["speed_error"])
                if done or truncated:
                    break
            records.append({"light": light, "initial_observation": initial, "outcome": info["outcome"],
                            "final_reward": reward[0], "final_speed": info["actual_speed"],
                            "speed_mode": info["speed_mode"], "maximum_speed_error": max(errors)})
        assert records[0]["outcome"] == "red_violation", records[0]
        assert records[1]["outcome"] == "complete", records[1]
        assert all(r["maximum_speed_error"] < 1e-4 for r in records), records
        time_cost = SETTINGS.time_objective * SETTINGS.time_weight * (1. if numeric_correction else SETTINGS.dt)
        assert abs(records[0]["final_reward"] - (time_cost - .25 - .02 * records[0]["final_speed"])) < 1e-6
        assert abs(records[1]["final_reward"] - (1. + time_cost)) < 1e-6
        changed = np.flatnonzero(np.asarray(records[0]["initial_observation"])[0] !=
                                 np.asarray(records[1]["initial_observation"])[0]).tolist()
        assert changed and all(env.names[i].startswith(("signal_", "time_to_green")) for i in changed)
        report = {"passed": True, "records": records, "changed_features": [env.names[i] for i in changed],
                  "conclusion": "Fixed acceleration crosses red. SUMO did not brake. Red/green changes observations only."}
        write_json(directory / "control_audit.json", report)
        return report
    finally:
        env.close()


def transition_probe(checkpoint, directory, episodes, seed_base=200_000):
    """External red-to-green change; all vehicle actions still come from the policy."""
    directory = Path(directory)
    env = ObservedDriving(directory)
    agent, data = load_agent(checkpoint, env)
    records = []
    try:
        for i in range(episodes):
            state = env.reset(seed=seed_base + i, light="red")
            env.maximum_steps = 2 * SETTINGS.stage_steps
            stopped = False
            error = 0.
            for step in range(2 * SETTINGS.stage_steps):
                if step == SETTINGS.stage_steps:
                    env.force_light("green")
                    state = env.observation()
                action, _ = agent.action(state, deterministic=True)
                state, _, done, truncated, info = env.step(action)
                if step < SETTINGS.stage_steps:
                    stopped |= info["actual_speed"] is not None and info["actual_speed"] <= 1e-6
                error = max(error, info["speed_error"] or 0.)
                if done or truncated:
                    break
            records.append({"seed": seed_base + i, "stopped_on_red": bool(stopped),
                            "outcome": info["outcome"], "steps": step + 1, "maximum_speed_error": error})
        report = {"checkpoint": str(Path(checkpoint).resolve()), "training_steps": data["steps"],
                  "experiment": "20 s red followed by 20 s green; exogenous signal intervention, no ego action filter",
                  "evaluation_seed_base": seed_base,
                  "episodes": episodes, "stopped_then_completed": sum(r["stopped_on_red"] and r["outcome"] == "complete" for r in records),
                  "outcomes": dict(Counter(r["outcome"] for r in records)), "records": records}
        write_json(directory / "transition_probe.json", report)
        return report
    finally:
        env.close()


def train(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    profile, candidate = PPO_PROFILE, None
    if args.candidate_config:
        profile, candidate = resolve_candidate(args.candidate_config, args.candidate_id)
        if (args.stage != "full" or args.traffic != 1. or args.workers != 4
                or args.numeric_correction or args.failure_time_tail or not args.resume):
            raise ValueError("Candidate tuning keeps full-route traffic1, four workers, original rewards/dynamics and a common initial checkpoint")
        if hashlib.sha256(Path(args.resume).read_bytes()).hexdigest() != candidate["initial_checkpoint_sha256"]:
            raise ValueError("Candidate initial checkpoint differs from the frozen common weights")
    if args.stage == "mixed" and args.workers != len(MIXED_STAGES):
        raise ValueError("Mixed training requires one worker for each of the four recorded task stages.")
    description = ObservedDriving(output / "schema", args.seed,
                                  "full" if args.stage == "mixed" else args.stage, args.traffic,
                                  numeric_correction=args.numeric_correction, failure_time_tail=args.failure_time_tail)
    description.p = profile
    agent = make_agent(description, args.device, profile)
    start_steps = 0
    if args.resume:
        agent, previous = load_agent(args.resume, description, args.device, profile)
        start_steps = previous["steps"]
    run_provenance = provenance(args.numeric_correction, args.failure_time_tail)
    if candidate:
        run_provenance["ppo"]["values"].update({key: getattr(profile, key) for key in run_provenance["ppo"]["values"]})
        run_provenance["candidate_tuning"] = candidate
    write_json(output / "config.json", {"provenance": run_provenance, "observation_names": description.names,
                                      "effective_profile": asdict(profile), "candidate": candidate,
                                      "runtime": vars(args), "batch_semantics": ("2048 transitions per worker; N*2048 per update" if args.numeric_correction else "2048 total transitions per update, divided across workers"),
                                      "mixed_stage_sampling": ({"stages": MIXED_STAGES, "transitions_per_stage": 512,
                                                                "source": "Seongsan task sampling choice; not a published paper hyperparameter"}
                                                               if args.stage == "mixed" else None),
                                      "episode_boundary": "Finite-horizon timeout treated as terminal for this task",
                                      "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None})
    start = time.monotonic()
    count = profile.horizon if args.numeric_correction else profile.horizon // args.workers
    if not args.numeric_correction and count * args.workers != profile.horizon:
        raise ValueError("Worker count must divide 2048.")
    steps = start_steps
    ctx = multiprocessing.get_context("spawn")
    with (ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=initialize_worker,
                              initargs=(str(output), args.seed, args.stage, args.traffic, args.numeric_correction, args.failure_time_tail, asdict(profile))) as pool,
          (output / "metrics.jsonl").open("a", encoding="utf-8") as log):
        while steps < args.steps:
            weights = {k: v.detach().cpu() for k, v in agent.model.state_dict().items()}
            curriculum_progress = (steps - start_steps) / max(1, args.steps - start_steps)
            batch = list(pool.map(sample_rollout, [(weights, count, curriculum_progress)] * args.workers))
            if args.stage == "mixed" and {item[5] for item in batch} != set(MIXED_STAGES):
                raise RuntimeError("Mixed update did not receive one rollout per task; refusing an unbalanced batch.")
            learning_progress = ((steps - start_steps) / max(1, args.steps - start_steps)
                                 if candidate else steps / args.steps)
            stats = agent.update([item[0] for item in batch], learning_progress)
            steps += count * args.workers
            outcomes = Counter()
            task_outcomes = {}
            for item in batch:
                outcomes.update(item[1])
                task_outcomes.setdefault(item[5], Counter()).update(item[1])
            elapsed = time.monotonic() - start
            metrics = {"steps": steps, "elapsed_seconds": elapsed, "steps_per_second": (steps - start_steps) / elapsed,
                       "outcomes": dict(outcomes), "max_speed_error": max(item[2] for item in batch),
                       "task_outcomes": {task: dict(counts) for task, counts in task_outcomes.items()},
                       "rejected_initializations": sum(item[4] for item in batch),
                       "braking_commands": sum(item[3] for item in batch), **stats}
            metrics["learning_rate"] = agent.actor_optimizer.param_groups[0]["lr"]
            log.write(json.dumps(metrics, allow_nan=False) + "\n")
            log.flush()
            checkpoint = {"model": {k: v.detach().cpu() for k, v in agent.model.state_dict().items()},
                          "steps": steps, "observation_names": description.names, "profile": asdict(profile),
                          "stage": args.stage, "provenance": run_provenance,
                          "candidate_id": args.candidate_id if candidate else None,
                          "training_seed": args.seed,
                          "numeric_correction": args.numeric_correction,
                          "failure_time_tail": args.failure_time_tail}
            torch.save(checkpoint, output / "model.tmp")
            replace_with_retry(output / "model.tmp", output / "model.pt")
            write_json(output / "progress.json", metrics)
            print(json.dumps(metrics), flush=True)
    report = evaluate(output / "model.pt", output / "evaluation",
                      "full" if args.stage == "mixed" else args.stage, args.traffic, args.episodes,
                      policy_only=args.policy_only or args.numeric_correction,
                      seed_base=args.evaluation_seed, numeric_correction=args.numeric_correction)
    print(json.dumps(report["summary"]), flush=True)
    if candidate and args.candidate_probes:
        signal = evaluate(output / "model.pt", output / "signal_evaluation", "signal", 0.,
                          candidate["evaluation"]["signal_reference_episodes"], trace=False, policy_only=True,
                          seed_base=candidate["evaluation"]["signal_reference_seed_base"])
        transition = transition_probe(output / "model.pt", output / "red_to_green",
                                      candidate["evaluation"]["red_to_green_episodes"],
                                      candidate["evaluation"]["signal_reference_seed_base"])
        print(json.dumps({"signal": signal["summary"], "transition": transition["outcomes"]}), flush=True)
    if candidate:
        write_json(output / "completed.json", {"candidate_id": args.candidate_id, "seed": args.seed,
                                              "steps": steps, "additional_steps": steps - start_steps,
                                              "evaluation_complete": True, "probes_complete": args.candidate_probes})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("audit", "train", "evaluate", "transition"))
    p.add_argument("--output", default="results/observed_driving_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--stage", choices=("signal", "junction", "stop", "full", "mixed"), default="signal")
    p.add_argument("--traffic", type=float, default=0.)
    p.add_argument("--steps", type=int, default=PPO_PROFILE.steps)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--resume")
    p.add_argument("--checkpoint")
    p.add_argument("--policy-only", action="store_true")
    p.add_argument("--evaluation-seed", type=int, default=100_000)
    p.add_argument("--numeric-correction", action="store_true",
                   help="Correct previous PPO numbers without changing its observation/action/network/map")
    p.add_argument("--failure-time-tail", action="store_true",
                   help="Experimental failure-only discounted remaining time cost; no new reward coefficient")
    p.add_argument("--candidate-config")
    p.add_argument("--candidate-id")
    p.add_argument("--candidate-probes", action="store_true")
    args = p.parse_args()
    if bool(args.candidate_config) != bool(args.candidate_id):
        p.error("--candidate-config and --candidate-id must be supplied together")
    if args.candidate_config and args.mode != "train":
        p.error("Candidate definitions configure training; evaluation reads the saved checkpoint profile")
    if args.stage == "mixed" and args.mode != "train":
        p.error("mixed is a training sampler; select a specific task stage for evaluation")
    if args.numeric_correction and args.mode == "train" and (args.stage != "full" or args.resume):
        p.error("Numeric correction starts a fresh PPO on the existing full-route task; no mixture or old-weight resume")
    if args.failure_time_tail and (args.numeric_correction or args.stage != "full" or args.mode != "train"):
        p.error("Failure-time-tail is a separate full-route training experiment; evaluate both arms in the original environment")
    torch.set_num_threads(1)
    if args.mode == "audit":
        print(json.dumps(audit(args.output, args.numeric_correction)))
    elif args.mode in ("evaluate", "transition"):
        if not args.checkpoint:
            p.error("--checkpoint is required for evaluation")
        if args.mode == "transition":
            report = transition_probe(args.checkpoint, args.output, args.episodes, args.evaluation_seed)
            print(json.dumps({k: v for k, v in report.items() if k != "records"}))
        else:
            print(json.dumps(evaluate(args.checkpoint, args.output, args.stage, args.traffic, args.episodes,
                                      policy_only=args.policy_only, seed_base=args.evaluation_seed,
                                      numeric_correction=args.numeric_correction)["summary"]))
    else:
        train(args)


if __name__ == "__main__":
    main()
