"""Headless evaluation of the final Seongsan PPO model (libsumo, parallel).

usage: python evaluate.py [--checkpoint models/final_ppo/model.pt] [--traffic 1.5] [--episodes 1000] [--seed 920000] [--workers 8]

Each episode = one lap of the full Seongsan route from a fixed start, seed = <seed> + episode index.
Outcomes: complete, collision, red_violation, stop_violation (passing a STOP line without a full stop),
route_failure, timeout. Prints the counts, the collision rate with a 95% interval and the mean speed,
and writes one JSON line per episode to results/eval_<traffic>x.jsonl.
"""
import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(job):
    ck, traffic, seeds, tag = job
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, ROOT)
    from paper_training.rule_curriculum import use_map, RuleCurriculumDriving, history_lags_from_names
    from train_observed_driving import load_agent
    d = torch.load(ck, map_location="cpu", weights_only=False)
    names = d["observation_names"]
    use_map("pluslane_v4")
    env = RuleCurriculumDriving(os.path.join(ROOT, "results", "eval_tmp", tag), 100_000, "full", traffic,
                                route_obs="coarse", signal_obs="color", route_lookahead=float(d.get("route_lookahead", 50.)),
                                extra_obs=3, history_lags=history_lags_from_names(names),
                                signal_aware=bool(d.get("signal_aware", False)), stop_zone=d.get("stop_zone"),
                                input_names=names, signal_next_only=bool(d.get("signal_next_only", False)))
    agent, _ = load_agent(ck, env)
    rows = []
    for seed in seeds:
        s = env.reset(seed=seed); speeds = []; road = None; info = {"outcome": "timeout"}
        for _ in range(env.maximum_steps):
            if "ego" in env.v.getIDList():
                road = env.v.getRoadID("ego"); speeds.append(env.v.getSpeed("ego"))
            s, _, done, trunc, info = env.step(agent.action(s, deterministic=True)[0])
            if done or trunc:
                break
        place = road.rsplit("_", 1)[0] if road and road.startswith(":") else road
        rows.append({"seed": seed, "outcome": info["outcome"], "place": place,
                     "mean_kmh": round(3.6 * sum(speeds) / max(len(speeds), 1), 1),
                     "seconds": round(len(speeds) * env.s.dt, 1)})
    env.close()
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=os.path.join(ROOT, "models", "final_ppo", "model.pt"))
    p.add_argument("--traffic", type=float, default=1.)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=920_000)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    a = p.parse_args()
    seeds = list(range(a.seed, a.seed + a.episodes))
    chunks = [seeds[i::a.workers] for i in range(a.workers) if seeds[i::a.workers]]
    jobs = [(os.path.abspath(a.checkpoint), a.traffic, c, f"t{a.traffic}_w{i}") for i, c in enumerate(chunks)]
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        rows = sorted((r for part in pool.map(run, jobs) for r in part), key=lambda r: r["seed"])
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", f"eval_{a.traffic:g}x.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    n = len(rows); c = collections.Counter(r["outcome"] for r in rows); q = c["collision"] / n
    print(f"traffic {a.traffic:g}x, {n} episodes (seeds {a.seed}..{a.seed + n - 1})")
    print("outcomes:", dict(c))
    print(f"collision rate {100 * q:.1f}% (95% CI +-{196 * math.sqrt(q * (1 - q) / n):.1f}), completion {100 * c['complete'] / n:.1f}%")
    print(f"mean speed {sum(r['mean_kmh'] for r in rows) / n:.1f} km/h, mean lap {sum(r['seconds'] for r in rows) / n:.1f} s")
    print("collisions at:", collections.Counter(r["place"] for r in rows if r["outcome"] == "collision").most_common(8))
    print("per-episode results:", out)


if __name__ == "__main__":
    main()
