"""Watch a trained Seongsan PPO policy drive the full route in sumo-gui (episodes repeat until the window closes)."""
import argparse
from pathlib import Path
import time

import torch

from paper_training.observed_driving import ObservedDriving
from paper_training.rule_curriculum import MAPS, use_map, history_lags_from_names
from train_observed_driving import load_agent


ROOT = Path(__file__).resolve().parent


def add_map_decorations(map_dir, work_dir):
    """The training environment loads only the net and routes; for viewing, also load the map's
    markings (seongsan.add.xml) and view settings (seongsan.gui.xml), as seongsan.sumocfg does."""
    import re
    import traci
    import traci.connection
    work_dir.mkdir(parents=True, exist_ok=True)
    settings = work_dir / "view.gui.xml"
    # Playback speed is paced here, so drop the file's 200 ms GUI delay.
    text = (Path(map_dir) / "seongsan.gui.xml").read_text(encoding="utf-8")
    settings.write_text(re.sub(r"\s*<delay[^>]*/>", "", text), encoding="utf-8")
    extra = ["-a", str(Path(map_dir) / "seongsan.add.xml"), "-g", str(settings)]
    start, load = traci.start, traci.connection.Connection.load
    traci.start = lambda cmd, *a, **k: start([*cmd, *extra], *a, **k)
    traci.connection.Connection.load = lambda self, args: load(self, [*args, *extra])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="models/final_ppo/model.pt")
    p.add_argument("--seed", type=int, default=800_000)
    p.add_argument("--speed", type=float, default=1., help="playback speed relative to real time")
    p.add_argument("--traffic", type=float, default=1.)
    p.add_argument("--map", choices=tuple(MAPS), default="pluslane_v4")
    p.add_argument("--episodes", type=int, default=0, help="stop after this many laps and keep the window open (0: repeat)")
    p.add_argument("--pause", type=float, default=0., help="seconds to hold the first frame (traffic already warmed up) before the ego moves")
    args = p.parse_args()
    map_dir = use_map(args.map)
    add_map_decorations(map_dir, ROOT / "results/gui_view")
    torch.set_num_threads(1)
    names = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)["observation_names"]
    if "route_lane_usable_distance" in names or "time_to_green_unused" in names or "conflict_0_distance" in names:
        from paper_training.rule_curriculum import RuleCurriculumDriving, history_lags_from_names
        env = RuleCurriculumDriving(ROOT / "results/gui_view", 100_000, "full", args.traffic, gui=True,
                                    route_obs="coarse" if "route_lane_usable_distance" in names else "lane",
                                    signal_obs="color" if "time_to_green_unused" in names else "color_time",
                                    route_lookahead=float(torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False).get("route_lookahead", 50.)),
                                    extra_obs=3 if "foe_0_ego_to_crossing" in names else 2 if "foe_0_distance" in names else int("conflict_0_distance" in names), history_lags=history_lags_from_names(names), signal_aware=bool(torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False).get("signal_aware", False)),
                                    stop_zone=torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False).get("stop_zone"),
                                    input_names=names,
                                    signal_next_only=bool(torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False).get("signal_next_only", False)))
    else:
        env = ObservedDriving(ROOT / "results/gui_view", 100_000, "full", args.traffic, gui=True)
    agent, _ = load_agent(ROOT / args.checkpoint, env)
    episode = 0
    try:
        while True:
            state = env.reset(seed=args.seed + episode)
            gui = env.sim.gui
            gui.trackVehicle("View #0", "ego")
            gui.setZoom("View #0", 400)
            env.v.setColor("ego", (255, 0, 0, 255))
            print(f"episode {episode} (seed {args.seed + episode})", flush=True)
            print(f"  {len(env.v.getIDList()) - 1} vehicles already on the map after {env.sim.simulation.getTime():.0f} s of warm-up", flush=True)
            time.sleep(args.pause)
            for _ in range(env.maximum_steps):
                tick = time.perf_counter()
                state, _, done, truncated, info = env.step(agent.action(state, deterministic=True)[0])
                time.sleep(max(0., env.s.dt / args.speed - (time.perf_counter() - tick)))
                if done or truncated:
                    break
            print(f"  outcome: {info['outcome']}", flush=True)
            time.sleep(2)
            episode += 1
            if args.episodes and episode >= args.episodes:
                print("done; close the window to exit", flush=True)
                while True:  # raises once the window is closed
                    env.sim.simulation.getTime()
                    time.sleep(.5)
    except Exception as error:  # closing the GUI window ends the TraCI connection
        print(f"stopped: {type(error).__name__}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
