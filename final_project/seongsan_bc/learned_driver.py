"""Observation-based route-following policy search and student imitation experiments."""
import argparse
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from study import Visual, ROOT, CK, Policy, Profile

OUT = ROOT / "driver_study"
# Free parameters of the actor: cruise, braking, front gap, rear gap, headway.
INITIAL = [7.0, 2.6, 7., 9., 1.2]
FEATURE_NAMES = [
    "ego_speed_mps", "lane_speed_limit_mps", "planned_lane_direction",
    "side_front_gap_m", "side_rear_gap_m", "side_front_relative_speed_mps",
    "side_rear_relative_speed_mps", "leader_gap_m", "leader_speed_mps",
    "mandatory_stop_distance_m", "observed_red_or_yellow_distance_m", "signal_readable",
    "control_distance_m", "route_index", "visibility_range_m", "lane_position_m",
    "predicted_conflict_distance_m",
]


def lane_plan(env):
    goals = []
    downstream = None
    for i in reversed(range(len(env.route))):
        edge = env.net.getEdge(env.route[i])
        lanes = edge.getLanes()
        costs = []
        for lane in lanes:
            if downstream is None:
                cost = 0 if lane.getIndex() == int(env.full_arrival_lane) else 1e4
            else:
                costs_next = [downstream[c.getToLane().getIndex()] for c in lane.getOutgoing()
                              if c.getTo().getID() == env.route[i+1]]
                cost = min(costs_next, default=1e4) * .95
            costs.append(cost)
        # Prefer to make a required maneuver on a longer road.
        effort = 1 + 30 / edge.getLength()
        downstream = [min(costs[j] + abs(j-k)*effort for j in range(len(lanes))) for k in range(len(lanes))]
        goals.append([min(range(len(lanes)),key=lambda j:costs[j] + abs(j-k)*effort)
                      for k in range(len(lanes))])
    return list(reversed(goals))


def features(env, obs, plan, static=None, params=INITIAL):
    x = obs[0]
    v = env.v
    road = v.getRoadID("ego")
    lane = v.getLaneIndex("ego")
    route_index = v.getRouteIndex("ego")
    target = lane if road.startswith(":") else plan[route_index][lane]
    direction = int(np.sign(target-lane))
    speed = float(x[1])*env.s.speed
    side = 200 if direction < 0 else 207
    front = x[side+2]*env.radius if x[side+1] else 100.
    rear = x[side+5]*env.radius if x[side+4] else 100.
    lead_gap = float(x[5])*50 if x[5]<.9999 else 100.
    lead_speed = float(x[6])*env.s.speed
    stop = x[195]*50 if x[196] and not x[198] else 999.
    light = x[190]*300 if x[187] or x[188] else 999.
    conflict = 100.
    if static is not None:
        from driver_map import conflict_features
        for de,dh,hv_speed in conflict_features(env,static,speed):
            if hv_speed<.35 and dh>2.5:
                continue
            hv_eta=max(0., dh-1.)/max(.4,hv_speed)
            hv_exit=(dh+7.)/max(.4,hv_speed)
            ego_eta=max(0.,de-2.)/max(2.,speed+.3)
            ego_exit=(de+6.)/max(2.5,speed+.3)
            margin=params[5]
            if hv_eta<ego_exit+margin and hv_exit>ego_eta-margin:
                conflict=min(conflict,max(0.,de))
    return np.asarray([speed, x[199]*env.s.speed, direction, front, rear, x[side+3]*14,
        x[side+6]*14, lead_gap, lead_speed, stop, light, float(x[186]), x[195]*50,
        route_index, env.radius, v.getLanePosition("ego"),conflict], dtype=np.float32)


def structured_action(f, p):
    speed, limit, direction, front, rear, fdv, rdv, gap, lead_speed, stop, light = f[:11]
    cruise, brake, front_min, rear_min, headway = p[:5]
    move = int(direction) if (front>front_min+max(0.,-fdv)*headway and
                              rear>rear_min+max(0.,rdv)*headway) else 0
    if len(p)>5 and f[15]<6.5:
        move=0
    target = min(cruise,limit)
    if direction and not move:
        target = min(target,3.5)
    target = min(target, math.sqrt(max(0.,lead_speed**2+2*brake*(gap-3.-speed*.45))))
    if gap<2.5:
        target=0.
    if min(stop,light)<100:
        target=min(target, math.sqrt(2*brake*max(0., min(stop,light)-2.8)))
    if len(p)>5 and f[16]<90:
        target=min(target, math.sqrt(2*brake*max(0., f[16]-p[6])))
    acc = 0 if speed>target+.05 else 2 if speed+.34<target else 1
    return 3*acc+1+move


def batch(task):
    tag, start, n, traffic, p, mode, *options = task
    stride=options[0] if options else 1
    directory=OUT/tag/f"{start}_{traffic}"
    directory.mkdir(parents=True,exist_ok=True)
    env=Visual(directory,start,traffic,0)
    plan=lane_plan(env)
    static=json.loads((OUT/"intersection_geometry.json").read_text()) if len(p)>5 else None
    model=None
    torch.set_num_threads(1)
    if mode=="audit":
        model=Policy(Profile(**torch.load(CK,map_location="cpu",weights_only=True)["profile"]),env).eval()
        model.load_state_dict(torch.load(ROOT/"traffic_trial/extension/train_dense_26_4/step_180224.pt",
                                         map_location="cpu",weights_only=True))
    elif mode.startswith("student"):
        model=torch.jit.load(str(OUT/(mode+".pt")),map_location="cpu").eval()
    rows=[]
    inputs, labels=[],[]
    try:
        for seed in range(start,start+n*stride,stride):
            obs=env.reset(seed=seed)
            counts=Counter()
            while True:
                f=features(env,obs,plan,static,p)
                if model is None:
                    a=structured_action(f,p)
                else:
                    with torch.inference_mode():
                        if mode=="audit":
                            a=int(model.distribution(torch.from_numpy(obs))[0].probs.argmax(-1).item())
                        else:
                            a=int(model(torch.from_numpy(f[None])).argmax(-1).item())
                if tag.startswith("collect"):
                    inputs.append(f)
                    labels.append(structured_action(f,p) if model is not None else a)
                d=int(a%3)-1
                side=200 if d<0 else 207
                near=obs[0,side:side+7]
                close=bool((near[1] and near[2]*env.radius<5) or (near[4] and near[5]*env.radius<5))
                road=env.v.getRoadID("ego")
                obs,r,end,tr,info=env.step([a])
                if info["lane_change_applied"]:
                    counts["changes"]+=1
                    counts["close_changes"]+=int(close)
                    counts["collision_changes"]+=int(info["collision"])
                    counts["close_collisions"]+=int(close and info["collision"])
                if end or tr:
                    ev=[(e.type,e.lane,e.collider,e.victim) for e in env.sim.simulation.getCollisions()
                        if "ego" in (e.collider,e.victim)] if info["collision"] else []
                    rows.append(dict(seed=seed,traffic=traffic,outcome=info["outcome"],steps=env.steps,
                        meters=env.last_distance-env.start_distance,road=road,features=f.tolist(),events=ev,**counts))
                    break
    finally:
        env.close()
    if inputs:
        np.savez_compressed(directory/"labels.npz",x=np.asarray(inputs),y=np.asarray(labels,np.int64),
                            names=np.asarray(FEATURE_NAMES))
    (directory/"rows.json").write_text(json.dumps(rows))
    return rows


def run_group(pool,tag,p,mode,start,n,traffic=(4,),block=5):
    tasks=[(tag,start+i,min(block,n-i),t,p,mode) for i in range(0,n,block) for t in traffic]
    return sum(pool.map(batch,tasks),[])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("tag")
    parser.add_argument("start",type=int)
    parser.add_argument("n",type=int)
    parser.add_argument("--traffic",type=int,nargs="+",default=[4])
    parser.add_argument("--params",type=float,nargs="+")
    args=parser.parse_args()
    if args.mode in ("best","best_neural"):
        config="recommended_policy.json" if args.mode=="best" else "recommended_neural.json"
        manifest=json.loads((OUT/config).read_text())
        args.mode=manifest["model"]
        args.params=manifest["parameters"]
    elif args.params is None:
        if args.mode.startswith("student_opt"):
            args.params=json.loads((OUT/"parameter_policy.json").read_text())["parameters"]
        else:
            args.params=([7,2.6,7,9,1.2,.8,3.] if args.mode.startswith("student") else INITIAL)
    with ProcessPoolExecutor(8) as pool:
        rows=run_group(pool,args.tag,args.params,args.mode,args.start,args.n,args.traffic)
    print(Counter(r["outcome"] for r in rows))
