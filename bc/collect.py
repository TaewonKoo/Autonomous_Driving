"""Collect the locked Seongsan BC in closed loop, one weather per episode."""
import csv, hashlib, json, math, os, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parent
STUDY = ROOT
sys.path.insert(0,str(STUDY))
from learned_driver import Visual, OUT, FEATURE_NAMES, features, lane_plan, torch

FIELDS = 'episode t observation action_raw action.accel_raw action.lane_change_raw action.lane_change action.lane_change_applied reward terminated truncated state detected_vehicles event.collided event.arrived event.lane_before event.lane_after next_observation next_state weather_id obs_profile_id signal_exists_map red_obs yellow_obs green_obs stopline_dist_m red_violation stop_violation'.split()
WEATHER = ['clear','light_fog','dense_fog']
# d50 for cars, lights and line, Gaussian position error (m).
CONFIG = json.loads((ROOT/'configs/collection.json').read_text())
PARAM = [CONFIG['weather'][name] for name in ('clear','light','dense')]
def pack(x):
    return json.dumps(x,separators=(',',':'),allow_nan=False)

class Sensor:
    def __init__(self, weather, seed):
        self.parameters = PARAM[weather]
        self.rng = np.random.default_rng(seed+CONFIG['sensor_seed_offset'])
        self.latents = {}
    def visible(self,key,distance,index):
        if self.parameters is None:
            return True
        # Correlated standard-normal latent; normal CDF gives a uniform draw.
        z = self.rng.normal()
        if key in self.latents:
            z = CONFIG['latent_persistence']*self.latents[key] + math.sqrt(1-CONFIG['latent_persistence']**2)*z
        self.latents[key] = z
        p = 1/(1+math.exp(min(700,(distance-self.parameters[index])/CONFIG['logistic_scale_m'])))
        return .5*(1+math.erf(z/math.sqrt(2))) < p
    def error(self):
        return self.rng.normal(0,self.parameters[3]) if self.parameters else 0.

def collect(task):
    ep,seed,weather,output=task
    folder=output/str(seed)
    folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    lock=json.loads((OUT/'v2_selection.json').read_text())['selected']['student_opt_bc']
    ck=OUT/'student_opt_bc.pt'
    assert hashlib.sha256(ck.read_bytes()).hexdigest()==lock['sha256']
    model=torch.jit.load(str(ck),map_location='cpu').eval()
    env=Visual(folder,seed,CONFIG['traffic_multiplier'],0)
    plan=lane_plan(env)
    geometry=json.loads((OUT/'intersection_geometry.json').read_text())
    sensor=Sensor(weather,seed)
    tm=time.perf_counter()
    try:
        raw=env.reset(seed=seed)
        v,sim=env.v,env.sim
        lanes={k:i for i,k in enumerate(sorted(sim.lane.getIDList()))}
        signals=sorted(sim.trafficlight.getIDList())
        ids={'ego':0}
        stops=[(e,env.net.getEdge(e).getLength()) for e in env.route if env.net.getEdge(e).getToNode().getType()=='allway_stop']
        pairs={(env.net.getLane(a).getEdge().getID(),env.net.getLane(b).getEdge().getID()) for s in signals for group in sim.trafficlight.getControlledLinks(s) for a,b,_ in group}
        def snapshot():
            cars={}
            for k in sorted(v.getIDList()):
                x,y=v.getPosition(k)
                cars[k]=dict(vehicle_id=ids.setdefault(k,len(ids)),lane_id=lanes[v.getLaneID(k)],x_m=x,y_m=y,speed_mps=v.getSpeed(k),heading_deg=v.getAngle(k))
            lights=[dict(tls_id=i,link_index=j,state_gt=ord(c)) for i,s in enumerate(signals) for j,c in enumerate(sim.trafficlight.getRedYellowGreenState(s))]
            return pack(dict(sim_time_s=sim.simulation.getTime(),ego_id=0,vehicles=list(cars.values()),signals=lights)),cars
        def observe(raw,cars):
            ego=cars['ego']
            h=math.radians(ego['heading_deg'])
            detections={}
            errors={}
            for k,c in cars.items():
                if k=='ego':continue
                dx,dy=c['x_m']-ego['x_m'],c['y_m']-ego['y_m']
                d=math.hypot(dx,dy)
                if d>env.radius or not sensor.visible(k,d,0):continue
                e1,e2=sensor.error(),sensor.error()
                errors[k]=e1
                detections[k]=dict(track_id=ids[k],rel_x_m=dx*math.sin(h)+dy*math.cos(h)+e1,rel_y_m=-dx*math.cos(h)+dy*math.sin(h)+e2,observed_speed_mps=c['speed_mps'])
            # Filter tracked HVs before the existing conflict predictor computes distances.
            class LaneView:
                def getLastStepVehicleIDs(self,lane):
                    return [k for k in sim.lane.getLastStepVehicleIDs(lane) if k in detections]
            class VehicleView:
                def getLanePosition(self,k):
                    return v.getLanePosition(k)+errors.get(k,0)*math.cos(math.radians(cars[k]['heading_deg']-ego['heading_deg']))
                def __getattr__(self,key):
                    return getattr(v,key)
            proxy=SimpleNamespace(v=VehicleView(),sim=SimpleNamespace(lane=LaneView()),radius=env.radius,s=env.s)
            f=features(proxy if weather else env,raw,plan,geometry,lock['parameters'])
            ahead=v.getNextTLS('ego')
            bits=[int(x) for x in raw[0,187:190]]
            future=env.route[max(0,v.getRouteIndex('ego')):]
            present=int(bool(ahead) or any(p in pairs for p in zip(future,future[1:])))
            ds=[x[2] for x in ahead if x[2]>=0]
            for e,pos in stops:
                if e in future:
                    d=v.getDrivingDistance('ego',e,pos)
                    if d>=0:ds.append(d)
            line=min(ds,default=-1)
            if weather:
                side=-1 if f[2]<0 else 1
                for sign,gap,dv in [(1,3,5),(-1,4,6)]:
                    item=env.near.get((side,sign))
                    if f[gap]<100 and item:
                        k=item[1]
                        if k in detections:f[gap]=max(0,f[gap]+sign*errors[k])
                        else:f[gap],f[dv]=100,0
                lead=v.getLeader('ego',50)
                if f[7]<100 and lead and lead[0]:
                    k=lead[0]
                    if k in detections:f[7]=max(0,f[7]+errors[k])
                    else:f[7],f[8]=100,0
                if ahead and f[11] and not sensor.visible(('tls',ahead[0][0],ahead[0][1]),ahead[0][2],1):
                    bits=[0,0,0];f[10],f[11]=999,0
                if f[9]<999:
                    f[9]=max(0,f[9]+sensor.error()) if sensor.visible(('stop',v.getRouteIndex('ego')),f[9],2) else 999
            return f,list(detections.values()),present,bits,line
        state,cars=snapshot()
        reading=observe(raw,cars)
        target=folder/'transitions.csv'
        with target.open('w',newline='',encoding='utf-8') as stream,torch.inference_mode():
            writer=csv.writer(stream)
            writer.writerow(FIELDS)
            for t in range(env.maximum_steps):
                f,det,present,bits,line=reading
                a=int(model(torch.from_numpy(f[None])).argmax().item())
                raw,r,end,cut,info=env.step([a])
                after,ncars=snapshot()
                terminal=end or cut
                nr=None if terminal else observe(raw,ncars)
                nf=np.zeros_like(f) if terminal else nr[0]
                acc,lc=info['requested_accel'],info['lane_change_requested']
                writer.writerow([ep,t,pack(f.tolist()),pack([acc,lc]),acc,lc,lc,info['lane_change_applied'],r[0],end,cut,state,pack(det),info['collision'],info['success'],cars['ego']['lane_id'],ncars.get('ego',{}).get('lane_id'),pack(nf.tolist()),after,weather,CONFIG['obs_profile_id'],present,*bits,line,info['red_violation'],info['stop_violation']])
                if terminal:break
                reading,state,cars=nr,after,ncars
            assert terminal
        if ep==1:
            (output/'lookup.json').write_text(pack(dict(lanes=lanes,signals=signals)))
        return dict(episode=ep,seed=seed,weather=WEATHER[weather],steps=t+1,outcome=info['outcome'],seconds=time.perf_counter()-tm,bytes=target.stat().st_size)
    finally:env.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--weather',choices=['clear','light','dense'],required=True)
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--start-seed',type=int,required=True)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.episodes<1 or args.workers<1:parser.error('episodes and workers must be positive')
    args.out.mkdir(parents=True,exist_ok=False)
    weather=['clear','light','dense'].index(args.weather)
    tasks=[(i+1,args.start_seed+i,weather,args.out) for i in range(args.episodes)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows=list(pool.map(collect,tasks))
    (args.out/'episodes.json').write_text(json.dumps(rows,indent=2))
    print(dict(episodes=len(rows),transitions=sum(r['steps'] for r in rows),outcomes=dict(Counter(r['outcome'] for r in rows))))
