"""Compare BC decisions under weather-specific observations of a shared trajectory."""
import csv, hashlib, json, math, sys

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parents[1] / 'bc'
sys.path.insert(0,str(STUDY))
from learned_driver import Visual, OUT, features, lane_plan, torch

from collect import Sensor

def collect(task):
    seed,record=task
    folder=ROOT/'runs'/str(seed)
    folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    lock=json.loads((OUT/'v2_selection.json').read_text())['selected']['student_opt_bc']
    ck=OUT/'student_opt_bc.pt'
    assert hashlib.sha256(ck.read_bytes()).hexdigest()==lock['sha256']
    model=torch.jit.load(str(ck),map_location='cpu').eval()
    env=Visual(folder,seed,4,0)
    plan=lane_plan(env)
    geometry=json.loads((OUT/'intersection_geometry.json').read_text())
    sensors=[Sensor(i,seed) for i in range(3)]
    try:
        raw=env.reset(seed=seed)
        v,sim=env.v,env.sim
        ids={'ego':0}
        def observe(raw,cars,weather,sensor):
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
            lead=v.getLeader('ego',50)
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
                    f[10],f[11]=999,0
                if f[9]<999:
                    f[9]=max(0,f[9]+sensor.error()) if sensor.visible(('stop',v.getRouteIndex('ego')),f[9],2) else 999
            return f,detections
        steps=[]
        trace=[]
        with torch.inference_mode():
            for t in range(env.maximum_steps):
                cars={}
                for k in sorted(v.getIDList()):
                    x,y=v.getPosition(k)
                    ids.setdefault(k,len(ids))
                    cars[k]=dict(x_m=x,y_m=y,speed_mps=v.getSpeed(k),heading_deg=v.getAngle(k))
                views=[observe(raw,cars,i,sensor) for i,sensor in enumerate(sensors)]
                clear=views[0][0]
                lead=v.getLeader('ego',50)
                lid=lead[0] if lead and lead[0] and clear[7]<100 else None
                candidate=bool(lid and lid in views[0][1])
                batch=[item[0] for item in views]
                for f,_ in views[1:]:
                    recovered=f.copy();recovered[7:9]=clear[7:9]
                    batch.append(recovered)
                actions=model(torch.from_numpy(np.stack(batch))).argmax(-1).tolist()
                steps.append([t,*actions,int(candidate),int(bool(lid in views[1][1])),int(bool(lid in views[2][1])),float(clear[7]),float(clear[0]),float(clear[8])])
                if record:
                    trace.append(dict(t=t,lead=lid,ego=cars['ego'],cars=cars,seen=[list(view[1]) for view in views],actions=actions,gap=float(clear[7]),road=v.getRoadID('ego')))
                raw,r,end,cut,info=env.step([actions[0]])
                if end or cut:break
            assert end or cut
        arr=np.array(steps)
        if record:
            (ROOT/'trace.json').write_text(json.dumps(trace))
        else:np.savez_compressed(ROOT/'frames'/f'{seed}.npz',frames=arr)
        return seed
    finally:env.close()


if __name__=='__main__':
    if len(sys.argv)>1:
        print(collect((int(sys.argv[1]),True)))
    else:
        with (ROOT/'episode_counts.csv').open() as source:
            seeds=sorted({int(row['seed']) for row in csv.DictReader(source)})
        (ROOT/'frames').mkdir(exist_ok=True)
        with ProcessPoolExecutor(10) as pool:
            for i,_ in enumerate(pool.map(collect,[(s,False) for s in seeds])):
                if (i+1)%25==0:print(i+1,flush=True)
