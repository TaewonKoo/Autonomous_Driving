from pathlib import Path
import math,sys,json,os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'implementation'))
from paper_training.observed_driving import ObservedDriving
from paper_training.profiles import Profile
from paper_training.agents import Policy
CK=ROOT.parents[1]/'models/seongsan_20261123/seongsan_ppo_r08c05_seed25.pt'
WEIGHTS=[(0,0),(.125,0),(.25,0),(.5,0),(0,.125),(.125,.125),(.25,.125),(.25,.25)]
class Visual(ObservedDriving):
 def __init__(self,path,seed,traffic,combo):
  super().__init__(path,seed,'full',traffic)
  self.names[186:192]=['signal_readable','red','yellow','green','map_distance','map_signal']
  self.names += [side+'_'+key for side in ('right','left') for key in ('exists','front','front_gap','front_dv','rear','rear_gap','rear_dv')]
  self.obs_dim=len(self.names);self.weights=WEIGHTS[combo]
 def reset(self,*,seed):
  self.radius=30 if seed%2 else 50
  self.last=None;self.stats=Counter()
  return super().reset(seed=seed)
 def observation(self):
  x=super().observation()[0]
  v=self.v; lane=v.getLaneID('ego');road=v.getRoadID('ego');pos=v.getLanePosition('ego');idx=v.getLaneIndex('ego');vel=v.getSpeed('ego')
  tls=v.getNextTLS('ego'); ex=bool(tls);dist=tls[0][2] if ex else 300.
  visible=False
  if ex and 0<=dist<=self.radius:
   conn=self.sim.trafficlight.getControlledLinks(tls[0][0])[tls[0][1]][0][0]
   sx,sy=self.net.getLane(conn).getShape()[-1];px,py=v.getPosition('ego');h=math.radians(v.getAngle('ego'))
   dx,dy=sx-px,sy-py
   visible=dx*math.sin(h)+dy*math.cos(h)>=.5*math.hypot(dx,dy)
  color=tls[0][3].lower() if visible else ''
  x[186:192]=[visible,color=='r',color=='y',color=='g',np.clip(dist/300,0,1),ex]
  x[197]=0 # dynamic priority must not reveal signal phase
  for i in (5,13):
   if x[i]*50>self.radius:x[i:i+4]=[1,0,0,0]
  grid=x[21:186].reshape(5,11,3)
  for col in range(11):
   if max(abs(-90+col*180/11),abs(-90+(col+1)*180/11))**2+20**2>self.radius**2:grid[:,col,:]=0
  self.cross=bool(np.any((grid[:,:,2]>0)&(abs(grid[:,:,0])>.25)&(abs(grid[:,:,0])<.75)))
  self.near={};sideobs=[]
  for direction in (-1,1):
   exists=not road.startswith(':') and 0<=idx+direction<self.net.getEdge(road).getLaneNumber()
   sideobs.append(float(exists));cars=[]
   if exists:
    other=self.net.getEdge(road).getLanes()[idx+direction].getID()
    cars=[(v.getLanePosition(k)-pos,k) for k in self.sim.lane.getLastStepVehicleIDs(other)]
   for sign in (1,-1):
    opp=min(((d*sign,k) for d,k in cars if d*sign>=0),default=None)
    self.near[direction,sign]=opp
    if opp and opp[0]<=self.radius:
     sideobs.extend([1,opp[0]/self.radius,np.clip((v.getSpeed(opp[1])-vel)/14,-1,1)])
    else:sideobs.extend([0,0,0])
  return np.concatenate((x,sideobs)).astype(np.float32)[None,:]
 def step(self,a):
  accel,d=self.decode(a);v=self.v;road=v.getRoadID('ego');idx=v.getLaneIndex('ego')
  risk=False
  rear=self.near.get((d,-1));front=self.near.get((d,1))
  self.stats['opportunity']+=int(any(self.near[k] and self.near[k][0]<self.radius for k in ((1,-1),(-1,-1))))
  self.stats['cross']+=int(self.cross);self.stats['steps']+=1
  if rear:
   gap,k=rear;speed=v.getSpeed(k);leader=v.getLeader(k,150)
   def pred(leader,gap):
    return (v.getFollowSpeed(k,speed,max(.01,gap),v.getSpeed(leader),v.getDecel(leader),leader)-speed)/self.s.dt
   before=pred(*leader) if leader and leader[0] else 0.
   after=pred('ego',gap-v.getLength('ego')-v.getMinGap(k));risk=after < -4 and after<before
  needs=not self.lane_reaches_next_route_edge(v.getLaneID('ego')) if not road.startswith(':') else False
  leader=v.getLeader('ego',50)
  out,r,end,tr,info=super().step(a)
  if info['lane_change_applied']:
   old=self.last
   # Conservative proxy: no front vehicle before either maneuver, both route-valid, no intervening close lead.
   empty=not(leader and leader[0]) and not needs
   w=bool(old and old[0]==road and old[1]==idx+d and self.steps-old[2]<=15 and old[3] and empty and not risk)
   self.last=(road,idx,self.steps,empty and not risk)
   self.stats['changes']+=1;self.stats['U']+=int(risk);self.stats['W']+=int(w)
   r[0]-=self.weights[0]*risk+self.weights[1]*w
  elif self.last and (road!=self.last[0] or needs or (leader and leader[0])):
   self.last=None
  return out,r,end,tr,info
