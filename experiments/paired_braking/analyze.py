from pathlib import Path
import numpy as np,json,csv
D=Path(__file__).resolve().parent
files=sorted((D/'frames').glob('*.npz')); assert len(files)==100
by=[]; events=[]
for p in files:
 a=np.load(p)['frames'];seed=int(p.stem);clear=a[:,1]<3
 for j,w in enumerate(['light','dense']):
    brake=a[:,2+j]<3;fixed=a[:,4+j]<3
    eligible=a[:,6].astype(bool);miss=eligible & (a[:,7+j]==0)
    drop=clear & ~brake
    recover=drop & miss & fixed
    by.append(dict(seed=seed,weather=w,n=len(a),clear_brake=int(clear.sum()),drop=int(drop.sum()),extra=int((~clear & brake).sum()),nonbrake=int((~clear).sum()),eligible=int(eligible.sum()),leader_missed=int(miss.sum()),drop_missed=int((drop & miss).sum()),missed_restored=int(recover.sum())))
    if w=='dense':
        for t in np.flatnonzero(recover):
            if 20<=t<len(a)-15 and a[t,10]>a[t,11]+.5 and a[t,10]>1:
                events.append(dict(seed=seed,t=int(t),gap=a[t,9],speed=a[t,10],leader_speed=a[t,11]))
rng=np.random.default_rng(2711);samp=rng.integers(0,100,(5000,100));summary={}
for w in ['light','dense']:
 g=[r for r in by if r['weather']==w]
 s={k:sum(r[k] for r in g) for k in g[0] if k not in ('seed','weather')}
 for label,num,den in [('drop_rate','drop','clear_brake'),('extra_rate','extra','nonbrake'),('restore_rate','missed_restored','drop_missed')]:
    v=np.array([r[num] for r in g])[samp].sum(axis=1);n=np.array([r[den] for r in g])[samp].sum(axis=1)
    s[label]=s[num]/s[den];s[label+'_ci']=np.quantile(v[n>0]/n[n>0],[.025,.975]).tolist()
 summary[w]=s
assert events
med=np.median([e['gap'] for e in events]); chosen=min(events,key=lambda e:(abs(e['gap']-med),e['seed'],e['t']))
summary['example']=chosen;summary['eligible_example_frames']=len(events)
(D/'summary.json').write_text(json.dumps(summary,indent=2))
with (D/'episode_counts.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(by[0]));w.writeheader();w.writerows(by)
print(json.dumps(summary))
