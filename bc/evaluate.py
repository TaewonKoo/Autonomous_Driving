"""Evaluate each weather with the same collector and independent simulator instances."""
import argparse
import json
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collect import collect, WEATHER

def evaluate(task):
    weather,seed=task
    with tempfile.TemporaryDirectory() as tmp:
        row=collect((1,seed,weather,Path(tmp)))
    return {key:row[key] for key in ('seed','weather','steps','outcome')}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--start-seed',type=int,default=83000000)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if min(a.episodes,a.workers)<1:p.error('positive episodes and workers required')
    with a.out.open('x') as f, ProcessPoolExecutor(a.workers) as pool:
        counts={k:Counter() for k in WEATHER}
        for row in pool.map(evaluate,((w,s) for s in range(a.start_seed,a.start_seed+a.episodes) for w in range(3))):
            print(json.dumps(row),file=f,flush=True)
            counts[row['weather']][row['outcome']]+=1
    print(counts)
