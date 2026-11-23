"""Evaluate the published Seongsan checkpoints with their original environment.

The wrapper partitions explicit case seeds and merges episode records. Observation,
action, reward and simulation code are imported unchanged from implementation/.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
KST = timezone(timedelta(hours=9), name='Asia/Seoul')
FILES = {
    'r08c05': 'seongsan_ppo_r08c05_seed25.pt',
    'r06c03': 'seongsan_ppo_r06c03_seed24.pt',
    'r08c01': 'seongsan_ppo_r08c01_seed24.pt',
}
OUTCOMES = ('complete', 'collision', 'red_violation', 'stop_violation', 'route_failure', 'timeout')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def package_protocol(model):
    manifest = read(ROOT/'SETTINGS_MANIFEST.json')
    entries = {x['path']: x for x in manifest['files']}
    required = [FILES[model], 'implementation/train_observed_driving.py',
                'implementation/paper_training/observed_driving.py',
                'implementation/paper_training/agents.py', 'implementation/paper_training/profiles.py',
                'implementation/paper_training/candidate_settings.py',
                'implementation/paper_training/environments.py', 'implementation/utils/networks.py',
                'implementation/maps/seongsan/generated/seongsan.net.xml',
                'implementation/maps/seongsan/generated/seongsan.rou.xml']
    hashes = {}
    for rel in required:
        path = ROOT/rel
        if not path.is_file():
            raise FileNotFoundError(f'Missing {rel}. Download this complete branch/folder, including implementation/maps/.')
        digest = sha(path)
        if digest != entries[rel]['sha256']:
            raise ValueError(f'File differs from the published evaluation bundle: {rel}')
        hashes[rel] = digest
    return {'model': model, 'checkpoint_file': FILES[model], 'checkpoint_sha256': hashes[FILES[model]],
            'stage': 'full', 'traffic': 1.0, 'dt_seconds': 0.2, 'max_episode_seconds': 500,
            'deterministic_argmax': True, 'policy_only': True, 'speed_mode': 96, 'lane_change_mode': 0,
            'numeric_correction': False, 'failure_time_tail': False, 'source_hashes': hashes}

def runtime():
    try:
        import numpy as np
        import torch
        import traci
        import sumolib
        import libsumo
    except ImportError as error:
        raise RuntimeError('Missing evaluation dependencies. Run: python -m pip install -r requirements-evaluation.txt') from error
    torch.set_num_threads(1)
    sumo_version = libsumo.getVersion()[1]
    if not re.search(r'(?<!\d)1[._]27[._]1(?!\d)', sumo_version):
        raise RuntimeError(f'Expected SUMO1.27.1; got {sumo_version}. Install requirements-evaluation.txt.')
    if torch.__version__.split('+')[0] != '2.14.1' or np.__version__ != '2.4.6':
        raise RuntimeError(f'Expected torch2.14.1 and numpy2.4.6, got {torch.__version__}, {np.__version__}. Install pinned requirements.')
    return {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
            'sumo': sumo_version, 'device': 'cpu', 'torch_threads': 1}

def evaluate_chunk(task):
    model, start, count, directory, trace = task
    environment = runtime()
    sys.path.insert(0, str(ROOT/'implementation'))
    from train_observed_driving import evaluate
    result = evaluate(ROOT/FILES[model], directory, stage='full', traffic=1.0,
                      episodes=count, trace=trace, policy_only=True, seed_base=start,
                      numeric_correction=False)
    return {'report': str(Path(directory)/'evaluation.json'), 'runtime': environment,
            'episodes': result['episodes']}

def summarize(episodes):
    counts = Counter(x['outcome'] for x in episodes)
    if set(counts)-set(OUTCOMES):
        raise ValueError(f'Unknown outcomes: {set(counts)-set(OUTCOMES)}')
    n = len(episodes)
    return {'episodes': n, 'outcomes': {k: counts[k] for k in OUTCOMES},
            'rates_percent': {k: 100*counts[k]/n for k in OUTCOMES},
            'collision_strictly_below_5_percent': 100*counts['collision'] < 5*n}

def save_result(output, protocol, episodes, seed_start, expected_count, shards):
    import csv
    episodes = sorted(episodes, key=lambda x: x['seed'])
    seeds = [x['seed'] for x in episodes]
    expected = list(range(seed_start, seed_start+expected_count))
    if seeds != expected:
        raise ValueError('Seeds must cover the requested range exactly once; duplicate, missing or extra cases found.')
    if any(x['controller'] != 'policy' for x in episodes):
        raise ValueError('Only policy episode records may be merged.')
    report = {'completed_at_kst': datetime.now(KST).isoformat(timespec='seconds'),
              'protocol': protocol, 'evaluation_seed_base': seed_start,
              'summary': {'policy': summarize(episodes)}, 'episodes': episodes, 'shards': shards}
    write(output/'evaluation.json', report)
    with (output/'episodes.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
        writer.writeheader();writer.writerows(episodes)
    print(json.dumps({'output': str(output), 'summary': report['summary']}, ensure_ascii=False), flush=True)
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=FILES, default='r08c05')
    parser.add_argument('--seed-start', type=int, default=800000)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--trace', action='store_true', help='Save per-step CSV/NPZ; larger files, unchanged policy.')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--merge', nargs='+', type=Path, help='Merge evaluation.json files from independent, nonoverlapping shards.')
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1 or args.seed_start < 0:
        parser.error('episodes/workers must be positive; seed-start must be nonnegative')
    protocol = package_protocol(args.model)
    if args.check_only:
        print(json.dumps({'protocol': protocol, 'runtime': runtime()}, ensure_ascii=False, indent=2))
        return
    output = (args.output or Path('evaluation_results')/
              (args.model+'_'+datetime.now(KST).strftime('%Y%m%d_%H%M%S'))).resolve()
    if (output/'evaluation.json').exists() or (output/'evaluation_plan.json').exists():
        raise FileExistsError(f'Output already has a recorded evaluation: {output}. Choose a new --output.')
    output.mkdir(parents=True, exist_ok=True)
    if args.merge:
        episodes=[];shards=[]
        for source in args.merge:
            data=read(source)
            if data['protocol'] != protocol:
                raise ValueError(f'Model, source files or protocol differ: {source}')
            episodes.extend(data['episodes'])
            shards.append({'report': str(source.resolve()), 'sha256': sha(source), 'shards': data['shards']})
        save_result(output, protocol, episodes, args.seed_start, args.episodes, shards)
        return
    parent_runtime = runtime()
    workers = min(args.workers, args.episodes)
    base, remainder = divmod(args.episodes, workers)
    start=args.seed_start;tasks=[]
    for worker in range(workers):
        count=base+(worker<remainder)
        tasks.append((args.model,start,count,str(output/f'worker_{worker+1:02d}'),args.trace))
        start+=count
    write(output/'evaluation_plan.json',{'protocol':protocol,'runtime':parent_runtime,
          'seed_start':args.seed_start,'episodes':args.episodes,'workers':workers,'tasks':tasks,
          'scope':'Seed partitioning only; calls the original full-route evaluator without changing its observations/actions/reward.'})
    episodes=[];shards=[]
    if workers==1:
        results=[evaluate_chunk(tasks[0])]
    else:
        results=[]
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            pending=[pool.submit(evaluate_chunk,t) for t in tasks]
            for future in as_completed(pending):
                result=future.result();results.append(result)
                print(f"Completed shard: {len(result['episodes'])} episodes",flush=True)
    for result in results:
        episodes.extend(result['episodes'])
        shards.append({'report':result['report'],'runtime':result['runtime']})
    save_result(output,protocol,episodes,args.seed_start,args.episodes,shards)

if __name__=='__main__':
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as error:
        print(f'Evaluation failed: {error}',file=sys.stderr,flush=True)
        raise
