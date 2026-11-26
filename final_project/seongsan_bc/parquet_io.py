"""Convert one complete episode CSV to a typed Parquet shard and restore next values."""
import argparse
import csv
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

integer, real, boolean = pa.int32(), pa.float32(), pa.bool_()
def record(names, types):
    return pa.struct(list(zip(names.split(),types)))
vehicles = record('vehicle_id lane_id x_m y_m speed_mps heading_deg',[integer]*2+[real]*4)
signals = record('tls_id link_index state_gt',[integer]*3)
state = record('sim_time_s ego_id vehicles signals',[real,integer,pa.list_(vehicles),pa.list_(signals)])
tracks = record('track_id rel_x_m rel_y_m observed_speed_mps',[integer,real,real,real])
types = dict(observation=pa.list_(real), next_observation=pa.list_(real), action_raw=pa.list_(real),
             state=state,next_state=state,detected_vehicles=pa.list_(tracks))
for name in 'action.accel_raw reward stopline_dist_m'.split(): types[name]=real
for name in 'terminated truncated action.lane_change_applied event.collided event.arrived red_violation stop_violation'.split(): types[name]=boolean

def convert(source,target):
    if target.exists(): raise FileExistsError(target)
    rows=[]
    with source.open(newline='') as f:
        reader=csv.DictReader(f)
        schema=pa.schema([(k,types.get(k,integer)) for k in reader.fieldnames])
        for raw in reader:
            row={}
            for k,v in raw.items():
                typ=schema.field(k).type
                if not v: row[k]=None
                elif pa.types.is_nested(typ): row[k]=json.loads(v)
                elif typ==boolean: row[k]={'True':True,'False':False}[v]
                else: row[k]=float(v) if typ==real else int(v)
            assert row['t']==len(rows) and len(row['observation'])==len(row['next_observation'])==17
            if rows:
                prev=rows[-1]
                assert prev['episode']==row['episode'] and not (prev['terminated'] or prev['truncated'])
                assert prev['next_state']==row['state'] and prev['next_observation']==row['observation']
                prev['next_state']=prev['next_observation']=None
            rows.append(row)
    assert rows and (rows[-1]['terminated'] or rows[-1]['truncated'])
    pq.write_table(pa.Table.from_pylist(rows,schema=schema),target,compression='zstd')

def read_episode(path):
    rows=pq.read_table(path).to_pylist()
    for i,row in enumerate(rows[:-1]):
        row['next_state']=rows[i+1]['state']
        row['next_observation']=rows[i+1]['observation']
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('csv',type=Path)
    parser.add_argument('parquet',type=Path)
    args=parser.parse_args()
    convert(args.csv,args.parquet)
    print(len(read_episode(args.parquet)), 'transitions')
