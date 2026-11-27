"""Static intersection geometry for the route. No signals, live routes or priorities."""
import json
from collections import defaultdict
from pathlib import Path
from shapely.geometry import LineString
import numpy as np


def connection_path(net, connection):
    lanes = []
    next_id = connection.getViaLaneID()
    target = connection.getToLane().getID()
    while next_id and next_id not in lanes:
        lanes.append(next_id)
        links = net.getLane(next_id).getOutgoing()
        link = next((x for x in links if x.getToLane().getID()==target),None)
        next_id = link.getViaLaneID() if link else ""
    return lanes


def build(net, route):
    geometry = {}
    def line(lane):
        if lane not in geometry:
            geometry[lane]=LineString(net.getLane(lane).getShape())
        return geometry[lane]
    static=defaultdict(lambda:defaultdict(list))
    for index, (road, dest) in enumerate(zip(route,route[1:])):
        origin = net.getEdge(road)
        others=[lane for edge in origin.getToNode().getIncoming() if edge!=origin
                and not edge.getID().startswith(":") for lane in edge.getLanes()]
        for lane in origin.getLanes():
            for conn in lane.getOutgoing():
                if conn.getTo().getID()!=dest:
                    continue
                ego=connection_path(net, conn)
                for other in others:
                    for alt in other.getOutgoing():
                        path=connection_path(net,alt)
                        for ei, el in enumerate(ego):
                            for hi, hl in enumerate(path):
                                if el==hl:
                                    continue
                                a,b=line(el),line(hl)
                                zone=a.intersection(b.buffer(2.1))
                                if zone.is_empty:
                                    continue
                                center=zone.centroid
                                sa=a.project(center,normalized=True)*net.getLane(el).getLength()
                                sb=b.project(center,normalized=True)*net.getLane(hl).getLength()
                                ep=ego[:ei+1]
                                hp=path[:hi+1]
                                def offsets(initial, partial, end_distance):
                                    offsets={partial[-1]:end_distance}
                                    for j in range(len(partial)-2,-1,-1):
                                        offsets[partial[j]]=net.getLane(partial[j]).getLength()+offsets[partial[j+1]]
                                    offsets[initial.getID()]=initial.getLength()+offsets[partial[0]]
                                    return offsets
                                eo=offsets(lane,ep,sa)
                                ho=offsets(other,hp,sb)
                                if other.getLength()<30:
                                    for prev in other.getIncoming():
                                        ho[prev.getID()]=prev.getLength()+ho[other.getID()]
                                for e,ed in eo.items():
                                    for h,hd in ho.items():
                                        key=f"{index}/{e}"
                                        pair=(round(ed,1), round(hd,1))
                                        if pair not in static[key][h]:
                                            static[key][h].append(pair)
    return static


def conflict_features(env, static, speed):
    v=env.v
    lane=v.getLaneID("ego")
    idx=v.getRouteIndex("ego")
    other_lanes=static.get(f"{idx}/{lane}",{})
    pos=v.getLanePosition("ego")
    x,y=v.getPosition("ego")
    conflicts=[]
    for hl,pairs in other_lanes.items():
        for hv in env.sim.lane.getLastStepVehicleIDs(hl):
            if hv=="ego":
                continue
            hx,hy=v.getPosition(hv)
            if (hx-x)**2+(hy-y)**2>env.radius**2:
                continue
            hv_speed=v.getSpeed(hv)
            hp=v.getLanePosition(hv)
            for ed,hd in pairs:
                de,dh=ed-pos,hd-hp
                if -1<de<40 and -6<dh<55:
                    conflicts.append((de,dh,hv_speed))
    return conflicts
