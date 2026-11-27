"""Smooth editable road points while keeping junctions and lane tapers fixed."""
import math


def _unit(vector):
    length = math.hypot(*vector)
    return tuple(x / length for x in vector) if length else (0.0, 0.0)


def smooth_streets(points, streets, scale, tapers, protected=(), already_curved=(), handle_m=12, step_m=.8):
    paths = {eid: [points[source], *mids, points[target]] for eid, source, target, *_rest, mids in streets}
    incident = {}
    for eid, source, target, *_ in streets:
        path = paths[eid]
        incident.setdefault(source, []).append((eid, _unit((path[1][0]-path[0][0], path[1][1]-path[0][1]))))
        incident.setdefault(target, []).append((eid, _unit((path[-2][0]-path[-1][0], path[-2][1]-path[-1][1]))))

    def endpoint(node, eid, outward):
        if node in tapers:
            dx, dy = tapers[node]['direction']
            axis = _unit((dx, -dy))
            return axis if sum(a*b for a,b in zip(axis,outward)) >= 0 else tuple(-x for x in axis)
        neighbors = incident[node]
        if len(neighbors) == 2:
            other = next(v for road,v in neighbors if road != eid)
            axis = _unit(tuple(a-b for a,b in zip(outward,other)))
            if math.hypot(*axis):
                return axis
        return outward

    result = []
    for eid, source, target, lanes, speed, mids in streets:
        if eid in already_curved:
            result.append((eid,source,target,lanes,speed,mids))
            continue
        path = paths[eid]
        directions = [_unit((b[0]-a[0], b[1]-a[1])) for a,b in zip(path,path[1:])]
        lengths = [math.dist(a,b) for a,b in zip(path,path[1:])]
        tangents = [endpoint(source,eid,directions[0])]
        for before, after in zip(directions,directions[1:]):
            tangent = _unit(tuple(a+b for a,b in zip(before,after)))
            tangents.append(tangent if math.hypot(*tangent) else after)
        outward = tuple(-x for x in directions[-1])
        tangents.append(tuple(-x for x in endpoint(target,eid,outward)))
        limit = (min(handle_m,3) if eid in protected else handle_m) / scale
        smooth = [path[0]]
        for i,(a,b) in enumerate(zip(path,path[1:])):
            distance = lengths[i]
            h1 = min(distance/3, lengths[max(0,i-1)]*.4, limit)
            h2 = min(distance/3, lengths[min(len(lengths)-1,i+1)]*.4, limit)
            c1 = tuple(a[j]+tangents[i][j]*h1 for j in (0,1))
            c2 = tuple(b[j]-tangents[i+1][j]*h2 for j in (0,1))
            samples = max(2, math.ceil(distance*scale/step_m))
            for k in range(1,samples+1):
                t=k/samples
                smooth.append(tuple((1-t)**3*a[j]+3*(1-t)**2*t*c1[j]+3*(1-t)*t*t*c2[j]+t**3*b[j] for j in (0,1)))
        result.append((eid,source,target,lanes,speed,smooth[1:-1]))
    return result
