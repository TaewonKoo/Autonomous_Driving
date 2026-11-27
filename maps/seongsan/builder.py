"""Create a reproducible SUMO network and its traffic from config.py."""
import heapq
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from sumolib import checkBinary
from sumolib.geomhelper import move2side
from sumolib.net import readNet
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring
from shapely import union_all

from . import config as C
from .speed_policy import remove_road_speed_limits
from .markings import asphalt_faces, boundary_lines, lane_dividers, line_parts, mask_faces, outside_strip, painted_stroke, pavement, small_islands, validate_lane_footprints


def xy(point):
    return point[0] * C.SCALE, (929 - point[1]) * C.SCALE


def write_xml(path, root):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def school_edges():
    return {eid + suffix for eid in C.SCHOOL_ZONE["edges"] for suffix in ("", "_rev")}


def island_markings(points=None, ratios=(1, .75, .5, .25)):
    """사진의 작은 노란 삼각형 빗금과 빨간 시선유도봉 배치."""
    points = C.TRAFFIC_ISLAND if points is None else points
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    outlines = []
    for ratio in ratios:
        outline = [(cx + (x-cx)*ratio, cy + (y-cy)*ratio) for x,y in points]
        outlines.append(outline + outline[:1])
    bollards = []
    for a,b in zip(points, points[1:] + points[:1]):
        count = max(1, math.ceil(math.dist(a,b)/7))
        for i in range(count):
            t = i/count
            bollards.append((a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t))
    return outlines, bollards


def road_surface(net):
    centers=[xy(C.RING_CENTER),xy(C.EAST_ENTRY_REFUGE_SEED)]
    for points in [C.TRAFFIC_ISLAND,C.LOWER_TRAFFIC_ISLAND]:
        centers.append(xy(tuple(sum(p[j] for p in points)/len(points) for j in (0,1))))
    entry_region=Polygon([xy(C.POINTS[node]) for node in
                          ['triangle_northeast','east_slip_split','east_cross']]).buffer(5)
    surface=pavement(net,centers,omit_node_shapes={'east_cross','east_slip_split'},
                     preserve_holes_in=[entry_region])
    # Round the local fork/curb notches by adding pavement, never clipping a lane.
    rounded=surface.buffer(.6,quad_segs=8).buffer(-.6,quad_segs=8)
    # The outer curb is more than 5m from the fork's node center. Include it
    # fully: mismatched flat lane caps otherwise leave a thin inward slit.
    fork_region=Point(xy(C.POINTS['east_slip_split'])).buffer(9)
    rounding_region=entry_region.union(fork_region)
    return surface.union(rounded.intersection(rounding_region))


def entry_markings(net, surface):
    """Yellow exclusion-area paint follows the compiled island footprint."""
    anchor = xy(C.EAST_ENTRY_REFUGE_SEED)
    candidates = [Polygon(shape) for shape in small_islands(surface)
                  if Polygon(shape).centroid.distance(Point(anchor)) < 30]
    if not candidates:
        raise ValueError('Eastern entry has no separated pedestrian refuge')
    island = min(candidates,key=lambda p:p.centroid.distance(Point(anchor)))
    inset = island.buffer(-.55,join_style='round').simplify(.08,preserve_topology=True)
    if inset.is_empty or inset.geom_type != 'Polygon':
        raise ValueError('Eastern entry island is too narrow for its yellow markings')
    yield 'entry_yellow_border',list(inset.exterior.coords),'1,0.8,0.1',False,.15
    # Short diagonal stripes inside the refuge, never across a vehicle lane.
    xmin,ymin,xmax,ymax = inset.bounds
    for i in range(math.ceil((ymax-ymin+xmax-xmin)/2.5)+1):
        intercept = ymin-xmax+i*2.5
        cut = LineString([(xmin-2,xmin-2+intercept),(xmax+2,xmax+2+intercept)]).intersection(inset.buffer(-.2))
        for j,shape in enumerate(line_parts(cut)):
            yield f'entry_hatch_{i}_{j}',shape,'1,0.8,0.1',False,.15


def entry_stop_markings(net, surface=None):
    """One perpendicular bar per stopped approach; a dashed yield line on the slip."""
    for edge_id in C.ROAD_STOP_OFFSETS:
        edge = net.getEdge(edge_id)
        lane = edge.getLanes()[-1]
        a,b = lane.getShape()[-2:]
        length = math.dist(a,b)
        ux,uy = (b[0]-a[0])/length,(b[1]-a[1])/length
        nx,ny = -uy,ux
        width = sum(l.getWidth() for l in edge.getLanes())
        # Innermost lane's left edge is the median; paint up to the right curb.
        median = (b[0]+nx*lane.getWidth()/2,b[1]+ny*lane.getWidth()/2)
        def rectangle(start,finish,depth,along=0):
            return [(median[0]+nx*side+ux*(along+longitudinal),
                     median[1]+ny*side+uy*(along+longitudinal))
                    for side,longitudinal in [(start,-depth/2),(finish,-depth/2),
                                              (finish,depth/2),(start,depth/2)]]
        # SUMO draws link-rule bars above its ordinary lane markings (z > 200).
        # Keep this local cover above those decals and below vehicles (z = 340).
        yield f'entry_bar_mask_{edge_id}',rectangle(-width-.06,.06,1.2),'0,0,0','210'
        yield f'entry_bar_curb_{edge_id}',rectangle(-width-.06,-width+.06,1.2),'1,1,1','211'
        median_back=-.6 if edge_id=='east_signal_approach_rev' else 0
        yield f'entry_bar_median_{edge_id}',rectangle(-.075,.075,1.2,along=median_back),'1,0.8,0.1','211'
        if edge_id == 'east_right_slip':
            # Preserve its yield rule instead of painting another mandatory stop.
            for i in range(3):
                start = -width+(i+.15)*width/3
                finish = -width+(i+.85)*width/3
                yield f'entry_yield_dash_{i}',rectangle(start,finish,.35),'1,1,1','211'
        else:
            yield f'entry_stop_bar_{edge_id}',rectangle(-width+.03,-.03,.35),'1,1,1','211'
    # The unregulated fork has no stopping requirement. Cover SUMO's leftover
    # link bars, then restore the actual curb, median, and any dashed lane line.
    fork_masks=[]
    # Priority-road lanes whose left turn yields ('M' and 'm' links) get a partial native bar,
    # which reads as a broken stop line. Priority roads carry no stop line, so hide those too.
    mixed=[]
    for edge in net.getEdges(withInternal=False):
        if edge.getToNode().getType() not in ('priority',):
            continue
        for lane in edge.getLanes():
            states={c.getState() for c in lane.getOutgoing()}
            if 'M' in states and states-{'M'}:
                mixed.append((lane,False))
    fork_lanes=[(lane,at_start) for edge_id,at_start in [('east_turn_pocket_rev',False),('east_signal_approach',False),
                             ('east_right_slip',True),('east_signal_approach_rev',True),('east_turn_pocket',True)]
                for lane in net.getEdge(edge_id).getLanes()]
    for lane,at_start in [*fork_lanes,*mixed]:
        if True:
            a,b=tuple(reversed(lane.getShape()[:2])) if at_start else lane.getShape()[-2:]
            distance=math.dist(a,b)
            ux,uy=(b[0]-a[0])/distance,(b[1]-a[1])/distance
            half=lane.getWidth()/2+.1
            fork_masks.append(Polygon([(b[0]+ux*along-uy*side,b[1]+uy*along+ux*side)
                                      for along,side in [(-.7,-half),(.7,-half),(.7,half),(-.7,half)]]))
    mask=union_all(fork_masks)
    surface=road_surface(net) if surface is None else surface
    for label,geometry,color in [('pavement',mask.intersection(surface),'0,0,0'),
                                 ('outside',mask.difference(surface),'51,128,51')]:
        for i,shape in enumerate(asphalt_faces(geometry)):
            yield f'fork_bar_cover_{label}_{i}',shape,color,'210'
    white=[*boundary_lines(net,surface),*lane_dividers(net)]
    yellow=[(name,shape,.15) for name,shape in solid_centerlines(net)]
    for color,lines in [('1,1,1',white),('1,0.8,0.1',yellow)]:
        for name,shape,width in lines:
            stroke=LineString(shape).buffer(width/2,cap_style='flat',join_style='round').intersection(mask.buffer(.01))
            if stroke.is_empty:
                continue
            for i,points in enumerate(asphalt_faces(stroke)):
                yield f'fork_restore_{name}_{i}',points,color,'211'



def align_lane_ends(out, edges, tolerance=0.05):
    """Pull protruding lanes of C.LANE_END_ALIGN edges back (lane endOffset) onto one stop line."""
    targets = set(getattr(C, "LANE_END_ALIGN", ()))
    if not targets:
        return False
    net = readNet(str(out / "seongsan.net.xml"))
    changed = False
    for element in edges.findall("edge"):
        edge_id = element.get("id")
        if edge_id not in targets:
            continue
        lanes = net.getEdge(edge_id).getLanes()
        a, b = lanes[-1].getShape()[-2:]
        length = math.dist(a, b)
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        ends = [lane.getShape()[-1] for lane in lanes]
        projection = [x * ux + y * uy for x, y in ends]
        base = min(projection)
        current = {int(l.get("index")): float(l.get("endOffset")) for l in element.findall("lane")}
        edge_offset = float(element.get("endOffset", 0))
        for index, value in enumerate(projection):
            delta = value - base
            if delta > tolerance:
                lane = next((l for l in element.findall("lane") if int(l.get("index")) == index), None)
                if lane is None:
                    lane = ET.SubElement(element, "lane", index=str(index))
                lane.set("endOffset", f"{current.get(index, edge_offset) + delta:.3f}")
                changed = True
    return changed


def circle_shape(center, radius, start=0, finish=2*math.pi, samples=128):
    return [(center[0]+radius*math.cos(start+(finish-start)*i/samples),
             center[1]+radius*math.sin(start+(finish-start)*i/samples)) for i in range(samples+1)]


def custom_arrow_lanes(net):
    result = {}
    for edge_id, style in C.CUSTOM_ARROW_EDGES.items():
        for lane in net.getEdge(edge_id).getLanes():
            result[lane.getID()] = style
    result.update(C.CUSTOM_LANE_ARROWS)
    return result


def custom_arrows(net):
    for lane_id,style in custom_arrow_lanes(net).items():
        shape=net.getLane(lane_id).getShape()
        total=sum(math.dist(a,b) for a,b in zip(shape,shape[1:]))
        remaining=total/2 if lane_id.startswith('ring_') or total<10 else max(1,min(total-7,total*.7))
        for a,b in zip(shape,shape[1:]):
            length=math.dist(a,b)
            if length and remaining<=length:
                dx,dy=(b[0]-a[0])/length,(b[1]-a[1])/length
                x,y=a[0]+dx*remaining,a[1]+dy*remaining
                break
            remaining-=length
        def point(along,side):
            return x+along*dx+side*dy,y+along*dy-side*dx
        if style=='straight':
            stem=[point(-1.4,0),point(.3,0)]
            head=[point(1.4,0),point(.25,-.45),point(.25,.45)]
        else:
            side=1 if style=='right' else -1
            stem=[point(-1.3,0),point(.2,0),point(.2,.8*side)]
            head=[point(.2,1.35*side),point(-.2,.65*side),point(.6,.65*side)]
        yield 'arrow_stem_'+lane_id,stem,False,.13
        yield 'arrow_head_'+lane_id,head+head[:1],True,.1


def connected_path(start, target):
    """Dijkstra on SUMO's compiled lane connections, including turn rules."""
    queue = [(0.0, start.getID(), [start.getID()])]
    seen = set()
    edges = {start.getID(): start}
    while queue:
        distance, eid, path = heapq.heappop(queue)
        if eid in seen:
            continue
        seen.add(eid)
        edge = edges[eid]
        if eid == target.getID():
            return path
        for successor in edge.getOutgoing():
            sid = successor.getID()
            edges[sid] = successor
            if sid not in seen:
                heapq.heappush(queue, (distance + successor.getLength(), sid, path + [sid]))
    raise ValueError(f"No connected route: {start.getID()} -> {target.getID()}")


def build(out_dir=None, laps=1, traffic=True, include_ego=True, route_name=C.DEFAULT_EGO_ROUTE):
    if laps < 1:
        raise ValueError("laps must be >= 1")
    out = Path(out_dir or Path(__file__).parent / "generated").resolve()
    out.mkdir(parents=True, exist_ok=True)
    nodes = ET.Element("nodes")
    signal_nodes = {node: controller for controller, members in C.SIGNAL_CONTROLLERS.items() for node in members}
    road_nodes = {node for street in C.STREETS for node in street[1:3]} | set(C.RING_NODES)
    for name, point in C.POINTS.items():
        if name not in road_nodes:
            continue
        x, y = xy(point)
        node = ET.SubElement(nodes, "node", id=name, x=str(x), y=str(y),
                             type="priority" if not name.startswith("port_") else "dead_end")
        for key, value in C.NODE_OPTIONS.get(name, {}).items():
            node.set(key, str(value).lower() if isinstance(value, bool) else str(value))
        if name in C.LANE_TAPERS:
            node.set("shape", " ".join(f"{px:.3f},{py:.3f}" for px, py in taper_outline(name)))
        if name in signal_nodes:
            node.set("type", "traffic_light")
            node.set("tl", signal_nodes[name])
        if name == "godeungeo":
            node.set("radius", "15")
    write_xml(out / "seongsan.nod.xml", nodes)

    edges = ET.Element("edges")
    for eid, source, target, lanes, speed, mids in C.STREETS:
        speed = C.ROAD_SPEED_OVERRIDES.get(eid, speed)
        if eid in C.SCHOOL_ZONE["edges"] or eid in C.SCHOOL_ZONE["approach_edges"]:
            speed = C.SCHOOL_ZONE.get("control_speed_kmh", C.SCHOOL_ZONE["speed_kmh"]) / 3.6
        points = [C.POINTS[source], *mids, C.POINTS[target]]
        if eid in C.ROAD_SHAPE_ENDPOINTS:
            points[0],points[-1] = C.ROAD_SHAPE_ENDPOINTS[eid]
        direction = C.ONE_WAY_STREETS.get(eid)
        directions = (False,) if direction == 1 else (True,) if direction == -1 else (False, True)
        for reverse in directions:
            shape = list(reversed(points)) if reverse else points
            directed_id = eid + ("_rev" if reverse else "")
            directed_lanes = C.ROAD_LANE_OVERRIDES.get(directed_id, lanes)
            attributes = {
                "id": directed_id,
                "from": target if reverse else source,
                "to": source if reverse else target,
                "numLanes": str(directed_lanes), "speed": str(speed),
                "width": str(C.ROAD_WIDTH_OVERRIDES.get(eid, C.LANE_WIDTH)),
                # Road class from the pre-widening lane count: the +1-lane edit turned 2-lane main roads
                # into 3-lane roads and must not demote them below the widened 1-lane side roads.
                "priority": "2" if getattr(C, "BASE_LANE_COUNTS", {}).get(directed_id, directed_lanes) == 2 else "1",
                "shape": " ".join(f"{x:.3f},{y:.3f}" for x, y in map(xy, shape)),
            }
            if directed_id in C.ROAD_STOP_OFFSETS:
                attributes['endOffset'] = str(C.ROAD_STOP_OFFSETS[directed_id])
            ET.SubElement(edges, "edge", attributes)
    for i, source in enumerate(C.RING_NODES):
        target = C.RING_NODES[(i + 1) % len(C.RING_NODES)]
        # Ring exits are ordered counterclockwise.
        a0 = math.radians(C.RING_ANGLES[i])
        a1 = math.radians(C.RING_ANGLES[i + 1] if i + 1 < len(C.RING_ANGLES)
                          else C.RING_ANGLES[0] + 360)
        samples = max(8, math.ceil((a1 - a0) / (math.pi / 32)))
        shape = [xy((C.RING_CENTER[0] + C.RING_RADIUS_PIXELS * math.cos(a0 + (a1 - a0) * j / samples),
                     C.RING_CENTER[1] - C.RING_RADIUS_PIXELS * math.sin(a0 + (a1 - a0) * j / samples)))
                 for j in range(samples + 1)]
        ET.SubElement(edges, "edge", {
            "id": f"ring_{i}", "from": source, "to": target,
            "numLanes": str(C.RING_LANES), "speed": str(C.SPEED_RING),
            "width": str(C.LANE_WIDTH), "priority": "5",
            "shape": " ".join(f"{x:.3f},{y:.3f}" for x, y in shape),
        })
    write_xml(out / "seongsan.edg.xml", edges)
    connections = ET.Element("connections")
    edge_attributes = {edge.get("id"): edge.attrib for edge in edges.findall("edge")}
    movement_limits = {**C.STRAIGHT_ONLY_CONNECTIONS, **C.RIGHT_ONLY_CONNECTIONS}
    for source, allowed in movement_limits.items():
        junction = edge_attributes[source]["to"]
        for target, attributes in edge_attributes.items():
            if attributes["from"] == junction and target != allowed:
                ET.SubElement(connections, "delete", **{"from": source, "to": target})
        source_lanes = int(edge_attributes[source]["numLanes"])
        target_lanes = int(edge_attributes[allowed]["numLanes"])
        for lane in range(min(source_lanes, target_lanes)):
            ET.SubElement(connections, "connection", **{"from": source, "to": allowed,
                                                          "fromLane": str(lane), "toLane": str(lane)})
    for source, rules in C.LANE_CONNECTION_RULES.items():
        junction = edge_attributes[source]["to"]
        allowed = {(rule["fromLane"], rule["to"], rule["toLane"]) for rule in rules}
        for target, attributes in edge_attributes.items():
            if attributes["from"] == junction:
                for from_lane in range(int(edge_attributes[source]["numLanes"])):
                    for to_lane in range(int(attributes["numLanes"])):
                        if (from_lane, target, to_lane) not in allowed:
                            ET.SubElement(connections, "delete", **{"from": source, "to": target,
                                                                  "fromLane": str(from_lane), "toLane": str(to_lane)})
        for rule in rules:
            ET.SubElement(connections, "connection", **{"from": source, **{k: str(v) for k, v in rule.items()}})
    for rule in C.EXTRA_CONNECTIONS:
        ET.SubElement(connections, "connection", **{k: str(v) for k, v in rule.items()})
    write_xml(out / "seongsan.con.xml", connections)
    command = [checkBinary("netconvert"), "--node-files", str(out / "seongsan.nod.xml"),
               "--edge-files", str(out / "seongsan.edg.xml"),
               "--connection-files", str(out / "seongsan.con.xml"),
               "--output-file", str(out / "seongsan.net.xml"),
               "--no-turnarounds", "true", "--roundabouts.guess", "true",
               "--junctions.corner-detail", "16", "--junctions.internal-link-detail", "16",
               "--geometry.remove", "false", "--offset.disable-normalization", "true",
               "--tls.default-type", "static", "--tls.cycle.time", str(C.SIGNAL_TIMING["cycle_seconds"]),
               "--tls.yellow.time", str(C.SIGNAL_TIMING["yellow_seconds"]),
               "--tls.allred.time", str(C.SIGNAL_TIMING["all_red_seconds"])]
    result = subprocess.run(command, capture_output=True, text=True)
    (out / "netconvert.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stderr)
    for _ in range(3):
        if not align_lane_ends(out, edges):
            break
        write_xml(out / "seongsan.edg.xml", edges)
        result = subprocess.run(command, capture_output=True, text=True)
        (out / "netconvert.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(result.stderr)
    if not C.ROAD_SPEED_LIMITS_ENABLED:
        remove_road_speed_limits(out / "seongsan.net.xml", C.UNRESTRICTED_NETWORK_SPEED,
                                 out / "seongsan.edg.xml")
    net = readNet(str(out / "seongsan.net.xml"), withInternal=True)
    for source, allowed in movement_limits.items():
        actual = {e.getID() for e in net.getEdge(source).getOutgoing()}
        if actual != {allowed}:
            raise ValueError(f"Allowed movement restriction failed: {source}: {actual}")
    route_edges,start_position,start_finish = ego_start(net,C.EGO_ROUTES[route_name])
    ego_edges = route_edges * laps + [route_edges[0]]
    validate_route(net, ego_edges)

    routes = ET.Element("routes")
    ET.SubElement(routes, "vType", id="ego_type", accel="2.5", decel="4.5",
                  maxSpeed=str(C.SPEED_MAIN), color="1,0.12,0.08", length="4.5",
                  minGap="2.5", speedFactor="1", speedDev="0",
                  vClass="passenger", guiShape="passenger/sedan")
    mix = ET.SubElement(routes, "vTypeDistribution", id="background")
    for name, model, probability, color in [
        ("hv_krauss", "Krauss", .35, "0.95,0.75,0.15"),
        ("hv_idm", "IDM", .30, "0.1,0.65,0.9"),
        ("hv_eidm", "EIDM", .20, "0.25,0.75,0.4"),
        ("hv_acc", "ACC", .15, "0.7,0.45,0.85"),
    ]:
        ET.SubElement(mix, "vType", id=name, carFollowModel=model,
                      probability=str(probability), color=color, accel="2.0", decel="4.5",
                      tau="1.8", minGap="2.5", maxSpeed=str(C.SPEED_MAIN),
                      jmTimegapMinor="2.0",
                      speedFactor="normc(0.95,0.08,0.75,1.0)", vClass="passenger")
    ET.SubElement(routes, "route", id="ego_loop", edges=" ".join(ego_edges))
    ports = [(eid, source) for eid, source, *_ in C.STREETS if source.startswith("port_")]
    flows = []
    manifest = []
    if traffic:
        for incoming, origin in ports:
            for outgoing, destination in ports:
                if origin == destination:
                    continue
                via = C.TRAFFIC_VIA_EDGES.get((origin, destination))
                if via:
                    waypoints = [via] if isinstance(via, str) else list(via)
                    targets = waypoints + [outgoing + "_rev"]
                    path = [incoming]
                    for target in targets:
                        path += connected_path(net.getEdge(path[-1]), net.getEdge(target))[1:]
                else:
                    path = connected_path(net.getEdge(incoming), net.getEdge(outgoing + "_rev"))
                validate_route(net, path)
                rid = f"{origin}_to_{destination}"
                ET.SubElement(routes, "route", id=rid, edges=" ".join(path))
                flows.append((rid, C.TRAFFIC_PER_PORT / (len(ports) - 1)))
                manifest.append({"route": rid, "edges": path})
    if include_ego:
        ET.SubElement(routes, "vehicle", id="ego", type="ego_type", route="ego_loop",
                      depart="0", departLane="0", departPos=str(start_position), departSpeed="0",
                      arrivalLane="0", arrivalPos=str(start_position))
    for rid, rate in flows:
        ET.SubElement(routes, "flow", id="flow_" + rid, type="background", route=rid,
                      begin="0", end=str(C.TRAFFIC_END), vehsPerHour=str(rate),
                      departLane="best", departSpeed="max")
    write_xml(out / "seongsan.rou.xml", routes)

    additional = ET.Element("additional")
    surface = road_surface(net)
    pavement_checks = validate_lane_footprints(net,surface)
    for i,shape in enumerate(mask_faces(surface)):
        ET.SubElement(additional,'poly',id=f'road_edge_clip_{i}',color='51,128,51',fill='true',layer='3',
                      shape=' '.join(f'{x:.4f},{y:.4f}' for x,y in shape))
    for i,shape in enumerate(small_islands(surface)):
        ET.SubElement(additional,'poly',id=f'junction_island_{i}',color='0.42,0.43,0.44',fill='true',layer='3.1',
                      shape=' '.join(f'{x:.4f},{y:.4f}' for x,y in shape))
    for i,shape in enumerate(asphalt_faces(surface)):
        ET.SubElement(additional,'poly',id=f'asphalt_face_{i}',color='0,0,0',fill='true',layer='2',
                      shape=' '.join(f'{x:.3f},{y:.3f}' for x,y in shape))
    ET.SubElement(additional, "poly", id="open_field", color="0.55,0.65,0.40", fill="true", layer="-2",
                  shape=" ".join(f"{x:.2f},{y:.2f}" for x,y in map(xy, C.FIELD_POLYGON + C.FIELD_POLYGON[:1])))
    outlines, bollards = island_markings()
    entry_paint = list(entry_markings(net,surface))
    for name,shape,color,fill,width in entry_paint:
        ET.SubElement(additional,'poly',id=name,color=color,fill=str(fill).lower(),layer='6',lineWidth=str(width),
                      shape=' '.join(f'{x:.4f},{y:.4f}' for x,y in shape))
    for name,shape,color,layer in entry_stop_markings(net,surface):
        ET.SubElement(additional,'poly',id=name,color=color,fill='true',layer=layer,
                      shape=' '.join(f'{x:.4f},{y:.4f}' for x,y in shape))
    ET.SubElement(additional, "poly", id="merge_pavement", color="0.42,0.43,0.44", fill="true", layer="-1",
                  shape=" ".join(f"{x:.2f},{y:.2f}" for x,y in map(xy, C.MERGE_PAVEMENT + C.MERGE_PAVEMENT[:1])))
    ET.SubElement(additional, "poly", id="small_merge_island", color="0.42,0.43,0.44", fill="true", layer="-1",
                  shape=" ".join(f"{x:.2f},{y:.2f}" for x,y in map(xy, outlines[0])))
    for i, outline in enumerate(outlines):
        ET.SubElement(additional, "poly", id=f"yellow_hatch_{i}", color="1,0.8,0.1", fill="false", layer="6",
                      lineWidth="0.25", shape=" ".join(f"{x:.2f},{y:.2f}" for x,y in map(xy, outline)))
    lower_outlines, _ = island_markings(C.LOWER_TRAFFIC_ISLAND, (1, .55))
    for i, outline in enumerate(lower_outlines):
        ET.SubElement(additional, "poly", id=f"lower_yellow_hatch_{i}", color="1,0.8,0.1", fill="false", layer="6",
                      lineWidth="0.25", shape=" ".join(f"{x:.2f},{y:.2f}" for x,y in map(xy, outline)))
    for i, point in enumerate(bollards):
        x,y = xy(point)
        shape = " ".join(f"{x+.18*math.cos(j*math.pi/4):.3f},{y+.18*math.sin(j*math.pi/4):.3f}" for j in range(9))
        ET.SubElement(additional, "poly", id=f"red_bollard_{i}", color="0.95,0.18,0.08", fill="true", layer="6.1", shape=shape)
    for lane_id, shape in hidden_arrow_masks(net):
        ET.SubElement(additional, "poly", id="hide_arrow_"+lane_id, color="0,0,0", fill="true", layer="5.2",
                      shape=" ".join(f"{x:.3f},{y:.3f}" for x,y in shape))
    for name,shape,width in [*boundary_lines(net,surface),*lane_dividers(net)]:
        for i,outline in enumerate(painted_stroke(shape,width)):
            ET.SubElement(additional,'poly',id=f'{name}_{i}',color='1,1,1',fill='true',layer='6',
                          shape=' '.join(f'{x:.4f},{y:.4f}' for x,y in outline))
    for name,shape,fill,width in custom_arrows(net):
        ET.SubElement(additional,'poly',id=name,color='1,1,1',fill=str(fill).lower(),layer='6',lineWidth=str(width),
                      shape=' '.join(f'{x:.3f},{y:.3f}' for x,y in shape))
    for eid, shape in solid_centerlines(net):
        for i,outline in enumerate(painted_stroke(shape,.15)):
            ET.SubElement(additional, "poly", id=f"solid_yellow_{eid}_{i}", color="1,0.8,0.1",
                          fill="true", layer="6",
                          shape=" ".join(f"{x:.4f},{y:.4f}" for x, y in reversed(outline)))
    for node_id in C.LANE_TAPERS:
        for i, shape in enumerate(taper_dashes(node_id)):
            ET.SubElement(additional, "poly", id=f"taper_dash_{node_id}_{i}", color="1,1,1",
                          fill="false", layer="6", lineWidth="0.10",
                          shape=" ".join(f"{x:.3f},{y:.3f}" for x, y in shape))
    for eid in sorted((school_edges() if getattr(C, 'SCHOOL_ZONE_MARKINGS_ENABLED', True) else ())):
        for lane in net.getEdge(eid).getLanes():
            strip=outside_strip(lane.getShape(),lane.getWidth()/2+2,surface)
            for i,shape in enumerate(asphalt_faces(strip)):
                ET.SubElement(additional, "poly", id=f"school_zone_{lane.getID()}_{i}",
                              type="school_zone_30", color="0.88,0.35,0.28", fill="true", layer="3.2",
                              shape=" ".join(f"{x:.4f},{y:.4f}" for x, y in shape))
    cx, cy = xy(C.RING_CENTER)
    island_radius = C.ISLAND_RADIUS_PIXELS * C.SCALE
    island = " ".join(f"{cx + island_radius * math.cos(i * math.pi / 24):.2f},"
                      f"{cy + island_radius * math.sin(i * math.pi / 24):.2f}" for i in range(49))
    ET.SubElement(additional, "poly", id="roundabout_island", color="0.35,0.60,0.28",
                  fill="true", layer="2.2", shape=island)
    # 움직이는 빨간 ego와 혼동되는 식당 위치 표시점은 두지 않는다.
    x,y = start_finish['roadside_marker_xy']
    ET.SubElement(additional, "poi", id="START_FINISH", x=str(x), y=str(y),
                  color="0.1,0.7,0.3", type="start")
    write_xml(out / "seongsan.add.xml", additional)
    settings = ET.Element("viewsettings")
    scheme = ET.SubElement(settings, "scheme", name="seongsan")
    ET.SubElement(scheme, "background", backgroundColor="51,128,51")
    ET.SubElement(scheme, "edges", realisticLinkRules="true")
    ET.SubElement(scheme, "vehicles", vehicleQuality="2")
    ET.SubElement(scheme, "polys", poly_minSize="0", poly_exaggeration="1", poly_constantSize="true",
                  polyUseCustomLayer="false")
    xmin, ymin, xmax, ymax = net.getBoundary()
    ET.SubElement(settings, "viewport", zoom="100", x=str((xmin + xmax) / 2),
                  y=str((ymin + ymax) / 2), angle="0")
    ET.SubElement(settings, "delay", value=str(C.GUI_DELAY_MS))
    write_xml(out / "seongsan.gui.xml", settings)
    configuration = ET.Element("configuration")
    inputs = ET.SubElement(configuration, "input")
    for key, filename in [("net-file", "seongsan.net.xml"),
                          ("route-files", "seongsan.rou.xml"),
                          ("additional-files", "seongsan.add.xml")]:
        ET.SubElement(inputs, key, value=filename)
    gui = ET.SubElement(configuration, "gui_only")
    ET.SubElement(gui, "gui-settings-file", value="seongsan.gui.xml")
    time = ET.SubElement(configuration, "time")
    ET.SubElement(time, "step-length", value=str(C.STEP_LENGTH))
    processing = ET.SubElement(configuration, "processing")
    ET.SubElement(processing, "time-to-teleport", value="-1")
    random = ET.SubElement(configuration, "random_number")
    ET.SubElement(random, "seed", value=str(C.SEED))
    write_xml(out / "seongsan.sumocfg", configuration)
    reference_length = route_distance(net, C.REFERENCE_EDGES)
    info = {"scale_m_per_pixel_calibrated": C.SCALE, "laps": laps,
            "layout_compression": {"global_length_factor": C.GLOBAL_LENGTH_FACTOR,
                                   "northern_length_factor": C.NORTH_LENGTH_FACTOR,
                                   "northern_source_y_anchor": C.NORTH_COMPRESSION_ANCHOR_Y,
                                   "entry_junction_reference_factor": C.EAST_ENTRY_REFERENCE_FACTOR,
                                   "southern_length_factor": C.SOUTH_LENGTH_FACTOR,
                                   "southern_source_y_threshold": C.SOUTH_COMPRESSION_START_Y,
                                   "roundabout_size_preserved": True,
                                   "merge_length_factor": C.MERGE_LENGTH_FACTOR,
                                   "lane_widths_preserved": True},
            "selected_ego_route": route_name,
            "ego_included": include_ego,
            "calibration": {"source": "User approximate distance followed by requested layout compression; not a survey",
                            "from": "roundabout east exit", "to": "northbound left-turn entry",
                            "reference_edges": C.REFERENCE_EDGES,
                            "original_reference_m": C.REFERENCE_DISTANCE_M,
                            "target_m": C.REFERENCE_DISTANCE_M * C.GLOBAL_LENGTH_FACTOR,
                            "actual_lane0_m_including_junctions": round(reference_length, 2)},
            "ego_edges": ego_edges, "ego_depart_arrival_position": start_position,
            "start_finish": start_finish,
            "loop_length_m": round(route_distance(net, route_edges + [route_edges[0]])
                                    - net.getEdge(route_edges[0]).getLane(0).getLength(), 1),
            "ports": [p for _, p in ports], "traffic_routes": manifest,
            "road_speed_policy": {"posted_limits_enabled": C.ROAD_SPEED_LIMITS_ENABLED,
                                  "required_sumo_lane_ceiling_mps": C.UNRESTRICTED_NETWORK_SPEED if not C.ROAD_SPEED_LIMITS_ENABLED else None,
                                  "vehicle_max_speed_retained": True},
            "school_zone": {**C.SCHOOL_ZONE, "speed_limit_applied": C.ROAD_SPEED_LIMITS_ENABLED,
                            "bidirectional_edges": sorted(school_edges())},
            "signals": {"controllers": C.SIGNAL_CONTROLLERS, "timing_assumed": C.SIGNAL_TIMING},
            "stop_controlled_junctions": C.STOP_JUNCTIONS,
            "one_way_streets": C.ONE_WAY_STREETS,
            "directed_lane_overrides": C.ROAD_LANE_OVERRIDES,
            "lane_connection_rules": C.LANE_CONNECTION_RULES,
            "merge_connections": C.EXTRA_CONNECTIONS,
            "node_options": C.NODE_OPTIONS,
            "zipper_taper_length_m": C.ZIPPER_TAPER_LENGTH_M,
            "lane_tapers": C.LANE_TAPERS,
            "hidden_lane_arrows": C.HIDDEN_LANE_ARROWS,
            "custom_arrow_edges": C.CUSTOM_ARROW_EDGES,
            "custom_lane_arrows": C.CUSTOM_LANE_ARROWS,
            "unified_pavement_and_white_boundary": True,
            "pavement_lane_coverage": pavement_checks,
            "straightened_streets": sorted(C.STRAIGHT_STREETS),
            "road_curve_handle_m": C.ROAD_CURVE_HANDLE_M,
            "road_curve_step_m": C.ROAD_CURVE_STEP_M,
            "native_junction_markings": True,
            "solid_centerline_streets": C.SOLID_CENTERLINE_STREETS,
            "straight_only_connections": C.STRAIGHT_ONLY_CONNECTIONS,
            "right_only_connections": C.RIGHT_ONLY_CONNECTIONS,
            "east_entry": {"left_signal_edge": "east_signal_approach_rev",
                           "left_turn_control": "allway_stop" if "triangle_northeast" in C.STOP_JUNCTIONS else "traffic_light",
                           "right_yield_edge": "east_right_slip", "yield_node": "east_cross",
                           "painted_island_border_world_xy": next(p[1] for p in entry_paint if p[0]=='entry_yellow_border'),
                           "crosswalks_enabled":False,
                           "stop_line_setback_m": C.ROAD_STOP_OFFSETS,
                           "local_stop_markings": {"stopped_approaches":3,"right_slip":"three yield dashes",
                                                   "native_bar_mask_layer":210,"paint_layer":211},
                           "slip_straight_tail_m":8,
                           "side_approach_stop_bar":"diagonal at mainline mouth, median ends at bar",
                           "outer_entry_straight_length_m":round(C.OUTER_ENTRY_STRAIGHT_LENGTH_M,2),
                           "local_curb_rounding_radius_m":.6,
                           },
            "gui_playback": {"step_seconds": C.STEP_LENGTH, "delay_ms": C.GUI_DELAY_MS},
            "feature_edge_groups": C.FEATURE_EDGE_GROUPS,
            "small_island": {"photo_points": C.TRAFFIC_ISLAND, "bollards_count": len(bollards),
                             "lower_photo_points": C.LOWER_TRAFFIC_ISLAND,
                             "pavement_photo_points": C.MERGE_PAVEMENT,
                             "cross_connection": ["triangle_cross", "triangle_cross_rev"],
                             "description": "painted exclusion area; roads route around it; polygons are visual markers"},
            "traffic_per_hour": len(ports) * C.TRAFFIC_PER_PORT if traffic else 0}
    (out / "manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    write_preview(net, out.parent / "preview.svg", route_edges, start_finish)
    return out / "seongsan.sumocfg"


def validate_route(net, route):
    for source, target in zip(route, route[1:]):
        if net.getEdge(target) not in net.getEdge(source).getOutgoing():
            raise ValueError(f"Disconnected route: {source} -> {target}")


def ego_start(net, cycle):
    """Rotate the existing directed loop to a measured point before the junction."""
    anchor = net.getNode(C.EGO_START_ANCHOR_NODE)
    index = cycle.index(C.EGO_START_ANCHOR_EDGE)
    incoming = net.getEdge(cycle[index]).getLane(0)
    gap = math.dist(incoming.getShape()[-1],anchor.getCoord())
    remaining = C.EGO_START_DISTANCE_M-gap
    if remaining < 0:
        raise ValueError('Start distance is inside the reference junction')
    for _ in cycle:
        edge = net.getEdge(cycle[index])
        if remaining <= edge.getLane(0).getLength():
            position = edge.getLane(0).getLength()-remaining
            if not 4.5 < position < edge.getLane(0).getLength()-.5:
                raise ValueError('Start position needs clearance from the road endpoints')
            shape = edge.getLane(0).getShape()
            offset = position/edge.getLane(0).getLength()*sum(math.dist(a,b) for a,b in zip(shape,shape[1:]))
            for a,b in zip(shape,shape[1:]):
                length = math.dist(a,b)
                if length and offset <= length:
                    direction = tuple((b[i]-a[i])/length for i in (0,1))
                    point = tuple(a[i]+direction[i]*offset for i in (0,1))
                    marker = (point[0]+direction[1]*5,point[1]-direction[0]*5)
                    info = {'same_location':True,'edge':edge.getID(),'lane':0,'position_m':position,
                            'anchor_node':C.EGO_START_ANCHOR_NODE,'distance_from_junction_m':C.EGO_START_DISTANCE_M,
                            'world_position_xy':point,'roadside_marker_xy':marker,
                            'direction':'toward northwest junction, then south toward the roundabout'}
                    return cycle[index:]+cycle[:index],position,info
                offset -= length
            raise ValueError('Could not locate the start point on the lane')
        remaining -= edge.getLane(0).getLength()
        previous = net.getEdge(cycle[(index-1)%len(cycle)])
        links = previous.getConnections(edge)
        connection = next(c for c in links if c.getFromLane().getIndex()==0 and c.getToLane().getIndex()==0)
        junction_length = net.getLane(connection.getViaLaneID()).getLength() if connection.getViaLaneID() else 0
        if remaining < junction_length:
            raise ValueError('Requested start point falls inside a junction')
        remaining -= junction_length
        index = (index-1)%len(cycle)
    raise ValueError('Start distance exceeds the loop length')


def solid_centerlines(net):
    """Yellow lines on the inner boundary of a road's innermost lane."""
    edges = {edge.getID(): edge for edge in net.getEdges()}
    for eid in C.SOLID_CENTERLINE_STREETS:
        edge = edges.get(eid, edges.get(eid + "_rev"))
        lane = edge.getLanes()[-1]
        shape=move2side(lane.getShape(), -lane.getWidth() / 2)
        if eid=='east_signal_approach':
            stopped=net.getEdge(eid+'_rev').getLanes()[-1]
            end=stopped.getShape()[-1]
            a,b=stopped.getShape()[-2:]
            length=math.dist(a,b)
            median=(end[0]-(b[1]-a[1])/length*stopped.getWidth()/2,
                    end[1]+(b[0]-a[0])/length*stopped.getWidth()/2)
            line=LineString(shape)
            shape=list(substring(line,line.project(Point(median)),line.length).coords)
        yield eid, shape
    pairs = set(C.CONTINUOUS_CENTERLINE_CONNECTIONS)
    # Continue the yellow median through geometry-only points and lane tapers.
    # True intersections keep gaps unless a solid median is explicitly requested.
    for node in net.getNodes():
        incoming = [e for e in node.getIncoming() if not e.getID().startswith(":")]
        if len(incoming) != 2:
            continue
        if any(e.getID().startswith('ring_') for e in incoming):
            continue
        for edge in incoming:
            outgoing = [e for e in edge.getOutgoing() if not e.getID().startswith(":")
                        and e.getToNode() != edge.getFromNode()]
            if len(outgoing) == 1:
                pairs.add((edge.getID(), outgoing[0].getID()))
                break
    for source, target in sorted(pairs):
        incoming, outgoing = net.getEdge(source), net.getEdge(target)
        lane_in, lane_out = incoming.getLanes()[-1], outgoing.getLanes()[-1]
        before = move2side(lane_in.getShape(), -lane_in.getWidth()/2)
        after = move2side(lane_out.getShape(), -lane_out.getWidth()/2)
        start, finish = before[-1], after[0]
        distance = math.dist(start, finish)
        if distance < 0.01:
            continue
        tangent_in = (before[-1][0]-before[-2][0], before[-1][1]-before[-2][1])
        tangent_out = (after[1][0]-after[0][0], after[1][1]-after[0][1])
        lengths = math.hypot(*tangent_in), math.hypot(*tangent_out)
        control = min(distance*.4, 10)
        a = tuple(start[i]+tangent_in[i]/max(lengths[0],.001)*control for i in (0,1))
        b = tuple(finish[i]-tangent_out[i]/max(lengths[1],.001)*control for i in (0,1))
        shape = [tuple((1-t)**3*start[i]+3*(1-t)**2*t*a[i]+3*(1-t)*t*t*b[i]+t**3*finish[i]
                       for i in (0,1)) for t in (j/20 for j in range(21))]
        yield source + "_to_" + target, shape


def hidden_arrow_masks(net):
    """Hide selected lane decals, leaving signal bars and lane lines visible."""
    for lane_id in sorted(set(C.HIDDEN_LANE_ARROWS) | set(custom_arrow_lanes(net))):
        lane=net.getLane(lane_id); a,b=lane.getShape()[-2:]
        dx,dy=b[0]-a[0],b[1]-a[1]; length=math.hypot(dx,dy)
        dx,dy=dx/length,dy/length
        half_width=min(1.15,lane.getWidth()/2-.2)
        def point(back,side):
            return b[0]-back*dx+side*dy,b[1]-back*dy-side*dx
        shape=[point(.7,-half_width),point(.7,half_width),point(4.6,half_width),point(4.6,-half_width)]
        yield lane_id,shape+shape[:1]


def taper_point(node_id, along, side):
    x, y = xy(C.POINTS[node_id])
    dx, dy = C.LANE_TAPERS[node_id]['direction']
    length = math.hypot(dx, dy)
    dx, dy = dx / length, dy / length
    return x + along*dx + side*dy, y + along*dy - side*dx


def taper_outline(node_id='coast_lane_change'):
    """A continuous, 18m-long road taper; no cross road or stop line."""
    half_length = C.ZIPPER_TAPER_LENGTH_M / 2
    spec = C.LANE_TAPERS[node_id]
    width = spec['width']
    south_n = spec.get('southbound_north', spec['southbound'])
    south_s = spec.get('southbound_south', spec['southbound'])
    return [taper_point(node_id, half_length, -south_n*width),
            taper_point(node_id, half_length, spec['north_after']*width),
            taper_point(node_id, -half_length, spec['north_before']*width),
            taper_point(node_id, -half_length, -south_s*width)]


def taper_dashes(node_id='coast_lane_change'):
    """Lane dividers that run through the whole taper (lanes present on both sides)."""
    start = -C.ZIPPER_TAPER_LENGTH_M / 2
    spec = C.LANE_TAPERS[node_id]
    width = spec['width']
    north = max(1, min(spec['north_before'], spec['north_after']))
    south = min(spec.get('southbound_north', spec['southbound']), spec.get('southbound_south', spec['southbound']))
    offsets = [k * width for k in range(1, north + 1)] + [-k * width for k in range(1, south)]
    for offset in offsets:
        for distance in range(0, int(C.ZIPPER_TAPER_LENGTH_M), 3):
            yield [taper_point(node_id, start + distance, offset),
                   taper_point(node_id, min(start + distance + 1.5, C.ZIPPER_TAPER_LENGTH_M / 2), offset)]


def route_distance(net, route):
    """Representative driving distance, including compiled junction connections.

    Prefer outer lane; use the permitted lane for lane-restricted turns.
    """
    total = sum(net.getEdge(eid).getLane(0).getLength() for eid in route)
    for source, target in zip(route, route[1:]):
        connections = net.getEdge(source).getConnections(net.getEdge(target))
        connection = next((c for c in connections
                           if c.getFromLane().getIndex() == 0 and c.getToLane().getIndex() == 0),
                          connections[0])
        via = connection.getViaLaneID()
        if via:
            total += net.getLane(via).getLength()
    return total


def write_preview(net, path, route_edges=None, start_finish=None):
    """Native SVG preview from compiled lane shapes, without plotting packages."""
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    def tag(name):
        return "{" + ns + "}" + name
    route_edges = route_edges or C.EGO_LOOP
    all_points = list(C.POINTS.values()) + [p for _, p in C.LABELS]
    xmin = min(p[0] for p in all_points) - 110
    ymin = min(p[1] for p in all_points) - 95
    xmax = max(p[0] for p in all_points) + 330
    ymax = max(p[1] for p in all_points) + 180
    width, height = xmax - xmin, ymax - ymin
    svg = ET.Element(tag("svg"), viewBox=f"{xmin} {ymin} {width} {height}",
                     width="1240", height=str(round(1240 * height / width)))
    ET.SubElement(svg, tag("rect"), x=str(xmin), y=str(ymin), width=str(width), height=str(height), fill="#eef2e8")
    def screen(points):
        return " ".join(f"{x / C.SCALE:.2f},{929 - y / C.SCALE:.2f}" for x, y in points)
    ET.SubElement(svg, tag("polygon"), points=" ".join(f"{x},{y}" for x,y in C.FIELD_POLYGON), fill="#b4c496")
    ET.SubElement(svg, tag("polygon"), points=" ".join(f"{x},{y}" for x,y in C.MERGE_PAVEMENT), fill="#65717a")
    surface=road_surface(net)
    for eid in (school_edges() if getattr(C, 'SCHOOL_ZONE_MARKINGS_ENABLED', True) else ()):
        for lane in net.getEdge(eid).getLanes():
            for shape in asphalt_faces(outside_strip(lane.getShape(),lane.getWidth()/2+2,surface)):
                ET.SubElement(svg,tag('polygon'),points=screen(shape),fill='#e98474')
    for shape in small_islands(surface):
        ET.SubElement(svg,tag('polygon'),points=screen(shape),fill='#9a9fa3')
    for shape in asphalt_faces(surface):
        ET.SubElement(svg,tag('polygon'),points=screen(shape),fill='#65717a')
    for _,shape,width in [*boundary_lines(net,surface),*lane_dividers(net)]:
        ET.SubElement(svg,tag('polyline'),points=screen(shape),fill='none',stroke='white',
                      **{'stroke-width':str(width/C.SCALE),'stroke-linejoin':'round'})
    for node in C.STOP_JUNCTIONS:
        for edge in net.getNode(node).getIncoming():
            if edge.getID().startswith(':'):
                continue
            if edge.getID() in C.ROAD_STOP_OFFSETS:
                continue
            for lane in edge.getLanes():
                stop=[move2side(lane.getShape(),side*lane.getWidth()/2)[-1] for side in (-1,1)]
                ET.SubElement(svg,tag('polyline'),points=screen(stop),fill='none',stroke='white',
                              **{'stroke-width':str(.4/C.SCALE)})
    for _,shape,fill,width in custom_arrows(net):
        ET.SubElement(svg,tag('polygon') if fill else tag('polyline'),points=screen(shape),
                      fill='white' if fill else 'none',stroke='white',**{'stroke-width':str(width/C.SCALE)})
    for _, shape in solid_centerlines(net):
        ET.SubElement(svg, tag("polyline"), points=screen(shape), fill="none", stroke="#f2ca4b",
                      **{"stroke-width": str(.15 / C.SCALE)})
    for node_id in C.LANE_TAPERS:
        for shape in taper_dashes(node_id):
            ET.SubElement(svg, tag("polyline"), points=screen(shape), fill="none", stroke="white",
                          **{"stroke-width": str(.10 / C.SCALE)})
    for _,shape,color,fill,width in entry_markings(net,surface):
        color = '#f2ca4b' if color=='1,0.8,0.1' else 'white'
        ET.SubElement(svg,tag('polygon') if fill else tag('polyline'),points=screen(shape),
                      fill=color if fill else 'none',stroke=color,**{'stroke-width':str(width/C.SCALE)})
    for name,shape,color,_ in entry_stop_markings(net,surface):
        if name.startswith(('entry_bar_mask_','fork_')):
            continue
        ET.SubElement(svg,tag('polygon'),points=screen(shape),fill='#f2ca4b' if color=='1,0.8,0.1' else 'white')
    for eid in route_edges:
        ET.SubElement(svg, tag("polyline"), points=screen(net.getEdge(eid).getLane(0).getShape()),
                      fill="none", stroke="#ed4638", **{"stroke-width": "2.7"})
    for source, target in zip(route_edges, route_edges[1:] + route_edges[:1]):
        connections = net.getEdge(source).getConnections(net.getEdge(target))
        connection = next((c for c in connections if c.getFromLane().getIndex() == 0), connections[0])
        via = connection.getViaLaneID()
        points = [net.getEdge(source).getLane(0).getShape()[-1]]
        if via:
            points.extend(net.getLane(via).getShape())
        points.append(connection.getToLane().getShape()[0])
        ET.SubElement(svg, tag("polyline"), points=screen(points), fill="none", stroke="#ed4638",
                      **{"stroke-width": "2.7"})
    ET.SubElement(svg, tag("circle"), cx=str(C.RING_CENTER[0]), cy=str(C.RING_CENTER[1]), r=str(C.ISLAND_RADIUS_PIXELS), fill="#79a95a")
    outlines, bollards = island_markings()
    ET.SubElement(svg, tag("polygon"), points=" ".join(f"{x},{y}" for x,y in outlines[0]), fill="#65717a")
    for outline in outlines:
        ET.SubElement(svg, tag("polyline"), points=" ".join(f"{x},{y}" for x,y in outline), fill="none",
                      stroke="#f2ca4b", **{"stroke-width":"1.2"})
    lower_outlines, _ = island_markings(C.LOWER_TRAFFIC_ISLAND, (1, .55))
    for outline in lower_outlines:
        ET.SubElement(svg, tag("polyline"), points=" ".join(f"{x},{y}" for x,y in outline), fill="none",
                      stroke="#f2ca4b", **{"stroke-width":"1.2"})
    for x,y in bollards:
        ET.SubElement(svg, tag("circle"), cx=str(x), cy=str(y), r=".65", fill="#ed4936")
    for eid in route_edges[::2]:
        shape = net.getEdge(eid).getLane(0).getShape()
        a, b = shape[len(shape) // 2 - 1:len(shape) // 2 + 1]
        x, y = b[0] / C.SCALE, 929 - b[1] / C.SCALE
        angle = math.degrees(math.atan2(-(b[1] - a[1]), b[0] - a[0]))
        ET.SubElement(svg, tag("path"), d="M -10,-6 L 0,0 L -10,6", fill="none", stroke="#ed4638",
                      **{"stroke-width": "3", "transform": f"translate({x},{y}) rotate({angle})"})
    for label, (x, y) in C.LABELS:
        ET.SubElement(svg, tag("text"), x=str(x), y=str(y), fill="#172f40",
                      **{"font-family": "Malgun Gothic, sans-serif", "font-size": "16",
                         "paint-order": "stroke", "stroke": "#eef2e8", "stroke-width": "4"}).text = label
    for name, (x, y) in C.POINTS.items():
        if name.startswith("port_"):
            ET.SubElement(svg, tag("circle"), cx=str(x), cy=str(y), r="8", fill="#227cb5")
            ET.SubElement(svg, tag("text"), x=str(x - 20), y=str(y + 26),
                          **{"font-size": "14", "fill": "#227cb5"}).text = name
    for members in C.SIGNAL_CONTROLLERS.values():
        for node in members:
            x, y = C.POINTS[node]
            ET.SubElement(svg, tag("rect"), x=str(x + 16), y=str(y - 30), width="15", height="40",
                          rx="3", fill="#243544")
            for offset, color in [(7, "#f04b43"), (20, "#ffd25b"), (33, "#47c482")]:
                ET.SubElement(svg, tag("circle"), cx=str(x + 23.5), cy=str(y - 30 + offset), r="4", fill=color)
    for node in (C.SCHOOL_ZONE["sign_nodes"] if C.ROAD_SPEED_LIMITS_ENABLED else []):
        x, y = C.POINTS[node]
        ET.SubElement(svg, tag("circle"), cx=str(x - 25), cy=str(y), r="14", fill="white",
                      stroke="#d63b30", **{"stroke-width": "3"})
        ET.SubElement(svg, tag("text"), x=str(x - 25), y=str(y + 5),
                      **{"font-size": "14", "text-anchor": "middle", "font-family": "sans-serif"}).text = "30"
    for eid, direction in C.ONE_WAY_STREETS.items():
        lane = net.getEdge(eid + ("_rev" if direction == -1 else "")).getLane(0)
        points = lane.getShape()
        # 형상점이 두 개뿐이어도 진행 방향이 보이도록 중간에 화살표를 배치.
        # Follow the lane at half its arc length, including curved one-way roads.
        remaining = sum(math.dist(a, b) for a, b in zip(points, points[1:])) / 2
        for a, b in zip(points, points[1:]):
            length = math.dist(a, b)
            if length and remaining <= length:
                t = remaining / length
                x = (a[0] + (b[0] - a[0]) * t) / C.SCALE
                y = 929 - (a[1] + (b[1] - a[1]) * t) / C.SCALE
                break
            remaining -= length
        angle = math.degrees(math.atan2(-(b[1] - a[1]), b[0] - a[0]))
        ET.SubElement(svg, tag("path"), d="M -9,0 L 9,0 M 4,-2 L 9,0 L 4,2", fill="none",
                      stroke="#227cb5", **{"stroke-width": "1.4", "transform": f"translate({x},{y}) rotate({angle})"})
    if start_finish:
        x,y = start_finish['world_position_xy']
        x,y = x/C.SCALE,929-y/C.SCALE
        ET.SubElement(svg,tag('circle'),cx=str(x),cy=str(y),r='7',fill='#ed4638')
        ET.SubElement(svg,tag('text'),x=str(x+18),y=str(y-20),fill='#172f40',
                      **{'font-size':'16','font-family':'Malgun Gothic, sans-serif'}).text='출발 / 복귀 · 사거리 동쪽 100m'
    scale_x, scale_y = xmax - 100 / C.SCALE - 50, ymax - 100
    ET.SubElement(svg, tag("path"), d=f"M {scale_x},{scale_y} h {100 / C.SCALE:.2f}",
                  stroke="#172f40", **{"stroke-width": "4"})
    ET.SubElement(svg, tag("text"), x=str(scale_x), y=str(scale_y - 15), **{"font-size": "18"}).text = "100 m"
    reference_length = route_distance(net, C.REFERENCE_EDGES)
    for y, text in [(ymin + 35, "SEONGSAN · 남쪽 확장 맵 / 빨강: 시연 경로 / 파랑: 외부 진출입·일방통행"),
                    (ymax - 30, f"로터리 동쪽 출구 → 좌회전 진입부 {reference_length:.1f}m · 전체 길이 {C.GLOBAL_LENGTH_FACTOR:.0%} / 남쪽 추가 압축")]:
        ET.SubElement(svg, tag("text"), x=str(xmin + 25), y=str(y),
                      **{"font-size": "19", "font-family": "Malgun Gothic, sans-serif", "fill": "#172f40"}).text = text
    write_xml(path, svg)
    for suffix, box, dimensions in [
        ("north", "-155 -75 1800 1300", (1240, 896)),
        ("south", "-340 650 2060 1100", (1100, 587)),
        ("triangle", "300 500 850 700", (1100, 906)),
        ("west_junction", "245 700 310 250", (1000, 806)),
        ("lane_merge", "625 1150 240 230", (900, 862)),
        ("east_signal", "740 440 300 390", (850, 1105)),
        ("merge", "625 890 215 310", (780, 1125)),
    ]:
        if suffix in {"merge", "lane_merge", "east_signal", "west_junction"}:
            for child in list(svg):
                if child.tag == tag("text"):
                    svg.remove(child)
        svg.set("viewBox", box)
        svg.set("width", str(dimensions[0]))
        svg.set("height", str(dimensions[1]))
        write_xml(Path(path).with_name(f"preview_{suffix}.svg"), svg)
