"""Create a reproducible SUMO network and its traffic from config.py."""
import heapq
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from sumolib import checkBinary
from sumolib.net import readNet

from . import config as C


def xy(point):
    return point[0] * C.SCALE, (929 - point[1]) * C.SCALE


def write_xml(path, root):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def school_edges():
    return {eid + suffix for eid in C.SCHOOL_ZONE["edges"] for suffix in ("", "_rev")}


def ribbon(points, half_width):
    """Visual background strip for a road; has no effect on vehicle movement."""
    left, right = [], []
    for i, (x, y) in enumerate(points):
        prev, after = points[max(i - 1, 0)], points[min(i + 1, len(points) - 1)]
        dx, dy = after[0] - prev[0], after[1] - prev[1]
        length = math.hypot(dx, dy) or 1
        nx, ny = -dy / length * half_width, dx / length * half_width
        left.append((x + nx, y + ny))
        right.append((x - nx, y - ny))
    return left + list(reversed(right)) + [left[0]]


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


def build(out_dir=None, laps=1, traffic=True, include_ego=True):
    if laps < 1:
        raise ValueError("laps must be >= 1")
    out = Path(out_dir or Path(__file__).parent / "generated").resolve()
    out.mkdir(parents=True, exist_ok=True)
    nodes = ET.Element("nodes")
    signal_nodes = {node: controller for controller, members in C.SIGNAL_CONTROLLERS.items() for node in members}
    for name, point in C.POINTS.items():
        x, y = xy(point)
        node = ET.SubElement(nodes, "node", id=name, x=str(x), y=str(y),
                             type="priority" if not name.startswith("port_") else "dead_end")
        if name in signal_nodes:
            node.set("type", "traffic_light")
            node.set("tl", signal_nodes[name])
        if name == "godeungeo":
            node.set("radius", "15")
    write_xml(out / "seongsan.nod.xml", nodes)

    edges = ET.Element("edges")
    for eid, source, target, lanes, speed, mids in C.STREETS:
        if eid in C.SCHOOL_ZONE["edges"]:
            speed = C.SCHOOL_ZONE["speed_kmh"] / 3.6
        points = [C.POINTS[source], *mids, C.POINTS[target]]
        direction = C.ONE_WAY_STREETS.get(eid)
        directions = (False,) if direction == 1 else (True,) if direction == -1 else (False, True)
        for reverse in directions:
            shape = list(reversed(points)) if reverse else points
            ET.SubElement(edges, "edge", {
                "id": eid + ("_rev" if reverse else ""),
                "from": target if reverse else source,
                "to": source if reverse else target,
                "numLanes": str(lanes), "speed": str(speed),
                "width": str(C.LANE_WIDTH), "priority": "2" if lanes == 2 else "1",
                "shape": " ".join(f"{x:.3f},{y:.3f}" for x, y in map(xy, shape)),
            })
    for i, source in enumerate(C.RING_NODES):
        target = C.RING_NODES[(i + 1) % len(C.RING_NODES)]
        # N -> W -> S -> E -> NE -> N: all lanes counterclockwise.
        a0 = math.radians(C.RING_ANGLES[i])
        a1 = math.radians(C.RING_ANGLES[i + 1] if i + 1 < len(C.RING_ANGLES)
                          else C.RING_ANGLES[0] + 360)
        samples = max(8, math.ceil((a1 - a0) / (math.pi / 32)))
        shape = [xy((155 + 34 * math.cos(a0 + (a1 - a0) * j / samples),
                     858 - 34 * math.sin(a0 + (a1 - a0) * j / samples)))
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
    for source, allowed in C.STRAIGHT_ONLY_CONNECTIONS.items():
        junction = edge_attributes[source]["to"]
        for target, attributes in edge_attributes.items():
            if attributes["from"] == junction and target != allowed:
                ET.SubElement(connections, "delete", **{"from": source, "to": target})
        ET.SubElement(connections, "connection", **{"from": source, "to": allowed,
                                                      "fromLane": "0", "toLane": "0"})
    write_xml(out / "seongsan.con.xml", connections)
    command = [checkBinary("netconvert"), "--node-files", str(out / "seongsan.nod.xml"),
               "--edge-files", str(out / "seongsan.edg.xml"),
               "--connection-files", str(out / "seongsan.con.xml"),
               "--output-file", str(out / "seongsan.net.xml"),
               "--no-turnarounds", "true", "--roundabouts.guess", "true",
               "--junctions.corner-detail", "8", "--junctions.internal-link-detail", "8",
               "--geometry.remove", "false", "--offset.disable-normalization", "true",
               "--tls.default-type", "static", "--tls.cycle.time", str(C.SIGNAL_TIMING["cycle_seconds"]),
               "--tls.yellow.time", str(C.SIGNAL_TIMING["yellow_seconds"]),
               "--tls.allred.time", str(C.SIGNAL_TIMING["all_red_seconds"])]
    result = subprocess.run(command, capture_output=True, text=True)
    (out / "netconvert.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stderr)
    net = readNet(str(out / "seongsan.net.xml"), withInternal=True)
    for source, allowed in C.STRAIGHT_ONLY_CONNECTIONS.items():
        actual = {e.getID() for e in net.getEdge(source).getOutgoing()}
        if actual != {allowed}:
            raise ValueError(f"Straight-only connection failed: {source}: {actual}")
    ego_edges = C.EGO_LOOP * laps + [C.EGO_LOOP[0]]
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
                    first = connected_path(net.getEdge(incoming), net.getEdge(via))
                    second = connected_path(net.getEdge(via), net.getEdge(outgoing + "_rev"))
                    path = first + second[1:]
                else:
                    path = connected_path(net.getEdge(incoming), net.getEdge(outgoing + "_rev"))
                validate_route(net, path)
                rid = f"{origin}_to_{destination}"
                ET.SubElement(routes, "route", id=rid, edges=" ".join(path))
                flows.append((rid, C.TRAFFIC_PER_PORT / (len(ports) - 1)))
                manifest.append({"route": rid, "edges": path})
    if include_ego:
        ET.SubElement(routes, "vehicle", id="ego", type="ego_type", route="ego_loop",
                      depart="0", departLane="0", departPos="10", departSpeed="0",
                      arrivalPos="10")
    for rid, rate in flows:
        ET.SubElement(routes, "flow", id="flow_" + rid, type="background", route=rid,
                      begin="0", end=str(C.TRAFFIC_END), vehsPerHour=str(rate),
                      departLane="best", departSpeed="max")
    write_xml(out / "seongsan.rou.xml", routes)

    additional = ET.Element("additional")
    for eid in sorted(school_edges()):
        for lane in net.getEdge(eid).getLanes():
            shape = ribbon(lane.getShape(), lane.getWidth() / 2 + 2)
            ET.SubElement(additional, "poly", id="school_zone_" + lane.getID(),
                          type="school_zone_30", color="0.88,0.35,0.28", fill="true", layer="-1",
                          shape=" ".join(f"{x:.3f},{y:.3f}" for x, y in shape))
    cx, cy = xy((155, 858))
    island_radius = C.ISLAND_RADIUS_PIXELS * C.SCALE
    island = " ".join(f"{cx + island_radius * math.cos(i * math.pi / 24):.2f},"
                      f"{cy + island_radius * math.sin(i * math.pi / 24):.2f}" for i in range(49))
    ET.SubElement(additional, "poly", id="roundabout_island", color="0.35,0.60,0.28",
                  fill="true", layer="-1", shape=island)
    # 움직이는 빨간 ego와 혼동되는 식당 위치 표시점은 두지 않는다.
    x, y = xy(C.POINTS["hamabo"])
    ET.SubElement(additional, "poi", id="START_HAMABO", x=str(x - 8), y=str(y),
                  color="0.1,0.7,0.3", type="start")
    write_xml(out / "seongsan.add.xml", additional)
    settings = ET.Element("viewsettings")
    ET.SubElement(settings, "scheme", name="real world")
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
            "ego_included": include_ego,
            "calibration": {"source": "User approximate distance, not a survey",
                            "from": "roundabout east exit", "to": "northbound left-turn entry",
                            "reference_edges": C.REFERENCE_EDGES,
                            "target_m": C.REFERENCE_DISTANCE_M,
                            "actual_lane0_m_including_junctions": round(reference_length, 2)},
            "ego_edges": ego_edges, "ego_depart_arrival_position": 10,
            "loop_length_m": round(route_distance(net, C.EGO_LOOP + [C.EGO_LOOP[0]])
                                    - net.getEdge(C.EGO_LOOP[0]).getLane(0).getLength(), 1),
            "ports": [p for _, p in ports], "traffic_routes": manifest,
            "school_zone": {**C.SCHOOL_ZONE, "bidirectional_edges": sorted(school_edges())},
            "signals": {"controllers": C.SIGNAL_CONTROLLERS, "timing_assumed": C.SIGNAL_TIMING},
            "one_way_streets": C.ONE_WAY_STREETS,
            "straight_only_connections": C.STRAIGHT_ONLY_CONNECTIONS,
            "gui_playback": {"step_seconds": C.STEP_LENGTH, "delay_ms": C.GUI_DELAY_MS},
            "traffic_per_hour": len(ports) * C.TRAFFIC_PER_PORT if traffic else 0}
    (out / "manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    write_preview(net, out.parent / "preview.svg")
    return out / "seongsan.sumocfg"


def validate_route(net, route):
    for source, target in zip(route, route[1:]):
        if net.getEdge(target) not in net.getEdge(source).getOutgoing():
            raise ValueError(f"Disconnected route: {source} -> {target}")


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


def write_preview(net, path):
    """Native SVG preview from compiled lane shapes, without plotting packages."""
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    def tag(name):
        return "{" + ns + "}" + name
    svg = ET.Element(tag("svg"), viewBox="-155 -75 1550 1240", width="1240", height="992")
    ET.SubElement(svg, tag("rect"), x="-155", y="-75", width="1550", height="1240", fill="#eef2e8")
    def screen(points):
        return " ".join(f"{x / C.SCALE:.2f},{929 - y / C.SCALE:.2f}" for x, y in points)
    for eid in school_edges():
        for lane in net.getEdge(eid).getLanes():
            ET.SubElement(svg, tag("polyline"), points=screen(lane.getShape()),
                          fill="none", stroke="#e98474",
                          **{"stroke-width": str((lane.getWidth() + 4) / C.SCALE)})
    for node in net.getNodes():
        if node.getShape():
            ET.SubElement(svg, tag("polygon"), points=screen(node.getShape()), fill="#65717a")
    for edge in net.getEdges():
        for lane in edge.getLanes():
            ET.SubElement(svg, tag("polyline"), points=screen(lane.getShape()),
                          fill="none", stroke="#65717a",
                          **{"stroke-width": str(lane.getWidth() / C.SCALE + 0.1), "stroke-linejoin": "round"})
    for eid in C.EGO_LOOP:
        ET.SubElement(svg, tag("polyline"), points=screen(net.getEdge(eid).getLane(0).getShape()),
                      fill="none", stroke="#ed4638", **{"stroke-width": "2.7"})
    for source, target in zip(C.EGO_LOOP, C.EGO_LOOP[1:] + C.EGO_LOOP[:1]):
        connections = net.getEdge(source).getConnections(net.getEdge(target))
        connection = next((c for c in connections if c.getFromLane().getIndex() == 0), connections[0])
        via = connection.getViaLaneID()
        points = [net.getEdge(source).getLane(0).getShape()[-1]]
        if via:
            points.extend(net.getLane(via).getShape())
        points.append(connection.getToLane().getShape()[0])
        ET.SubElement(svg, tag("polyline"), points=screen(points), fill="none", stroke="#ed4638",
                      **{"stroke-width": "2.7"})
    ET.SubElement(svg, tag("circle"), cx="155", cy="858", r=str(C.ISLAND_RADIUS_PIXELS), fill="#79a95a")
    for eid in C.EGO_LOOP[::2]:
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
    for node in C.SCHOOL_ZONE["sign_nodes"]:
        x, y = C.POINTS[node]
        ET.SubElement(svg, tag("circle"), cx=str(x - 25), cy=str(y), r="14", fill="white",
                      stroke="#d63b30", **{"stroke-width": "3"})
        ET.SubElement(svg, tag("text"), x=str(x - 25), y=str(y + 5),
                      **{"font-size": "14", "text-anchor": "middle", "font-family": "sans-serif"}).text = "30"
    for eid, direction in C.ONE_WAY_STREETS.items():
        lane = net.getEdge(eid + ("_rev" if direction == -1 else "")).getLane(0)
        points = lane.getShape()
        # 형상점이 두 개뿐이어도 진행 방향이 보이도록 중간에 화살표를 배치.
        a, b = points[0], points[-1]
        x, y = (a[0] + b[0]) / 2 / C.SCALE, 929 - (a[1] + b[1]) / 2 / C.SCALE
        angle = math.degrees(math.atan2(-(b[1] - a[1]), b[0] - a[0]))
        ET.SubElement(svg, tag("path"), d="M -20,0 L 20,0 M 8,-8 L 20,0 L 8,8", fill="none",
                      stroke="#227cb5", **{"stroke-width": "4", "transform": f"translate({x},{y}) rotate({angle})"})
    ET.SubElement(svg, tag("circle"), cx="64", cy="263", r="7", fill="#ed4638")
    ET.SubElement(svg, tag("path"), d=f"M 900,1050 h {100 / C.SCALE:.2f}",
                  stroke="#172f40", **{"stroke-width": "4"})
    ET.SubElement(svg, tag("text"), x="900", y="1035", **{"font-size": "18"}).text = "100 m"
    reference_length = route_distance(net, C.REFERENCE_EDGES)
    for y, text in [(-40, "SEONGSAN · 고성리 폐루프 / 빨강: 우리 차 경로 / 파랑: 외부 차량 진입·진출"),
                    (1110, f"로터리 동쪽 출구 → 좌회전 진입부 {reference_length:.1f}m · 사용자 제공 약 600m 기준 축척 보정")]:
        ET.SubElement(svg, tag("text"), x="-115", y=str(y),
                      **{"font-size": "19", "font-family": "Malgun Gothic, sans-serif", "fill": "#172f40"}).text = text
    write_xml(path, svg)
