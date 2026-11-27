"""One pavement footprint shared by the asphalt and its white boundary."""
import math

from shapely import constrained_delaunay_triangles, make_valid, union_all
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring
from sumolib.geomhelper import move2side


def polygon_parts(geometry):
    if geometry.geom_type == 'Polygon':
        yield geometry
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def line_parts(geometry):
    if geometry.geom_type in ('LineString', 'LinearRing'):
        if geometry.length > .05:
            yield list(geometry.coords)
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            yield from line_parts(part)


def pavement(net, island_centers=(), omit_node_shapes=(), preserve_holes_in=()):
    pieces = []
    for edge in net.getEdges(withInternal=True):
        for lane in edge.getLanes():
            points = list(dict.fromkeys(lane.getShape()))
            if len(points) < 2:
                continue
            # move2side skips vertices at narrow turns and can cut across a lane.
            # A swept centerline keeps the full lane visible through every bend.
            pieces.append(LineString(points).buffer(lane.getWidth()/2,
                          cap_style='flat',join_style='round',quad_segs=8))
    for node in net.getNodes():
        if node.getID() in omit_node_shapes:
            continue
        if len(node.getShape()) >= 3:
            pieces.append(make_valid(Polygon(node.getShape())))
    # Close rounding cracks in SUMO's coordinates without widening the road.
    surface = union_all(pieces).buffer(.025, quad_segs=4).buffer(-.025, quad_segs=4)
    cleaned = []
    for part in polygon_parts(surface):
        holes = [ring.coords for ring in part.interiors
                 if Polygon(ring).area > 500 or any(Polygon(ring).covers(Point(p)) for p in island_centers)
                 or any(region.covers(Polygon(ring).representative_point()) for region in preserve_holes_in)]
        cleaned.append(Polygon(part.exterior.coords, holes))
    return union_all(cleaned)


def asphalt_faces(surface):
    # SUMO polygons cannot contain holes. Interior-only triangulation preserves them.
    for triangle in constrained_delaunay_triangles(surface).geoms:
        yield list(triangle.exterior.coords)


def validate_lane_footprints(net, surface):
    """Catch pavement clipping a simulated lane, including its full width."""
    tolerance = surface.buffer(.03)
    checked = 0
    worst = 0
    for edge in net.getEdges(withInternal=True):
        for lane in edge.getLanes():
            footprint = LineString(lane.getShape()).buffer(lane.getWidth()/2,
                        cap_style='flat',join_style='round',quad_segs=8)
            missing = footprint.difference(tolerance).area
            if missing > .01:
                raise ValueError(f'Pavement clips lane {lane.getID()}: {missing:.3f} m2 missing')
            checked += 1
            worst = max(worst,missing)
    return {'lanes_checked':checked,'max_missing_lane_area_m2':worst,'boundary_tolerance_m':.03}


def mask_faces(surface):
    for shape in line_parts(surface.boundary):
        for chunk in line_chunks(LineString(shape)):
            outside=chunk.buffer(3,quad_segs=4).difference(surface).buffer(.025,quad_segs=2)
            for part in polygon_parts(outside):
                if part.interiors:
                    yield from asphalt_faces(part)
                else:
                    yield list(part.exterior.coords)


def outside_strip(shape, half_width, surface):
    return LineString(shape).buffer(half_width, cap_style='flat').difference(surface)


def painted_stroke(shape, width):
    """Filled strokes avoid SUMO's long miters on dense curved polylines."""
    for chunk in line_chunks(LineString(shape)):
        stroke=chunk.buffer(width/2,cap_style='flat',join_style='round',quad_segs=4)
        for part in polygon_parts(stroke):
            if part.interiors:
                yield from asphalt_faces(part)
            else:
                yield list(part.exterior.coords)


def line_chunks(line, length=20):
    if line.is_ring:
        length=min(length,line.length/2)
    for i in range(max(1,math.ceil(line.length/length))):
        chunk=substring(line,i*length,min((i+1)*length,line.length))
        if chunk.geom_type=='LineString' and chunk.length>.001:
            yield chunk


def small_islands(surface):
    for part in polygon_parts(surface):
        for hole in part.interiors:
            if Polygon(hole).area < 500:
                yield list(hole.coords)


def boundary_lines(net, surface):
    boundary = surface.boundary
    caps = []
    for node in net.getNodes():
        if node.getType() != 'dead_end':
            continue
        for edge in [*node.getIncoming(), *node.getOutgoing()]:
            if edge.getID().startswith(':'):
                continue
            at_end = edge.getToNode() == node
            index = -1 if at_end else 0
            lane = edge.getLane(0)
            points = lane.getShape()
            a,b = (points[-2],points[-1]) if at_end else (points[1],points[0])
            dx,dy = b[0]-a[0],b[1]-a[1]
            length = math.hypot(dx,dy)
            center = points[index]
            normal = (-dy/length,dx/length)
            width = sum(l.getWidth() for l in edge.getLanes())+2
            caps.append(LineString([(center[0]+normal[0]*width,center[1]+normal[1]*width),
                                    (center[0]-normal[0]*width,center[1]-normal[1]*width)]).buffer(.15))
    if caps:
        boundary = boundary.difference(union_all(caps))
    for i,shape in enumerate(line_parts(boundary)):
        yield f'pavement_boundary_{i}',shape,.12


def lane_dividers(net):
    paths = []
    for edge in net.getEdges(withInternal=False):
        for lane in edge.getLanes()[:-1]:
            paths.append((lane.getID(),move2side(lane.getShape(), -lane.getWidth()/2)))
    for node in net.getNodes():
        incoming = [e for e in node.getIncoming() if not e.getID().startswith(':')]
        outgoing = [e for e in node.getOutgoing() if not e.getID().startswith(':')]
        ring = any(e.getID().startswith('ring_') for e in incoming)
        if not ring and (len(incoming)!=2 or len(outgoing)!=2):
            continue
        # Same rule as the original two-lane map, generalised to N lanes: only through
        # geometry nodes (2 in / 2 out) and around the ring, continue the divider between
        # lane i and i+1 along the internal lane of the same-index connection i -> i.
        for source in incoming:
            lanes=source.getLaneNumber()
            if lanes<2 or (ring and not source.getID().startswith('ring_')):
                continue
            for target in source.getOutgoing():
                if target.getLaneNumber()!=lanes or target.getToNode()==source.getFromNode():
                    continue
                if ring and not target.getID().startswith('ring_'):
                    continue
                for index in range(lanes-1):
                    links=[c for c in source.getConnections(target)
                           if c.getFromLane().getIndex()==index and c.getToLane().getIndex()==index and c.getViaLaneID()]
                    if not links:
                        continue
                    lane=net.getLane(links[0].getViaLaneID())
                    paths.append((lane.getID(),move2side(lane.getShape(),-lane.getWidth()/2)))
    for name,path in paths:
        path = list(dict.fromkeys(path))
        if len(path) < 2:
            continue
        line=LineString(path)
        for i in range(math.ceil(line.length/6)):
            dash=substring(line,i*6,min(i*6+3,line.length))
            for shape in line_parts(dash):
                yield f'divider_{name}_{i}',shape,.12
