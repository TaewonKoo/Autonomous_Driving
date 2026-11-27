"""Remove road-speed restrictions without changing geometry or priorities."""
from pathlib import Path
import xml.etree.ElementTree as ET


def remove_road_speed_limits(network, ceiling, edge_file=None):
    """SUMO requires lane speed values: use a ceiling >= vehicle maxima."""
    ceiling=float(ceiling)
    if ceiling <= 0:
        raise ValueError("SUMO requires positive lane speed")
    path=Path(network);tree=ET.parse(path)
    counts={"lanes":0,"connections":0,"edge_definitions":0}
    for lane in tree.getroot().iter("lane"):
        lane.set("speed",str(ceiling));counts["lanes"]+=1
    for connection in tree.getroot().iter("connection"):
        if "speed" in connection.attrib:
            connection.set("speed",str(ceiling));counts["connections"]+=1
    ET.indent(tree);tree.write(path,encoding="utf-8",xml_declaration=True)
    if edge_file is not None:
        path=Path(edge_file);tree=ET.parse(path)
        for edge in tree.getroot().iter("edge"):
            edge.set("speed",str(ceiling));counts["edge_definitions"]+=1
        ET.indent(tree);tree.write(path,encoding="utf-8",xml_declaration=True)
    return counts


def remove_speed_signs(preview):
    """Remove generated school speed-sign symbols, preserving other art."""
    path=Path(preview);tree=ET.parse(path);removed=0
    for parent in tree.getroot().iter():
        for child in list(parent):
            tag=child.tag.rsplit('}',1)[-1]
            circle=(tag=='circle' and child.get('stroke')=='#d63b30' and child.get('r')=='14')
            text=(tag=='text' and child.text=='30' and child.get('text-anchor')=='middle' and child.get('font-size')=='14')
            if circle or text:
                parent.remove(child);removed+=1
    tree.write(path,encoding='utf-8',xml_declaration=True)
    return removed
