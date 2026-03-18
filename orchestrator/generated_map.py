from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Any


def _as_float(value: str | None, fallback: float = 0.0) -> float:
    try:
        return float(value) if value is not None else fallback
    except (TypeError, ValueError):
        return fallback


def _point_bounds(points: list[tuple[float, float]]) -> dict[str, float]:
    if not points:
        return {"minX": 0.0, "minY": 0.0, "maxX": 0.0, "maxY": 0.0, "width": 0.0, "height": 0.0}
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return {
        "minX": min_x,
        "minY": min_y,
        "maxX": max_x,
        "maxY": max_y,
        "width": max_x - min_x,
        "height": max_y - min_y,
    }


def _merge_bounds(bounds_list: list[dict[str, float]]) -> dict[str, float]:
    filtered = [bounds for bounds in bounds_list if bounds["width"] or bounds["height"]]
    if not filtered:
        return {"minX": 0.0, "minY": 0.0, "maxX": 0.0, "maxY": 0.0, "width": 0.0, "height": 0.0}
    min_x = min(bounds["minX"] for bounds in filtered)
    min_y = min(bounds["minY"] for bounds in filtered)
    max_x = max(bounds["maxX"] for bounds in filtered)
    max_y = max(bounds["maxY"] for bounds in filtered)
    return {
        "minX": min_x,
        "minY": min_y,
        "maxX": max_x,
        "maxY": max_y,
        "width": max_x - min_x,
        "height": max_y - min_y,
    }


def _sample_line_with_s(x0: float, y0: float, heading: float, length: float, steps: int, s_start: float) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for index in range(steps + 1):
        ds = (length * index) / steps
        points.append((x0 + ds * math.cos(heading), -(y0 + ds * math.sin(heading)), s_start + ds))
    return points


def _sample_arc_with_s(x0: float, y0: float, heading: float, length: float, curvature: float, steps: int, s_start: float) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for index in range(steps + 1):
        ds = (length * index) / steps
        x = x0 + (math.sin(heading + curvature * ds) - math.sin(heading)) / curvature
        y = y0 - (math.cos(heading + curvature * ds) - math.cos(heading)) / curvature
        points.append((x, -y, s_start + ds))
    return points


def _to_svg_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    segments: list[str] = []
    for index, (x, y) in enumerate(points):
        segments.append(f"{'M' if index == 0 else 'L'}{x:.1f} {y:.1f}")
    return " ".join(segments)


def _to_svg_polygon(left_edge: list[tuple[float, float]], right_edge: list[tuple[float, float]]) -> str:
    if not left_edge or not right_edge:
        return ""
    reversed_right = list(reversed(right_edge))
    return _to_svg_path(left_edge) + " " + " ".join(f"L{x:.1f} {y:.1f}" for x, y in reversed_right) + " Z"


def _sample_road_points(road: ET.Element) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    plan_view = road.find("planView")
    geometries = plan_view.findall("geometry") if plan_view is not None else []
    for geometry in geometries:
        length = _as_float(geometry.get("length"))
        steps = max(2, math.ceil(length / 6.0))
        x0 = _as_float(geometry.get("x"))
        y0 = _as_float(geometry.get("y"))
        heading = _as_float(geometry.get("hdg"))
        s_start = _as_float(geometry.get("s"))
        arc = geometry.find("arc")
        if arc is not None and abs(_as_float(arc.get("curvature"))) >= 1e-9:
            sampled = _sample_arc_with_s(x0, y0, heading, length, _as_float(arc.get("curvature")), steps, s_start)
        else:
            sampled = _sample_line_with_s(x0, y0, heading, length, steps, s_start)
        if points and sampled:
            sampled = sampled[1:]
        points.extend(sampled)
    return points


def _compute_normals(points: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    normals: list[tuple[float, float]] = []
    if len(points) < 2:
        return normals
    for index in range(len(points)):
        if index == 0:
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
        elif index == len(points) - 1:
            dx = points[index][0] - points[index - 1][0]
            dy = points[index][1] - points[index - 1][1]
        else:
            dx = points[index + 1][0] - points[index - 1][0]
            dy = points[index + 1][1] - points[index - 1][1]
        length = math.sqrt(dx * dx + dy * dy) or 1.0
        normals.append((dy / length, -dx / length))
    return normals


def _offset_points(points: list[tuple[float, float, float]], normals: list[tuple[float, float]], distance: float) -> list[tuple[float, float]]:
    return [
        (point[0] + normals[index][0] * distance, point[1] + normals[index][1] * distance)
        for index, point in enumerate(points)
    ]


def _width_at_start(lane: ET.Element) -> float:
    width = lane.find("width")
    return max(0.0, _as_float(width.get("a") if width is not None else None))


def _lane_side_summary(lanes: list[ET.Element]) -> dict[str, Any]:
    summary = {"driving": 0, "parking": 0, "totalWidth": 0.0, "types": set()}
    for lane in lanes:
        lane_type = lane.get("type", "unknown")
        summary["types"].add(lane_type)
        summary["totalWidth"] += _width_at_start(lane)
        if lane_type == "driving":
            summary["driving"] += 1
        if lane_type == "parking":
            summary["parking"] += 1
    return summary


def _classify_section(section: ET.Element, index: int) -> dict[str, Any]:
    left_lanes = [lane for lane in section.findall("./left/lane")]
    right_lanes = [lane for lane in section.findall("./right/lane")]
    left = _lane_side_summary(left_lanes)
    right = _lane_side_summary(right_lanes)
    total_driving = left["driving"] + right["driving"]
    total_width = left["totalWidth"] + right["totalWidth"]
    lane_types = sorted(set(left["types"]) | set(right["types"]))
    tags: set[str] = set()
    if total_driving == 1:
        tags.add("single_lane_road")
    if left["driving"] == 1 and right["driving"] == 1:
        tags.add("single_lane_each_way")
    if (left["driving"] == 2 and right["driving"] == 0) or (left["driving"] == 0 and right["driving"] == 2):
        tags.add("two_lane_one_way")
    if left["driving"] == 2 and right["driving"] == 2:
        tags.add("two_lane_each_way")
    if left["parking"] > 0 or right["parking"] > 0:
        tags.add("parking")
    return {
        "index": index,
        "label": f"{left['driving']}L / {right['driving']}R",
        "s": _as_float(section.get("s")),
        "drivingLeft": left["driving"],
        "drivingRight": right["driving"],
        "parkingLeft": left["parking"],
        "parkingRight": right["parking"],
        "totalDriving": total_driving,
        "totalWidth": total_width,
        "laneTypes": lane_types,
        "tags": sorted(tags),
    }


def _feature_tags_for_object(name: str) -> list[str]:
    lower_name = (name or "").lower()
    tags: set[str] = set()
    if "crosswalk" in lower_name:
        tags.add("crosswalk")
    if "stop" in lower_name:
        tags.add("stop_control")
    return sorted(tags)


def _build_lane_geometry(points: list[tuple[float, float, float]], normals: list[tuple[float, float]], road: ET.Element) -> dict[str, Any] | None:
    if len(points) < 2:
        return None
    lane_sections = road.findall("./lanes/laneSection")
    if not lane_sections:
        return None
    section = lane_sections[0]
    left_lanes = [lane for lane in section.findall("./left/lane") if _as_float(lane.get("id")) > 0]
    left_lanes.sort(key=lambda lane: _as_float(lane.get("id")))
    right_lanes = [lane for lane in section.findall("./right/lane") if _as_float(lane.get("id")) < 0]
    right_lanes.sort(key=lambda lane: _as_float(lane.get("id")), reverse=True)

    left_total = 0.0
    for lane in left_lanes:
        left_total += _width_at_start(lane)
    right_total = 0.0
    for lane in right_lanes:
        right_total += _width_at_start(lane)

    left_edge = _offset_points(points, normals, left_total)
    right_edge = _offset_points(points, normals, -right_total)
    surface = _to_svg_polygon(left_edge, right_edge)

    left_driving_width = 0.0
    for lane in left_lanes:
        lane_type = lane.get("type", "unknown")
        if lane_type in {"driving", "shoulder"}:
            left_driving_width += _width_at_start(lane)
        else:
            break
    right_driving_width = 0.0
    for lane in right_lanes:
        lane_type = lane.get("type", "unknown")
        if lane_type in {"driving", "shoulder"}:
            right_driving_width += _width_at_start(lane)
        else:
            break

    driving_left_edge = _offset_points(points, normals, left_driving_width)
    driving_right_edge = _offset_points(points, normals, -right_driving_width)
    driving_surface = _to_svg_polygon(driving_left_edge, driving_right_edge)
    lane_lines: list[dict[str, Any]] = []

    has_left_driving = any(lane.get("type", "unknown") == "driving" for lane in left_lanes)
    has_right_driving = any(lane.get("type", "unknown") == "driving" for lane in right_lanes)
    if has_left_driving and has_right_driving:
        lane_lines.append({"path": _to_svg_path([(point[0], point[1]) for point in points]), "dashed": False, "type": "center"})

    left_accum = 0.0
    for index in range(len(left_lanes) - 1):
        left_accum += _width_at_start(left_lanes[index])
        current_type = left_lanes[index].get("type", "unknown")
        next_type = left_lanes[index + 1].get("type", "unknown")
        if current_type in {"driving", "parking"} and next_type in {"driving", "parking"}:
            lane_lines.append({"path": _to_svg_path(_offset_points(points, normals, left_accum)), "dashed": True, "type": "divider"})

    right_accum = 0.0
    for index in range(len(right_lanes) - 1):
        right_accum += _width_at_start(right_lanes[index])
        current_type = right_lanes[index].get("type", "unknown")
        next_type = right_lanes[index + 1].get("type", "unknown")
        if current_type in {"driving", "parking"} and next_type in {"driving", "parking"}:
            lane_lines.append({"path": _to_svg_path(_offset_points(points, normals, -right_accum)), "dashed": True, "type": "divider"})

    lane_lines.append({"path": _to_svg_path(driving_left_edge), "dashed": False, "type": "edge"})
    lane_lines.append({"path": _to_svg_path(driving_right_edge), "dashed": False, "type": "edge"})
    return {"surface": surface, "drivingSurface": driving_surface, "laneLines": lane_lines}


def _eval_road_at(road: ET.Element, target_s: float) -> dict[str, float]:
    geometries = road.findall("./planView/geometry")
    geometry = geometries[0] if geometries else None
    for candidate in geometries:
        if _as_float(candidate.get("s")) <= target_s:
            geometry = candidate
    if geometry is None:
        return {"x": 0.0, "y": 0.0, "heading": 0.0}
    s_start = _as_float(geometry.get("s"))
    ds = target_s - s_start
    x0 = _as_float(geometry.get("x"))
    y0 = _as_float(geometry.get("y"))
    heading = _as_float(geometry.get("hdg"))
    arc = geometry.find("arc")
    if arc is not None and abs(_as_float(arc.get("curvature"))) >= 1e-9:
        curvature = _as_float(arc.get("curvature"))
        x = x0 + (math.sin(heading + curvature * ds) - math.sin(heading)) / curvature
        y = y0 - (math.cos(heading + curvature * ds) - math.cos(heading)) / curvature
        return {"x": x, "y": y, "heading": heading + curvature * ds}
    return {"x": x0 + ds * math.cos(heading), "y": y0 + ds * math.sin(heading), "heading": heading}


def _transform_local_to_world(u: float, v: float, ref_x: float, ref_y: float, ref_hdg: float, obj_hdg: float) -> tuple[float, float]:
    total_hdg = ref_hdg + obj_hdg
    cos_h = math.cos(total_hdg)
    sin_h = math.sin(total_hdg)
    world_x = ref_x + u * cos_h - v * sin_h
    world_y = ref_y + u * sin_h + v * cos_h
    return (world_x, -world_y)


def _build_crosswalk_polygon(obj: ET.Element, road: ET.Element) -> dict[str, str] | None:
    corners = obj.findall("./outline/cornerLocal")
    if len(corners) < 3:
        return None
    ref = _eval_road_at(road, _as_float(obj.get("s")))
    lateral = _as_float(obj.get("t"))
    ref_x = ref["x"] + lateral * (-math.sin(ref["heading"]))
    ref_y = ref["y"] + lateral * math.cos(ref["heading"])
    obj_hdg = _as_float(obj.get("hdg"))
    world_corners = [
        _transform_local_to_world(_as_float(corner.get("u")), _as_float(corner.get("v")), ref_x, ref_y, ref["heading"], obj_hdg)
        for corner in corners
    ]
    path = " ".join(f"{'M' if index == 0 else 'L'}{x:.1f} {y:.1f}" for index, (x, y) in enumerate(world_corners)) + " Z"
    return {"path": path}


def _build_stop_marker(obj: ET.Element, road: ET.Element) -> dict[str, float]:
    ref = _eval_road_at(road, _as_float(obj.get("s")))
    lateral = _as_float(obj.get("t"))
    world_x = ref["x"] + lateral * (-math.sin(ref["heading"]))
    world_y = ref["y"] + lateral * math.cos(ref["heading"])
    return {"x": round(world_x, 1), "y": round(-world_y, 1)}


def _build_road(road: ET.Element) -> dict[str, Any]:
    points = _sample_road_points(road)
    xy_points = [(point[0], point[1]) for point in points]
    normals = _compute_normals(points)
    sections = [_classify_section(section, index) for index, section in enumerate(road.findall("./lanes/laneSection"))]
    objects = []
    for obj in road.findall("./objects/object"):
        objects.append({
            "id": str(obj.get("id", "")),
            "name": str(obj.get("name", "")),
            "s": _as_float(obj.get("s")),
            "t": _as_float(obj.get("t")),
            "tags": _feature_tags_for_object(str(obj.get("name", ""))),
        })
    tags: set[str] = set()
    if str(road.get("junction", "-1")) != "-1":
        tags.add("intersection")
    for section in sections:
        tags.update(section["tags"])
    for obj in objects:
        tags.update(obj["tags"])
    lane_geo = _build_lane_geometry(points, normals, road)
    return {
        "id": str(road.get("id", "")),
        "name": str(road.get("name", "")),
        "junctionId": str(road.get("junction", "-1")),
        "isIntersection": str(road.get("junction", "-1")) != "-1",
        "length": _as_float(road.get("length")),
        "path": _to_svg_path(xy_points),
        "surface": lane_geo["surface"] if lane_geo else "",
        "drivingSurface": lane_geo["drivingSurface"] if lane_geo else "",
        "laneLines": lane_geo["laneLines"] if lane_geo else [],
        "bounds": _point_bounds(xy_points),
        "tags": sorted(tags),
        "sections": sections,
        "objects": objects,
    }


def _lane_type_counts(roads: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for road in roads:
        for section in road["sections"]:
            for lane_type in section["laneTypes"]:
                counts[lane_type] = counts.get(lane_type, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _feature_counts(roads: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "intersection": 0,
        "parking": 0,
        "single_lane_road": 0,
        "single_lane_each_way": 0,
        "two_lane_one_way": 0,
        "two_lane_each_way": 0,
        "crosswalk": 0,
        "stop_control": 0,
    }
    for road in roads:
        for tag in road["tags"]:
            if tag in counts:
                counts[tag] += 1
    return counts


def build_generated_map(map_name: str, xodr_text: str) -> dict[str, Any]:
    root = ET.fromstring(xodr_text)
    raw_roads = root.findall("road")
    roads = [_build_road(road) for road in raw_roads]
    crosswalks: list[dict[str, str]] = []
    stop_markers: list[dict[str, float]] = []
    for raw_road in raw_roads:
        for obj in raw_road.findall("./objects/object"):
            name = str(obj.get("name", "")).lower()
            if "crosswalk" in name and obj.find("outline") is not None:
                polygon = _build_crosswalk_polygon(obj, raw_road)
                if polygon is not None:
                    crosswalks.append(polygon)
            if "stop" in name:
                stop_markers.append(_build_stop_marker(obj, raw_road))
    counts = _feature_counts(roads)
    counts["crosswalk"] = len(crosswalks)
    counts["stop_control"] = len(stop_markers)
    return {
        "name": map_name,
        "fileName": f"{map_name}.xodr",
        "optimized": "_Opt" in map_name,
        "bounds": _merge_bounds([road["bounds"] for road in roads]),
        "stats": {
            "roads": len(roads),
            "junctionDefinitions": len(root.findall("junction")),
            "laneTypes": _lane_type_counts(roads),
            "featureCounts": counts,
        },
        "roads": roads,
        "crosswalks": crosswalks,
        "stopMarkers": stop_markers,
    }
