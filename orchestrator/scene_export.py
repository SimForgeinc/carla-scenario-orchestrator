"""Render scene setup as SVG/PNG for visual validation."""
from __future__ import annotations
import math
from typing import Any


def _get_actor_world_positions(service, actors: list[dict], map_name: str) -> list[dict]:
    """Get CARLA world x,y,yaw for each actor using road_position_to_world."""
    positioned = []
    meta = service.carla_metadata
    if meta is None:
        return positioned

    for actor in actors:
        spawn = actor.get("spawn", {})
        road_id = spawn.get("road_id")
        s_fraction = spawn.get("s_fraction", 0.5)
        lane_id = spawn.get("lane_id", -1)

        # Try spawn_point first (for point-placement actors)
        if actor.get("spawn_point"):
            positioned.append({
                **actor,
                "world_x": actor["spawn_point"]["x"],
                "world_y": actor["spawn_point"]["y"],
                "world_yaw": actor.get("spawn_yaw", 0),
            })
            continue

        if road_id is None:
            continue

        # Try requested lane first, then fallback lanes
        wp = None
        for try_lane in [lane_id, -1, 1, -2, 2]:
            try:
                wp = meta.road_position_to_world(int(road_id), s_fraction, try_lane)
                if wp:
                    break
            except Exception:
                continue

        if wp:
            positioned.append({
                **actor,
                "world_x": wp["x"],
                "world_y": wp["y"],
                "world_yaw": wp.get("yaw", 0),
            })

    return positioned


def _get_road_paths(service, selected_road_ids: list[str]) -> list[dict]:
    """Get SVG path data for selected roads from generated map."""
    try:
        gen_map = service.get_generated_map()
        if not gen_map:
            return []
        roads = gen_map.get("roads", [])
        paths = []
        for road in roads:
            if road.get("id") in selected_road_ids:
                paths.append({
                    "id": road["id"],
                    "name": road.get("name", ""),
                    "path": road.get("path", ""),
                    "surface": road.get("drivingSurface", road.get("surface", "")),
                    "bounds": road.get("bounds", {}),
                    "is_intersection": road.get("isIntersection", False),
                })
        return paths
    except Exception:
        return []


def render_scene_svg(service, actors: list[dict], selected_roads: list[dict], map_name: str) -> str:
    """Render scene as SVG string with actors and roads."""
    road_ids = [r.get("id", "") for r in selected_roads]
    positioned_actors = _get_actor_world_positions(service, actors, map_name)
    road_paths = _get_road_paths(service, road_ids)

    if not positioned_actors:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200"><rect width="400" height="200" fill="#1a1a2e"/><text x="10" y="100" fill="red" font-size="16" font-family="sans-serif">No actors with valid positions</text></svg>'

    # Compute bounding box around actors with margin
    xs = [a["world_x"] for a in positioned_actors]
    ys = [a["world_y"] for a in positioned_actors]

    margin = 40  # meters of padding
    min_x = min(xs) - margin
    max_x = max(xs) + margin
    min_y = min(ys) - margin
    max_y = max(ys) + margin

    # Ensure minimum size
    vw = max(max_x - min_x, 80)
    vh = max(max_y - min_y, 80)

    # Fixed 800px output on the larger dimension
    if vw >= vh:
        svg_w = 800
        svg_h = max(400, int(800 * vh / vw))
    else:
        svg_h = 800
        svg_w = max(400, int(800 * vw / vh))

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="{min_x} {min_y} {vw} {vh}">')

    # Background
    parts.append(f'<rect x="{min_x}" y="{min_y}" width="{vw}" height="{vh}" fill="#1a1a2e"/>')

    # Road surfaces
    if road_paths:
        parts.append('<g opacity="0.5">')
        for rp in road_paths:
            surface = rp.get("surface", "")
            if surface:
                parts.append(f'<path d="{surface}" fill="#334155" stroke="none"/>')
            path = rp.get("path", "")
            if path:
                parts.append(f'<path d="{path}" fill="none" stroke="#64748b" stroke-width="0.8"/>')
        parts.append('</g>')

    # Grid
    grid_step = 20
    parts.append(f'<g stroke="#ffffff10" stroke-width="0.2">')
    gx = math.floor(min_x / grid_step) * grid_step
    while gx <= max_x:
        parts.append(f'<line x1="{gx}" y1="{min_y}" x2="{gx}" y2="{max_y}"/>')
        gx += grid_step
    gy = math.floor(min_y / grid_step) * grid_step
    while gy <= max_y:
        parts.append(f'<line x1="{min_x}" y1="{gy}" x2="{max_x}" y2="{gy}"/>')
        gy += grid_step
    parts.append('</g>')

    # Font size relative to view
    font_size = max(2.5, min(vw, vh) / 40)

    # Actors
    for actor in positioned_actors:
        ax = actor["world_x"]
        ay = actor["world_y"]
        yaw = actor.get("world_yaw", 0)
        kind = actor.get("kind", "vehicle")
        role = actor.get("role", "traffic")
        label = actor.get("label", "?")
        is_static = actor.get("is_static", False)

        # Color by role
        if role == "ego":
            color = "#22d3ee"  # cyan
            outline = "#06b6d4"
        elif kind == "walker":
            color = "#f59e0b"  # amber
            outline = "#d97706"
        elif is_static:
            color = "#94a3b8"  # gray
            outline = "#64748b"
        else:
            color = "#f43f5e"  # red
            outline = "#e11d48"

        # Size by kind
        if kind == "walker":
            w, h = 1.2, 1.2
        elif "bus" in actor.get("blueprint", "") or "truck" in actor.get("blueprint", ""):
            w, h = 3.0, 9.0
        else:
            w, h = 2.2, 4.8

        # Render as rotated rectangle
        parts.append(f'<g transform="translate({ax},{ay}) rotate({-yaw})">')
        parts.append(f'<rect x="{-w/2}" y="{-h/2}" width="{w}" height="{h}" fill="{color}" stroke="{outline}" stroke-width="0.4" rx="0.6"/>')
        # Direction indicator (front of vehicle)
        if kind == "vehicle":
            parts.append(f'<polygon points="0,{-h/2 - 0.3} {-w/3},{-h/2 + 0.8} {w/3},{-h/2 + 0.8}" fill="white" opacity="0.8"/>')
        parts.append('</g>')

        # Label with background
        parts.append(f'<text x="{ax}" y="{ay + h/2 + font_size + 1.5}" text-anchor="middle" fill="white" font-size="{font_size}" font-family="sans-serif" font-weight="bold">{label}</text>')

    # Title / legend
    parts.append(f'<text x="{min_x + 3}" y="{min_y + font_size + 2}" fill="#94a3b8" font-size="{font_size * 0.9}" font-family="sans-serif">{map_name} | {len(positioned_actors)} actors</text>')

    # Color legend
    ly = min_y + font_size * 2 + 6
    for lbl, clr in [("Ego", "#22d3ee"), ("Traffic", "#f43f5e"), ("Pedestrian", "#f59e0b"), ("Static", "#94a3b8")]:
        parts.append(f'<rect x="{min_x + 3}" y="{ly - font_size * 0.6}" width="{font_size}" height="{font_size * 0.7}" fill="{clr}" rx="0.3"/>')
        parts.append(f'<text x="{min_x + font_size + 5}" y="{ly}" fill="#cbd5e1" font-size="{font_size * 0.7}" font-family="sans-serif">{lbl}</text>')
        ly += font_size + 2

    parts.append('</svg>')
    return '\n'.join(parts)


def render_scene_png(service, actors: list[dict], selected_roads: list[dict], map_name: str) -> bytes | None:
    """Render scene as PNG bytes. Returns None if cairosvg is not available."""
    try:
        import cairosvg
    except ImportError:
        return None

    svg_str = render_scene_svg(service, actors, selected_roads, map_name)
    return cairosvg.svg2png(bytestring=svg_str.encode(), output_width=1200)
