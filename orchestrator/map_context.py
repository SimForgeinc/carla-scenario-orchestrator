"""Query Aurora for map asset context by CARLA map name."""
from __future__ import annotations

import json
import os
from typing import Any

import boto3


def _get_rds_client():
    region = os.environ.get("AURORA_REGION", "us-east-1")
    return boto3.client("rds-data", region_name=region)


def _rds_params():
    return {
        "resourceArn": os.environ.get("AURORA_CLUSTER_ARN", ""),
        "secretArn": os.environ.get("AURORA_SECRET_ARN", ""),
        "database": os.environ.get("AURORA_DATABASE", "simcloud"),
    }


def _field_value(field: dict) -> Any:
    if field.get("isNull"):
        return None
    if "stringValue" in field:
        return field["stringValue"]
    if "longValue" in field:
        return field["longValue"]
    if "doubleValue" in field:
        return field["doubleValue"]
    if "booleanValue" in field:
        return field["booleanValue"]
    return None


def get_map_context(carla_map_name: str) -> dict[str, Any] | None:
    """Fetch map asset context (description, tags, enrichments) for a CARLA map name."""
    client = _get_rds_client()
    params = _rds_params()

    result = client.execute_statement(
        **params,
        includeResultMetadata=True,
        sql="""
            SELECT 
                ma.id AS map_asset_id,
                ma.name,
                ma.description,
                ma.carla_map_name,
                ma.tags::text AS tags_json,
                ma.place_context::text AS place_context_json,
                mae.summary_json::text AS enrichment_summary_json,
                mae.candidate_locations_json::text AS candidate_locations_json
            FROM map_assets ma
            LEFT JOIN map_asset_enrichments mae ON ma.id = mae.map_asset_id
            WHERE ma.carla_map_name = :carla_map_name AND ma.is_active = true
            LIMIT 1
        """,
        parameters=[
            {"name": "carla_map_name", "value": {"stringValue": carla_map_name}},
        ],
    )

    records = result.get("records", [])
    if not records:
        return None

    columns = result.get("columnMetadata", [])
    col_names = [c["name"] for c in columns]
    row = records[0]

    data: dict[str, Any] = {}
    for col_name, field in zip(col_names, row):
        data[col_name] = _field_value(field)

    # Parse JSON fields
    def parse_json(key: str, default: Any = None) -> Any:
        raw = data.get(key)
        if not raw:
            return default
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return default

    return {
        "map_asset_id": data.get("map_asset_id"),
        "name": data.get("name"),
        "description": data.get("description"),
        "carla_map_name": data.get("carla_map_name"),
        "place_context": parse_json("place_context_json", {}),
        "tags": parse_json("tags_json", []),
        "enrichment": parse_json("enrichment_summary_json", {}),
        "candidate_locations": (parse_json("candidate_locations_json", []) or [])[:10],
    }


def search_maps(tags: list[str] | None = None, query: str | None = None) -> list[dict[str, Any]]:
    """Search all active map assets by tags or free-text query on name/description."""
    client = _get_rds_client()
    params = _rds_params()

    conditions = ["ma.is_active = true"]
    sql_params: list[dict] = []

    if tags:
        # JSONB array containment: tags column must contain ALL specified tags
        for i, tag in enumerate(tags):
            conditions.append(f"ma.tags::jsonb @> :tag_{i}::jsonb")
            sql_params.append({"name": f"tag_{i}", "value": {"stringValue": json.dumps([tag])}})

    if query:
        conditions.append("(LOWER(ma.name) LIKE :query OR LOWER(ma.description) LIKE :query)")
        sql_params.append({"name": "query", "value": {"stringValue": f"%{query.lower()}%"}})

    where = " AND ".join(conditions)

    result = client.execute_statement(
        **params,
        includeResultMetadata=True,
        sql=f"""
            SELECT
                ma.id AS map_asset_id,
                ma.name,
                ma.description,
                ma.carla_map_name,
                ma.tags::text AS tags_json,
                ma.place_context::text AS place_context_json
            FROM map_assets ma
            WHERE {where}
            ORDER BY ma.name
            LIMIT 20
        """,
        parameters=sql_params,
    )

    columns = result.get("columnMetadata", [])
    col_names = [c["name"] for c in columns]
    maps = []
    for row in result.get("records", []):
        data: dict[str, Any] = {}
        for col_name, field in zip(col_names, row):
            data[col_name] = _field_value(field)

        def parse_json(key: str, default: Any = None) -> Any:
            raw = data.get(key)
            if not raw:
                return default
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return default

        maps.append({
            "map_asset_id": data.get("map_asset_id"),
            "name": data.get("name"),
            "carla_map_name": data.get("carla_map_name"),
            "description": (data.get("description") or "")[:200],
            "tags": parse_json("tags_json", []),
            "place_context": parse_json("place_context_json", {}),
        })

    return maps


def _latlon_to_carla(lat: float, lon: float, coord_ref: dict) -> tuple[float, float] | None:
    """Convert WGS84 lat/lon to CARLA local x,y using the map's projection."""
    import math
    origin_lat = coord_ref.get("origin_lat")
    origin_lon = coord_ref.get("origin_lon")
    if origin_lat is None or origin_lon is None:
        return None
    # Simplified Transverse Mercator approximation (accurate within ~1km of origin)
    R = 6378137.0  # Earth radius in meters
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    olat_rad = math.radians(origin_lat)
    olon_rad = math.radians(origin_lon)
    # x = easting (east-west offset from origin)
    x = R * (lon_rad - olon_rad) * math.cos(olat_rad)
    # y = northing (north-south offset from origin) — CARLA uses -y for south
    y = R * (lat_rad - olat_rad)
    # CARLA coordinate system: x=east, y=-north (or y=south)
    # OpenDRIVE typically: x=east, y=north
    # The actual mapping depends on the XODR export. Most SimForge maps use x=east, y=north
    # so we keep y as northing. The CARLA rendering flips y for display.
    return (x, y)


def get_candidate_locations_with_carla_coords(carla_map_name: str) -> list[dict]:
    """Get candidate locations with CARLA x,y coordinates projected from lat/lng."""
    client = _get_rds_client()
    params = _rds_params()

    result = client.execute_statement(
        **params,
        includeResultMetadata=True,
        sql="""
            SELECT
                ma.map_coordinate_ref::text AS coord_ref_json,
                mae.candidate_locations_json::text AS candidate_locations_json
            FROM map_assets ma
            LEFT JOIN map_asset_enrichments mae ON ma.id = mae.map_asset_id
            WHERE ma.carla_map_name = :carla_map_name AND ma.is_active = true
            LIMIT 1
        """,
        parameters=[
            {"name": "carla_map_name", "value": {"stringValue": carla_map_name}},
        ],
    )

    records = result.get("records", [])
    if not records:
        return []

    row = records[0]
    coord_ref_raw = row[0].get("stringValue") if not row[0].get("isNull") else None
    candidates_raw = row[1].get("stringValue") if not row[1].get("isNull") else None

    coord_ref = json.loads(coord_ref_raw) if coord_ref_raw else {}
    candidates = json.loads(candidates_raw) if candidates_raw else []

    if not candidates or not coord_ref:
        return candidates or []

    # Project each candidate's bbox center to CARLA coordinates
    for loc in candidates:
        region = loc.get("region", {})
        bbox = region.get("bbox", {})
        if bbox:
            center_lat = (bbox.get("min_lat", 0) + bbox.get("max_lat", 0)) / 2
            center_lng = (bbox.get("min_lng", 0) + bbox.get("max_lng", 0)) / 2
            carla_xy = _latlon_to_carla(center_lat, center_lng, coord_ref)
            if carla_xy:
                loc["carla_x"] = round(carla_xy[0], 1)
                loc["carla_y"] = round(carla_xy[1], 1)

    return candidates
