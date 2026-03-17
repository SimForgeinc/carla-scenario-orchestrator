from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import RuntimeRoadSectionSummary, RuntimeRoadSummary, SelectedRoad


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "maps.generated.json"


def normalize_map_name(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.replace("\\", "/").split("/")[-1].strip()
    if normalized.endswith(".xodr"):
        normalized = normalized[:-5]
    return normalized


@lru_cache(maxsize=1)
def _load_dataset() -> dict[str, Any]:
    with DEFAULT_DATASET_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_supported_maps() -> list[str]:
    dataset = _load_dataset()
    return [normalize_map_name(item.get("name")) for item in dataset.get("maps", [])]


def find_map_record(map_name: str) -> dict[str, Any] | None:
    target = normalize_map_name(map_name)
    dataset = _load_dataset()
    for item in dataset.get("maps", []):
        if normalize_map_name(item.get("name")) == target:
            return item
    return None


def build_selected_roads(map_name: str, road_ids: list[str]) -> list[SelectedRoad]:
    map_record = find_map_record(map_name)
    if not map_record:
        return []
    wanted = {str(road_id) for road_id in road_ids}
    roads: list[SelectedRoad] = []
    for road in map_record.get("roads", []):
        if str(road.get("id")) not in wanted:
            continue
        roads.append(
            SelectedRoad(
                id=str(road.get("id")),
                name=road.get("name") or f"Road {road.get('id')}",
                length=float(road.get("length") or 0.0),
                tags=[str(tag) for tag in road.get("tags", [])],
                section_labels=[str(section.get("label")) for section in road.get("sections", [])],
            )
        )
    return roads


def build_runtime_road_summaries(map_name: str) -> list[RuntimeRoadSummary]:
    map_record = find_map_record(map_name)
    if not map_record:
        return []

    summaries: list[RuntimeRoadSummary] = []
    for road in map_record.get("roads", []):
        section_summaries = [
            RuntimeRoadSectionSummary(
                index=int(section.get("index") or 0),
                label=str(section.get("label") or ""),
                s=float(section.get("s") or 0.0),
                driving_left=int(section.get("drivingLeft") or 0),
                driving_right=int(section.get("drivingRight") or 0),
                parking_left=int(section.get("parkingLeft") or 0),
                parking_right=int(section.get("parkingRight") or 0),
                total_driving=int(section.get("totalDriving") or 0),
                total_width=float(section.get("totalWidth") or 0.0),
                lane_types=[str(lane_type) for lane_type in section.get("laneTypes", [])],
                tags=[str(tag) for tag in section.get("tags", [])],
            )
            for section in road.get("sections", [])
        ]
        lane_types = sorted(
            {
                lane_type
                for section in section_summaries
                for lane_type in section.lane_types
            }
        )
        summaries.append(
            RuntimeRoadSummary(
                id=str(road.get("id")),
                name=str(road.get("name") or f"Road {road.get('id')}"),
                is_intersection=bool(road.get("isIntersection")),
                tags=[str(tag) for tag in road.get("tags", [])],
                lane_types=lane_types,
                has_parking="parking" in lane_types,
                has_shoulder="shoulder" in lane_types,
                has_sidewalk="sidewalk" in lane_types,
                section_summaries=section_summaries,
            )
        )
    return summaries


def dataset_lane_type_counts(map_name: str) -> dict[str, int]:
    map_record = find_map_record(map_name)
    if not map_record:
        return {}
    stats = map_record.get("stats", {})
    counts = stats.get("laneTypes", {})
    return {str(key): int(value) for key, value in counts.items()}


def _road_sections_for_search(road: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(section) for section in road.get("sections", [])]


def _road_lane_types(road: dict[str, Any]) -> set[str]:
    lane_types: set[str] = set()
    for section in _road_sections_for_search(road):
        lane_types.update(str(lane_type).lower() for lane_type in section.get("laneTypes", []))
    return lane_types


def _road_matches_query_text(road: dict[str, Any], query: str) -> bool:
    if not query.strip():
        return True
    haystack_parts = [
        str(road.get("id") or ""),
        str(road.get("name") or ""),
        " ".join(str(tag) for tag in road.get("tags", [])),
    ]
    for section in _road_sections_for_search(road):
        haystack_parts.append(str(section.get("label") or ""))
        haystack_parts.append(" ".join(str(tag) for tag in section.get("tags", [])))
        haystack_parts.append(" ".join(str(lane_type) for lane_type in section.get("laneTypes", [])))
    haystack = " ".join(haystack_parts).lower()
    return all(token in haystack for token in query.lower().split())


def _section_matches_filters(
    section: dict[str, Any],
    *,
    driving_left: int | None,
    driving_right: int | None,
    total_driving: int | None,
    parking_left_min: int | None,
    parking_right_min: int | None,
    require_parking_on_both_sides: bool | None,
) -> bool:
    if driving_left is not None and int(section.get("drivingLeft") or 0) != driving_left:
        return False
    if driving_right is not None and int(section.get("drivingRight") or 0) != driving_right:
        return False
    if total_driving is not None and int(section.get("totalDriving") or 0) != total_driving:
        return False
    if parking_left_min is not None and int(section.get("parkingLeft") or 0) < parking_left_min:
        return False
    if parking_right_min is not None and int(section.get("parkingRight") or 0) < parking_right_min:
        return False
    if require_parking_on_both_sides is True:
        if int(section.get("parkingLeft") or 0) <= 0 or int(section.get("parkingRight") or 0) <= 0:
            return False
    return True


def _road_matches_filters(
    road: dict[str, Any],
    *,
    query: str = "",
    tags: list[str] | None = None,
    lane_types: list[str] | None = None,
    is_intersection: bool | None = None,
    has_parking: bool | None = None,
    driving_left: int | None = None,
    driving_right: int | None = None,
    total_driving: int | None = None,
    parking_left_min: int | None = None,
    parking_right_min: int | None = None,
    require_parking_on_both_sides: bool | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    if not _road_matches_query_text(road, query):
        return False, []
    road_tags = {str(tag).lower() for tag in road.get("tags", [])}
    wanted_tags = {str(tag).lower() for tag in tags or [] if str(tag).strip()}
    if wanted_tags and not wanted_tags.issubset(road_tags):
        return False, []
    road_lane_types = _road_lane_types(road)
    wanted_lane_types = {str(lane_type).lower() for lane_type in lane_types or [] if str(lane_type).strip()}
    if wanted_lane_types and not wanted_lane_types.issubset(road_lane_types):
        return False, []
    if is_intersection is not None and bool(road.get("isIntersection")) != is_intersection:
        return False, []
    if has_parking is True and "parking" not in road_lane_types:
        return False, []
    if has_parking is False and "parking" in road_lane_types:
        return False, []

    matching_sections = [
        section
        for section in _road_sections_for_search(road)
        if _section_matches_filters(
            section,
            driving_left=driving_left,
            driving_right=driving_right,
            total_driving=total_driving,
            parking_left_min=parking_left_min,
            parking_right_min=parking_right_min,
            require_parking_on_both_sides=require_parking_on_both_sides,
        )
    ]
    if any(
        value is not None
        for value in (
            driving_left,
            driving_right,
            total_driving,
            parking_left_min,
            parking_right_min,
            require_parking_on_both_sides,
        )
    ) and not matching_sections:
        return False, []
    return True, matching_sections or _road_sections_for_search(road)


def _road_search_result(map_record: dict[str, Any], road: dict[str, Any], matching_sections: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "map_name": normalize_map_name(map_record.get("name")),
        "road_id": str(road.get("id")),
        "name": str(road.get("name") or f"Road {road.get('id')}"),
        "length": float(road.get("length") or 0.0),
        "is_intersection": bool(road.get("isIntersection")),
        "tags": [str(tag) for tag in road.get("tags", [])],
        "lane_types": sorted(_road_lane_types(road)),
        "matching_sections": [
            {
                "index": int(section.get("index") or 0),
                "label": str(section.get("label") or ""),
                "s": float(section.get("s") or 0.0),
                "driving_left": int(section.get("drivingLeft") or 0),
                "driving_right": int(section.get("drivingRight") or 0),
                "parking_left": int(section.get("parkingLeft") or 0),
                "parking_right": int(section.get("parkingRight") or 0),
                "total_driving": int(section.get("totalDriving") or 0),
                "lane_types": [str(lane_type) for lane_type in section.get("laneTypes", [])],
                "tags": [str(tag) for tag in section.get("tags", [])],
            }
            for section in matching_sections
        ],
    }


def search_roads(
    map_name: str,
    *,
    query: str = "",
    tags: list[str] | None = None,
    lane_types: list[str] | None = None,
    is_intersection: bool | None = None,
    has_parking: bool | None = None,
    driving_left: int | None = None,
    driving_right: int | None = None,
    total_driving: int | None = None,
    parking_left_min: int | None = None,
    parking_right_min: int | None = None,
    require_parking_on_both_sides: bool | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    map_record = find_map_record(map_name)
    if not map_record:
        return []
    results: list[dict[str, Any]] = []
    for road in map_record.get("roads", []):
        matches, matching_sections = _road_matches_filters(
            road,
            query=query,
            tags=tags,
            lane_types=lane_types,
            is_intersection=is_intersection,
            has_parking=has_parking,
            driving_left=driving_left,
            driving_right=driving_right,
            total_driving=total_driving,
            parking_left_min=parking_left_min,
            parking_right_min=parking_right_min,
            require_parking_on_both_sides=require_parking_on_both_sides,
        )
        if not matches:
            continue
        results.append(_road_search_result(map_record, road, matching_sections))
        if len(results) >= max(1, limit):
            break
    return results


def search_maps_by_road(
    *,
    query: str = "",
    tags: list[str] | None = None,
    lane_types: list[str] | None = None,
    is_intersection: bool | None = None,
    has_parking: bool | None = None,
    driving_left: int | None = None,
    driving_right: int | None = None,
    total_driving: int | None = None,
    parking_left_min: int | None = None,
    parking_right_min: int | None = None,
    require_parking_on_both_sides: bool | None = None,
    map_limit: int = 8,
    roads_per_map_limit: int = 5,
) -> list[dict[str, Any]]:
    dataset = _load_dataset()
    results: list[dict[str, Any]] = []
    for map_record in dataset.get("maps", []):
        road_hits: list[dict[str, Any]] = []
        for road in map_record.get("roads", []):
            matches, matching_sections = _road_matches_filters(
                road,
                query=query,
                tags=tags,
                lane_types=lane_types,
                is_intersection=is_intersection,
                has_parking=has_parking,
                driving_left=driving_left,
                driving_right=driving_right,
                total_driving=total_driving,
                parking_left_min=parking_left_min,
                parking_right_min=parking_right_min,
                require_parking_on_both_sides=require_parking_on_both_sides,
            )
            if not matches:
                continue
            road_hits.append(_road_search_result(map_record, road, matching_sections))
            if len(road_hits) >= max(1, roads_per_map_limit):
                break
        if road_hits:
            results.append(
                {
                    "map_name": normalize_map_name(map_record.get("name")),
                    "road_count": len(road_hits),
                    "roads": road_hits,
                }
            )
            if len(results) >= max(1, map_limit):
                break
    return results
