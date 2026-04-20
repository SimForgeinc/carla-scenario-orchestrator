from __future__ import annotations

from dataclasses import dataclass

from .models import RenderOutputSpec, SensorConfig

ENCODABLE_SENSOR_CATEGORIES = {"camera", "lidar"}
GT_SENSOR_NAME_TO_MODALITY = {
    "semantic_seg": "semantic_segmentation",
    "depth": "depth",
    "instance_seg": "instance_segmentation",
}
GT_MODALITY_TO_SENSOR_NAME = {
    value: key for key, value in GT_SENSOR_NAME_TO_MODALITY.items()
}

PROFILE_PRESETS: dict[str, dict[str, list[str]]] = {
    "playback": {
        "modalities": [],
        "annotations": [],
        "metadata": ["manifest"],
        "encodings": ["mp4"],
    },
    "training_basic": {
        "modalities": [],
        "annotations": ["bbox_2d"],
        "metadata": ["manifest", "calibration", "timestamps"],
        "encodings": ["image_sequence"],
    },
    "training_multimodal": {
        "modalities": [],
        "annotations": ["bbox_2d", "bbox_3d", "tracking"],
        "metadata": ["manifest", "calibration", "timestamps", "opendrive"],
        "encodings": ["image_sequence", "mp4"],
    },
    "raw_multisensor": {
        "modalities": [],
        "annotations": [],
        "metadata": ["manifest", "calibration", "timestamps"],
        "encodings": ["image_sequence"],
    },
    "tao_detection": {
        "modalities": ["rgb"],
        "annotations": ["bbox_2d", "tracking"],
        "metadata": ["manifest", "calibration", "timestamps"],
        "encodings": ["image_sequence"],
    },
    "vss_smart_city": {
        "modalities": ["rgb", "depth", "semantic_segmentation", "instance_segmentation"],
        "annotations": ["bbox_2d", "bbox_3d", "tracking", "captions"],
        "metadata": ["manifest", "calibration", "timestamps", "opendrive"],
        "encodings": ["image_sequence", "mp4"],
    },
}


@dataclass(frozen=True)
class ResolvedOutputPlan:
    spec: RenderOutputSpec
    modalities: frozenset[str]
    annotations: frozenset[str]
    metadata: frozenset[str]
    encodings: frozenset[str]

    @property
    def profile(self) -> str:
        return self.spec.profile

    @property
    def restrict_modalities(self) -> bool:
        return bool(self.modalities)

    def include_annotation(self, name: str) -> bool:
        return name in self.annotations

    def include_metadata(self, name: str) -> bool:
        return name in self.metadata

    def include_encoding(self, name: str) -> bool:
        return name in self.encodings


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def resolve_output_spec(spec: RenderOutputSpec | None) -> ResolvedOutputPlan:
    effective = spec or RenderOutputSpec()
    if effective.profile != "custom":
        preset = PROFILE_PRESETS[effective.profile]
        modalities = _unique(preset["modalities"])
        annotations = _unique(preset["annotations"])
        metadata = _unique(["manifest", *preset["metadata"]])
        encodings = _unique(preset["encodings"])
    else:
        modalities = _unique(list(effective.modalities))
        annotations = _unique(list(effective.annotations))
        metadata = _unique(["manifest", *effective.metadata])
        encodings = _unique(list(effective.encodings))
    resolved = RenderOutputSpec(
        version=1,
        profile=effective.profile,
        modalities=modalities,
        annotations=annotations,
        metadata=metadata,
        encodings=encodings,
    )
    return ResolvedOutputPlan(
        spec=resolved,
        modalities=frozenset(resolved.modalities),
        annotations=frozenset(resolved.annotations),
        metadata=frozenset(resolved.metadata),
        encodings=frozenset(resolved.encodings),
    )


def should_capture_sensor(sensor: SensorConfig, plan: ResolvedOutputPlan) -> bool:
    return should_capture_raw_sensor(sensor, plan) or should_encode_sensor_video(sensor, plan)


def should_capture_raw_sensor(sensor: SensorConfig, plan: ResolvedOutputPlan) -> bool:
    if sensor.sensor_category == "camera":
        if plan.profile == "playback" and not plan.restrict_modalities:
            return False
        if plan.profile == "custom" and not plan.restrict_modalities:
            return "image_sequence" in plan.encodings
        return "image_sequence" in plan.encodings and _sensor_modality_enabled(sensor.output_modality, plan)
    if sensor.sensor_category == "lidar":
        if plan.profile == "playback" and not plan.restrict_modalities:
            return False
        if plan.profile == "custom" and not plan.restrict_modalities:
            return False
        return _sensor_modality_enabled(sensor.output_modality, plan)
    if sensor.sensor_category in {"radar", "imu", "gnss"}:
        if plan.profile == "playback" and not plan.restrict_modalities:
            return False
        if plan.profile == "custom" and not plan.restrict_modalities:
            return False
        return _sensor_modality_enabled(sensor.output_modality, plan)
    return False


def should_encode_sensor_video(sensor: SensorConfig, plan: ResolvedOutputPlan) -> bool:
    if "mp4" not in plan.encodings:
        return False
    if sensor.sensor_category not in ENCODABLE_SENSOR_CATEGORIES:
        return False
    if sensor.sensor_category == "camera" and sensor.output_modality not in {"rgb", "normals"}:
        return False
    if plan.profile == "playback" and not plan.restrict_modalities:
        return True
    if plan.profile == "custom" and not plan.restrict_modalities:
        return True
    return _sensor_modality_enabled(sensor.output_modality, plan)


def should_upload_gt_sensor_frames(sensor_name: str, plan: ResolvedOutputPlan) -> bool:
    modality = GT_SENSOR_NAME_TO_MODALITY.get(sensor_name)
    return bool(modality and modality in plan.modalities)


def required_gt_sensor_names(plan: ResolvedOutputPlan) -> list[str]:
    names: set[str] = set()
    if any(name in plan.annotations for name in {"bbox_2d", "bbox_3d", "tracking"}):
        names.add("semantic_seg")
    for modality in plan.modalities:
        sensor_name = GT_MODALITY_TO_SENSOR_NAME.get(modality)
        if sensor_name:
            names.add(sensor_name)
    return sorted(names)


def should_emit_bbox_annotations(plan: ResolvedOutputPlan) -> bool:
    return any(name in plan.annotations for name in {"bbox_2d", "bbox_3d", "tracking"})


def should_emit_caption(plan: ResolvedOutputPlan) -> bool:
    return "captions" in plan.annotations


def _sensor_modality_enabled(modality: str, plan: ResolvedOutputPlan) -> bool:
    if not plan.restrict_modalities:
        if plan.profile == "custom":
            return False
        return plan.profile != "playback" or modality in {"rgb", "point_cloud", "semantic_point_cloud"}
    return modality in plan.modalities
