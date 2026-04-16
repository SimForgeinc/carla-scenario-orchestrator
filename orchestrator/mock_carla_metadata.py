"""Mock CARLA metadata service for local development without GPU hardware.

Loads fixture data from ../fixtures/ and serves it for all metadata endpoints.
Enable with: MOCK_CARLA=true uvicorn orchestrator.app:app --reload --port 18422
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def _load_fixture(name: str) -> Any:
    path = _FIXTURES_DIR / name
    if not path.exists():
        logger.warning("Fixture not found: %s", path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


class MockCarlaMetadataService:
    """Drop-in replacement for CarlaMetadataService using static fixtures."""

    def __init__(self) -> None:
        logger.info("MockCarlaMetadataService initialized (fixtures from %s)", _FIXTURES_DIR)
        self._status = _load_fixture("carla_status.json")
        self._generated = _load_fixture("map_generated.json")
        self._blueprints = _load_fixture("blueprints.json")

    def warm_cache(self) -> None:
        pass

    def get_status(self) -> dict:
        return self._status

    def load_map(self, map_name: str) -> dict:
        return {"status": "mock", "map_name": map_name, "message": "Mock CARLA: map load is a no-op"}

    def get_runtime_map(self) -> Any:
        runtime = self._generated.get("runtime")
        if runtime:
            from .carla_runner.models import RuntimeMapResponse
            return RuntimeMapResponse.model_validate(runtime)
        return self._generated

    def get_map_xodr(self) -> str:
        return "<OpenDRIVE><header/></OpenDRIVE>"

    def get_generated_map_with_runtime(self) -> dict:
        return self._generated

    def list_blueprints(self) -> dict:
        return self._blueprints
