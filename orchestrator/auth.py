from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from fastapi import Header, HTTPException

from .carla_runner.models import SimulationRunRequest
from .models import JobListResponse, JobRecord, JobState

logger = logging.getLogger(__name__)

ACTIVE_JOB_STATES = {JobState.queued, JobState.starting, JobState.running}


@dataclass(frozen=True)
class SurfaceTokenClaims:
    token_id: str
    agent_id: str
    workspace_id: str
    scopes: frozenset[str]
    env: str | None = None
    max_duration_seconds: float | None = None
    max_concurrency: int | None = None
    artifact_ttl_hours: int | None = None
    enabled: bool = True

    def has_scope(self, scope: str) -> bool:
        return "*" in self.scopes or scope in self.scopes


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_hash(value: str | None) -> str:
    value = (value or "").strip()
    if value.lower().startswith("sha256:"):
        value = value.split(":", 1)[1]
    return value.lower()


def _catalog() -> list[dict[str, Any]]:
    raw = os.environ.get("ORCH_SURFACE_TOKENS_JSON") or os.environ.get("AGENT_SURFACE_TOKENS_JSON") or ""
    if not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Invalid surface token catalog JSON: %s", exc)
        raise HTTPException(status_code=500, detail="Surface token catalog is invalid") from exc
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and isinstance(parsed.get("tokens"), list):
        return parsed["tokens"]
    raise HTTPException(status_code=500, detail="Surface token catalog must be an array or {tokens: [...]}")


def _claims_from_record(record: dict[str, Any]) -> SurfaceTokenClaims:
    agent_id = str(record.get("agent_id") or record.get("agentId") or "").strip()
    workspace_id = str(record.get("workspace_id") or record.get("workspaceId") or "").strip()
    if not agent_id or not workspace_id:
        raise HTTPException(status_code=500, detail="Surface token record is missing agent_id or workspace_id")
    scopes = record.get("scopes") if isinstance(record.get("scopes"), list) else []
    return SurfaceTokenClaims(
        token_id=str(record.get("token_id") or record.get("tokenId") or agent_id),
        agent_id=agent_id,
        workspace_id=workspace_id,
        scopes=frozenset(str(scope) for scope in scopes),
        env=(str(record.get("env")).strip() if record.get("env") else None),
        max_duration_seconds=(
            float(record.get("max_duration_seconds") or record["maxDurationSeconds"])
            if record.get("max_duration_seconds") is not None or record.get("maxDurationSeconds") is not None
            else None
        ),
        max_concurrency=(
            int(record.get("max_concurrency") or record["maxConcurrency"])
            if record.get("max_concurrency") is not None or record.get("maxConcurrency") is not None
            else None
        ),
        artifact_ttl_hours=(
            int(record.get("artifact_ttl_hours") or record["artifactTtlHours"])
            if record.get("artifact_ttl_hours") is not None or record.get("artifactTtlHours") is not None
            else None
        ),
        enabled=record.get("enabled", True) is not False,
    )


def validate_surface_token(raw_token: str) -> SurfaceTokenClaims:
    token_hash = _sha256(raw_token)
    for record in _catalog():
        expected = _normalize_hash(record.get("token_hash") or record.get("tokenHash"))
        if expected and hmac.compare_digest(token_hash, expected):
            claims = _claims_from_record(record)
            if not claims.enabled:
                raise HTTPException(status_code=401, detail="Surface token is disabled")
            current_env = os.environ.get("SIMFORGE_ENV", "").strip()
            if claims.env and current_env and claims.env != current_env:
                raise HTTPException(status_code=403, detail="Surface token is not valid for this environment")
            return claims
    raise HTTPException(status_code=401, detail="Invalid surface token")


async def surface_token_from_authorization(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> SurfaceTokenClaims | None:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    return validate_surface_token(value.strip())


def require_surface_scope(claims: SurfaceTokenClaims, scope: str) -> None:
    if claims.has_scope(scope):
        return
    raise HTTPException(status_code=403, detail=f"Surface token is missing scope: {scope}")


def _job_belongs_to_agent(job: JobRecord, claims: SurfaceTokenClaims) -> bool:
    return getattr(job.request, "submitted_by_agent_id", None) == claims.agent_id


def ensure_job_read_access(job: JobRecord, claims: SurfaceTokenClaims) -> None:
    if claims.has_scope("carla:job:read:any") or _job_belongs_to_agent(job, claims):
        return
    raise HTTPException(status_code=404, detail="Job not found")


def ensure_job_cancel_access(job: JobRecord, claims: SurfaceTokenClaims) -> None:
    if claims.has_scope("carla:job:cancel:any") or _job_belongs_to_agent(job, claims):
        return
    raise HTTPException(status_code=404, detail="Job not found")


def filter_jobs_for_surface_token(job_list: JobListResponse, claims: SurfaceTokenClaims) -> JobListResponse:
    if claims.has_scope("carla:job:read:any"):
        return job_list
    return JobListResponse(items=[job for job in job_list.items if _job_belongs_to_agent(job, claims)])


def authorize_job_submission(
    existing_jobs: JobListResponse,
    request: SimulationRunRequest,
    claims: SurfaceTokenClaims,
) -> SimulationRunRequest:
    require_surface_scope(claims, "carla:job:create")

    if claims.max_concurrency is not None:
        active = sum(
            1
            for job in existing_jobs.items
            if _job_belongs_to_agent(job, claims) and job.state in ACTIVE_JOB_STATES
        )
        if active >= claims.max_concurrency:
            raise HTTPException(status_code=429, detail="Surface token concurrency limit exceeded")

    duration = request.duration_seconds
    if claims.max_duration_seconds is not None:
        duration = min(duration, claims.max_duration_seconds)

    updates = {
        "duration_seconds": duration,
        "submitted_by_agent_id": claims.agent_id,
        "workspace_id": claims.workspace_id,
        "test_run": True,
        "artifact_ttl_hours": claims.artifact_ttl_hours,
        "priority": "batch",
    }
    logger.info(
        "surface-token job authorized token_id=%s agent_id=%s workspace_id=%s duration=%s",
        claims.token_id,
        claims.agent_id,
        claims.workspace_id,
        duration,
    )
    return request.model_copy(update=updates)
