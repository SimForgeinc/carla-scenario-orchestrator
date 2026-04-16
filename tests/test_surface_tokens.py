from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from orchestrator.auth import SurfaceTokenClaims, authorize_job_submission, require_surface_scope, validate_surface_token
from orchestrator.carla_runner.models import SimulationRunRequest
from orchestrator.models import JobArtifacts, JobListResponse, JobRecord, JobState


def token_hash(token: str) -> str:
    return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()


def test_validate_surface_token_reads_hashed_catalog(monkeypatch):
    monkeypatch.setenv("SIMFORGE_ENV", "dev")
    monkeypatch.setenv(
        "ORCH_SURFACE_TOKENS_JSON",
        '[{"token_hash":"%s","token_id":"tok-dev","agent_id":"codex","workspace_id":"agent-test","env":"dev","scopes":["carla:job:create"]}]'
        % token_hash("sfat_secret"),
    )

    claims = validate_surface_token("sfat_secret")

    assert claims.token_id == "tok-dev"
    assert claims.agent_id == "codex"
    assert claims.workspace_id == "agent-test"
    assert claims.has_scope("carla:job:create")


def test_validate_surface_token_rejects_wrong_env(monkeypatch):
    monkeypatch.setenv("SIMFORGE_ENV", "staging")
    monkeypatch.setenv(
        "ORCH_SURFACE_TOKENS_JSON",
        '[{"token_hash":"%s","agent_id":"codex","workspace_id":"agent-test","env":"dev","scopes":["carla:job:create"]}]'
        % token_hash("sfat_secret"),
    )

    with pytest.raises(HTTPException) as exc:
        validate_surface_token("sfat_secret")

    assert exc.value.status_code == 403


def test_authorize_job_submission_caps_and_tags_request():
    claims = SurfaceTokenClaims(
        token_id="tok",
        agent_id="codex",
        workspace_id="agent-test",
        scopes=frozenset({"carla:job:create"}),
        max_duration_seconds=5,
        max_concurrency=1,
        artifact_ttl_hours=24,
    )
    request = SimulationRunRequest(map_name="Town01", duration_seconds=15, priority="interactive")

    updated = authorize_job_submission(JobListResponse(items=[]), request, claims)

    assert updated.duration_seconds == 5
    assert updated.submitted_by_agent_id == "codex"
    assert updated.workspace_id == "agent-test"
    assert updated.test_run is True
    assert updated.artifact_ttl_hours == 24
    assert updated.priority == "batch"


def test_authorize_job_submission_enforces_concurrency():
    claims = SurfaceTokenClaims(
        token_id="tok",
        agent_id="codex",
        workspace_id="agent-test",
        scopes=frozenset({"carla:job:create"}),
        max_concurrency=1,
    )
    request = SimulationRunRequest(map_name="Town01")
    job = JobRecord(
        job_id="job-1",
        state=JobState.running,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        request=request.model_copy(update={"submitted_by_agent_id": "codex"}),
        artifacts=JobArtifacts(
            output_dir="/tmp/job",
            request_file="/tmp/job/request.json",
            runtime_settings_file="/tmp/job/runtime.json",
        ),
    )

    with pytest.raises(HTTPException) as exc:
        authorize_job_submission(JobListResponse(items=[job]), request, claims)

    assert exc.value.status_code == 429


def test_require_surface_scope_rejects_missing_scope():
    claims = SurfaceTokenClaims(
        token_id="tok",
        agent_id="codex",
        workspace_id="agent-test",
        scopes=frozenset({"carla:job:read"}),
    )

    with pytest.raises(HTTPException) as exc:
        require_surface_scope(claims, "carla:job:create")

    assert exc.value.status_code == 403
