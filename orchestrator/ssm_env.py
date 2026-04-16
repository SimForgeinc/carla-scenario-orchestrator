from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

logger = logging.getLogger(__name__)
_SERVICE_SUFFIXES = {"editor", "platform", "admin", "orchestrator"}
_ALIAS_MAP = {
    "S3_BUCKET": "ORCH_STORAGE_BUCKET",
    "ORCHESTRATOR_WEBHOOK_SECRET": "ORCH_WEBHOOK_SECRET",
}

try:
    import boto3
except ModuleNotFoundError:  # pragma: no cover
    boto3 = None


@lru_cache(maxsize=4)
def load_ssm_parameters(env_name: str | None, service: str = "orchestrator", prefix: str = "/simforge") -> dict[str, str]:
    if not env_name or boto3 is None:
        return {}

    region = os.environ.get("AWS_REGION") or os.environ.get("AURORA_REGION") or "us-east-1"
    client = boto3.client(
        "ssm",
        region_name=region,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )
    path = f"{prefix.rstrip('/')}/{env_name}/"
    paginator = client.get_paginator("get_parameters_by_path")

    resolved: dict[str, str] = {}
    service_overrides: dict[str, str] = {}

    try:
        for page in paginator.paginate(Path=path, Recursive=False, WithDecryption=True):
            for parameter in page.get("Parameters", []):
                raw_name = parameter["Name"].rsplit("/", 1)[-1]
                value = parameter["Value"]
                match = re.match(r"^(?P<base>.+)_(?P<svc>[A-Za-z0-9-]+)$", raw_name)
                if match and match.group("svc") in _SERVICE_SUFFIXES:
                    if match.group("svc") == service:
                        service_overrides[match.group("base")] = value
                    continue
                resolved[raw_name] = value
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load SSM parameters from %s for service=%s: %s", path, service, exc)
        return {}

    resolved.update(service_overrides)
    for source, target in _ALIAS_MAP.items():
        if source in resolved:
            resolved[target] = resolved[source]

    for key, value in resolved.items():
        os.environ[key] = value

    logger.info("Loaded %s env vars from SSM path %s for service=%s", len(resolved), path, service)
    return resolved
