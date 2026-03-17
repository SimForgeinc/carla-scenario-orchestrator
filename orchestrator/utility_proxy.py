from __future__ import annotations

import json
from typing import Any
from urllib import error, parse, request


class UtilityBackendError(RuntimeError):
    pass


class UtilityBackendProxy:
    def __init__(self, base_url: str | None) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None

    def configured(self) -> bool:
        return bool(self.base_url)

    def fetch_json(self, path: str, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
        raw = self.fetch_bytes(path, method=method, payload=payload)
        return json.loads(raw.decode("utf-8"))

    def fetch_bytes(self, path: str, method: str = "GET", payload: dict[str, Any] | None = None) -> bytes:
        if not self.base_url:
            raise UtilityBackendError("ORCH_UTILITY_BACKEND_BASE is not configured.")
        url = f"{self.base_url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=60) as response:
                return response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise UtilityBackendError(f"Utility backend returned {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise UtilityBackendError(f"Failed to reach utility backend: {exc.reason}") from exc
