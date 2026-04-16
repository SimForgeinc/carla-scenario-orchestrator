"""Server-level status: GPU hardware, system resources, Docker containers.

Replaces the separate monitor SvelteKit app by providing these capabilities
directly from the orchestrator process (which already runs on the GPU server).
"""

import json
import logging
import re
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def _run(cmd: str, timeout: int = 10) -> str:
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning(f"Command failed: {cmd}: {e}")
        return ""


def get_gpu_hardware() -> list[dict[str, Any]]:
    raw = _run(
        "nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,temperature.gpu,power.draw "
        "--format=csv,noheader"
    )
    gpus = []
    for line in raw.split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        try:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1] if len(parts) > 1 else "Unknown",
                "utilizationGpu": int(parts[2].split()[0]) if len(parts) > 2 else 0,
                "utilizationMem": int(parts[3].split()[0]) if len(parts) > 3 else 0,
                "memoryUsed": int(parts[4].split()[0]) if len(parts) > 4 else 0,
                "memoryTotal": int(parts[5].split()[0]) if len(parts) > 5 else 0,
                "temperature": int(parts[6].split()[0]) if len(parts) > 6 else 0,
                "powerDraw": float(parts[7].split()[0]) if len(parts) > 7 else 0,
            })
        except (ValueError, IndexError):
            continue
    return gpus


def get_gpu_processes() -> list[dict[str, str]]:
    raw = _run("nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,name --format=csv,noheader 2>/dev/null")
    processes = []
    for line in raw.split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        processes.append({
            "pid": parts[0] if parts else "",
            "gpuUuid": parts[1] if len(parts) > 1 else "",
            "usedMemory": parts[2] if len(parts) > 2 else "",
            "processName": parts[3] if len(parts) > 3 else "",
        })
    return processes


def get_docker_containers() -> list[dict[str, str]]:
    fmt = '{"name":"{{.Names}}","status":"{{.Status}}","image":"{{.Image}}","ports":"{{.Ports}}","state":"{{.State}}"}'
    raw = _run(f"docker ps -a --format '{fmt}'")
    containers = []
    for line in raw.split("\n"):
        if not line.strip():
            continue
        try:
            containers.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return containers


def get_system_memory() -> dict[str, int]:
    raw = _run("free -m | grep Mem")
    parts = raw.split()
    return {
        "totalMB": int(parts[1]) if len(parts) > 1 else 0,
        "usedMB": int(parts[2]) if len(parts) > 2 else 0,
        "availableMB": int(parts[6]) if len(parts) > 6 else 0,
    }


def get_disk_usage() -> dict[str, Any]:
    raw = _run("df -h / | tail -1")
    parts = raw.split()
    return {
        "size": parts[1] if len(parts) > 1 else "0",
        "used": parts[2] if len(parts) > 2 else "0",
        "available": parts[3] if len(parts) > 3 else "0",
        "usePercent": int(parts[4].replace("%", "")) if len(parts) > 4 else 0,
    }


def get_uptime() -> dict[str, Any]:
    raw = _run("uptime")
    load_match = re.search(r"load average:\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)", raw)
    uptime_match = re.search(r"up\s+(.+?),\s+\d+\s+user", raw)
    return {
        "text": uptime_match.group(1).strip() if uptime_match else "unknown",
        "load1": float(load_match.group(1)) if load_match else 0,
        "load5": float(load_match.group(2)) if load_match else 0,
        "load15": float(load_match.group(3)) if load_match else 0,
    }


def get_full_server_status() -> dict[str, Any]:
    return {
        "gpus": get_gpu_hardware(),
        "gpuProcesses": get_gpu_processes(),
        "containers": get_docker_containers(),
        "memory": get_system_memory(),
        "disk": get_disk_usage(),
        "uptime": get_uptime(),
    }


def get_container_logs(container: str, lines: int = 100) -> str:
    if not re.match(r"^[a-zA-Z0-9_-]+$", container):
        raise ValueError("Invalid container name")
    return _run(f"docker logs {container} --tail {min(lines, 500)} 2>&1", timeout=15)


def restart_container(container: str) -> dict[str, Any]:
    if not re.match(r"^[a-zA-Z0-9_-]+$", container):
        raise ValueError("Invalid container name")
    _run(f"docker restart {container}", timeout=30)
    status = _run(f"docker ps --filter name={container} --format '{{{{.Status}}}}'")
    return {"ok": True, "container": container, "status": status}
