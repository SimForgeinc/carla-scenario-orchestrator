from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    jobs_root: Path
    gpu_devices: tuple[str, ...]
    carla_image: str
    carla_container_prefix: str
    carla_startup_timeout_seconds: float
    carla_rpc_port_base: int
    traffic_manager_port_base: int
    port_stride: int
    carla_timeout_seconds: float
    python_executable: str
    docker_network_mode: str
    carla_start_command_template: str

    @classmethod
    def load(cls) -> "Settings":
        repo_root = Path(__file__).resolve().parents[1]
        jobs_root = Path(os.environ.get("ORCH_JOBS_ROOT", repo_root / "runs")).resolve()
        jobs_root.mkdir(parents=True, exist_ok=True)
        gpu_devices = tuple(_split_csv(os.environ.get("ORCH_GPU_DEVICES", "0,1,2,3,4,5,6,7")))
        if not gpu_devices:
            raise RuntimeError("ORCH_GPU_DEVICES must contain at least one GPU identifier.")
        return cls(
            repo_root=repo_root,
            jobs_root=jobs_root,
            gpu_devices=gpu_devices,
            carla_image=os.environ.get("ORCH_CARLA_IMAGE", "carlasim/carla:0.9.16-phase3"),
            carla_container_prefix=os.environ.get("ORCH_CARLA_CONTAINER_PREFIX", "carla-orch"),
            carla_startup_timeout_seconds=float(os.environ.get("ORCH_CARLA_STARTUP_TIMEOUT", "90")),
            carla_rpc_port_base=int(os.environ.get("ORCH_CARLA_RPC_PORT_BASE", "2000")),
            traffic_manager_port_base=int(os.environ.get("ORCH_TRAFFIC_MANAGER_PORT_BASE", "8000")),
            port_stride=int(os.environ.get("ORCH_PORT_STRIDE", "100")),
            carla_timeout_seconds=float(os.environ.get("ORCH_CARLA_TIMEOUT", "20")),
            python_executable=os.environ.get("ORCH_PYTHON_EXECUTABLE", sys.executable),
            docker_network_mode=os.environ.get("ORCH_DOCKER_NETWORK_MODE", "host"),
            carla_start_command_template=os.environ.get(
                "ORCH_CARLA_START_COMMAND",
                "./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port={rpc_port}",
            ),
        )

