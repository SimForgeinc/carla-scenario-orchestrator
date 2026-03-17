from __future__ import annotations

import threading
from dataclasses import dataclass

from .config import Settings
from .models import CapacityResponse, CapacitySlot, GpuLeaseInfo


@dataclass(frozen=True)
class GpuLease:
    slot_index: int
    device_id: str
    carla_rpc_port: int
    traffic_manager_port: int

    def to_model(self) -> GpuLeaseInfo:
        return GpuLeaseInfo(
            slot_index=self.slot_index,
            device_id=self.device_id,
            carla_rpc_port=self.carla_rpc_port,
            traffic_manager_port=self.traffic_manager_port,
        )


class GpuScheduler:
    def __init__(self, settings: Settings) -> None:
        self._condition = threading.Condition()
        self._leases_by_job: dict[str, GpuLease] = {}
        self._job_by_slot: dict[int, str] = {}
        self._slots: list[GpuLease] = []
        for idx, device_id in enumerate(settings.gpu_devices):
            self._slots.append(
                GpuLease(
                    slot_index=idx,
                    device_id=device_id,
                    carla_rpc_port=settings.carla_rpc_port_base + idx * settings.port_stride,
                    traffic_manager_port=settings.traffic_manager_port_base + idx * settings.port_stride,
                )
            )

    def acquire(self, job_id: str, cancel_event: threading.Event) -> GpuLease:
        with self._condition:
            while True:
                if cancel_event.is_set():
                    raise RuntimeError("Job cancelled before a GPU slot was assigned.")
                for slot in self._slots:
                    if slot.slot_index in self._job_by_slot:
                        continue
                    self._job_by_slot[slot.slot_index] = job_id
                    self._leases_by_job[job_id] = slot
                    return slot
                self._condition.wait(timeout=0.5)

    def release(self, job_id: str) -> None:
        with self._condition:
            lease = self._leases_by_job.pop(job_id, None)
            if lease is None:
                return
            self._job_by_slot.pop(lease.slot_index, None)
            self._condition.notify_all()

    def queue_position(self, job_id: str, queued_job_ids: list[str]) -> int:
        try:
            return queued_job_ids.index(job_id) + 1
        except ValueError:
            return 0

    def snapshot(self) -> CapacityResponse:
        with self._condition:
            slots = []
            for slot in self._slots:
                job_id = self._job_by_slot.get(slot.slot_index)
                slots.append(
                    CapacitySlot(
                        slot_index=slot.slot_index,
                        device_id=slot.device_id,
                        busy=job_id is not None,
                        job_id=job_id,
                        carla_rpc_port=slot.carla_rpc_port,
                        traffic_manager_port=slot.traffic_manager_port,
                    )
                )
            busy_slots = sum(1 for slot in slots if slot.busy)
            return CapacityResponse(
                total_slots=len(slots),
                busy_slots=busy_slots,
                free_slots=len(slots) - busy_slots,
                slots=slots,
            )

