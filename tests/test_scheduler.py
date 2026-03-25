from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

from orchestrator.config import Settings
from orchestrator.scheduler import GpuScheduler


def make_settings(gpu_devices=("0", "1", "2"), metadata_slot_index=2) -> Settings:
    repo_root = Path("/tmp/carla-scenario-orchestrator-tests")
    return Settings(
        repo_root=repo_root,
        jobs_root=repo_root / "runs",
        gpu_devices=gpu_devices,
        carla_image="carla:test",
        carla_container_prefix="carla-orch",
        carla_startup_timeout_seconds=5,
        carla_rpc_port_base=2000,
        traffic_manager_port_base=8000,
        port_stride=100,
        carla_timeout_seconds=20,
        python_executable="python3",
        docker_network_mode="host",
        carla_start_command_template="./CarlaUE4.sh -carla-rpc-port={rpc_port}",
        metadata_slot_index=metadata_slot_index,
        carla_metadata_host="127.0.0.1",
        carla_metadata_port=2000 + metadata_slot_index * 100,
        carla_metadata_timeout=20,
        storage_bucket=None,
        storage_region="us-east-1",
        storage_prefix="runs",
    )


class GpuSchedulerTests(unittest.TestCase):
    def test_allocates_unique_ports_per_execution_gpu(self) -> None:
        scheduler = GpuScheduler(make_settings())
        lease_a = scheduler.acquire("job-a", threading.Event())
        lease_b = scheduler.acquire("job-b", threading.Event())
        self.assertNotEqual(lease_a.device_id, lease_b.device_id)
        self.assertEqual(lease_a.carla_rpc_port, 2000)
        self.assertEqual(lease_b.carla_rpc_port, 2100)
        self.assertEqual(lease_a.role, "execution")
        self.assertEqual(lease_b.role, "execution")
        scheduler.release("job-a")
        scheduler.release("job-b")

    def test_reserves_metadata_slot_from_job_leasing(self) -> None:
        scheduler = GpuScheduler(make_settings())
        snapshot = scheduler.snapshot()
        self.assertEqual(snapshot.total_slots, 2)
        self.assertEqual(snapshot.metadata_slots, 1)
        self.assertTrue(snapshot.metadata_ready)
        self.assertEqual(snapshot.metadata_slot_index, 2)
        metadata_slots = [slot for slot in snapshot.slots if slot.role == "metadata"]
        self.assertEqual(len(metadata_slots), 1)
        self.assertEqual(metadata_slots[0].container_name, "carla-orch-metadata")

    def test_waits_until_execution_slot_is_free(self) -> None:
        scheduler = GpuScheduler(make_settings(gpu_devices=("0", "1"), metadata_slot_index=1))
        scheduler.acquire("job-a", threading.Event())
        acquired: list[str] = []

        def worker() -> None:
            lease = scheduler.acquire("job-b", threading.Event())
            acquired.append(lease.device_id)
            scheduler.release("job-b")

        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.2)
        self.assertEqual(acquired, [])
        scheduler.release("job-a")
        thread.join(timeout=1)
        self.assertEqual(acquired, ["0"])

    # --- Map-aware scheduling tests ---

    def test_prefers_slot_with_matching_map(self) -> None:
        scheduler = GpuScheduler(make_settings())
        scheduler.set_slot_map(0, "Town05")
        scheduler.set_slot_map(1, "VW_Poc")
        lease = scheduler.acquire("job-a", threading.Event(), map_name="VW_Poc")
        self.assertEqual(lease.slot_index, 1)
        scheduler.release("job-a")

    def test_falls_back_when_no_map_match(self) -> None:
        scheduler = GpuScheduler(make_settings())
        scheduler.set_slot_map(0, "Town05")
        scheduler.set_slot_map(1, "Town05")
        lease = scheduler.acquire("job-a", threading.Event(), map_name="VW_Poc")
        # Should get first free slot (0) since no match exists
        self.assertEqual(lease.slot_index, 0)
        scheduler.release("job-a")

    def test_acquire_without_map_name_backward_compat(self) -> None:
        scheduler = GpuScheduler(make_settings())
        scheduler.set_slot_map(0, "Town05")
        scheduler.set_slot_map(1, "VW_Poc")
        # No map_name → first free slot
        lease = scheduler.acquire("job-a", threading.Event())
        self.assertEqual(lease.slot_index, 0)
        scheduler.release("job-a")

    def test_map_match_skips_busy_slots(self) -> None:
        scheduler = GpuScheduler(make_settings())
        scheduler.set_slot_map(0, "VW_Poc")
        scheduler.set_slot_map(1, "Town05")
        # Occupy slot 0 (which has VW_Poc)
        scheduler.acquire("job-a", threading.Event())
        # Now request VW_Poc — slot 0 is busy, should get slot 1 (fallback)
        lease = scheduler.acquire("job-b", threading.Event(), map_name="VW_Poc")
        self.assertEqual(lease.slot_index, 1)
        scheduler.release("job-a")
        scheduler.release("job-b")

    def test_set_and_get_slot_map(self) -> None:
        scheduler = GpuScheduler(make_settings())
        self.assertIsNone(scheduler.get_slot_map(0))
        scheduler.set_slot_map(0, "Town05")
        self.assertEqual(scheduler.get_slot_map(0), "Town05")
        self.assertIsNone(scheduler.get_slot_map(1))

    def test_snapshot_includes_current_map(self) -> None:
        scheduler = GpuScheduler(make_settings())
        scheduler.set_slot_map(0, "VW_Poc")
        snapshot = scheduler.snapshot()
        exec_slots = [s for s in snapshot.slots if s.role == "execution"]
        self.assertEqual(exec_slots[0].current_map, "VW_Poc")
        self.assertIsNone(exec_slots[1].current_map)

    def test_map_tracking_persists_across_release(self) -> None:
        scheduler = GpuScheduler(make_settings())
        lease = scheduler.acquire("job-a", threading.Event())
        scheduler.set_slot_map(lease.slot_index, "VW_Poc")
        scheduler.release("job-a")
        # Map should still be tracked after release
        self.assertEqual(scheduler.get_slot_map(lease.slot_index), "VW_Poc")
        # And next acquire with same map should prefer this slot
        lease2 = scheduler.acquire("job-b", threading.Event(), map_name="VW_Poc")
        self.assertEqual(lease2.slot_index, lease.slot_index)
        scheduler.release("job-b")



    # --- LRU slot selection tests ---

    def test_lru_distributes_evenly(self) -> None:
        """Jobs should round-robin across slots, not always pick slot 0."""
        scheduler = GpuScheduler(make_settings())
        scheduler.set_slot_map(0, "VW_Poc")
        scheduler.set_slot_map(1, "VW_Poc")

        slot_counts = {0: 0, 1: 0}
        for i in range(10):
            lease = scheduler.acquire(f"job-{i}", threading.Event(), map_name="VW_Poc")
            slot_counts[lease.slot_index] += 1
            scheduler.release(f"job-{i}")

        # Both slots should get jobs (not all going to slot 0)
        self.assertGreater(slot_counts[0], 0)
        self.assertGreater(slot_counts[1], 0)
        # Should be roughly even (within 2 of each other)
        self.assertLessEqual(abs(slot_counts[0] - slot_counts[1]), 2)

    def test_lru_prefers_idle_longest(self) -> None:
        """Among free matching slots, should pick the one released earliest."""
        import time
        scheduler = GpuScheduler(make_settings())
        scheduler.set_slot_map(0, "VW_Poc")
        scheduler.set_slot_map(1, "VW_Poc")

        # Acquire both, release slot 1 first, then slot 0
        scheduler.acquire("job-a", threading.Event())  # gets slot 0
        scheduler.acquire("job-b", threading.Event())  # gets slot 1
        scheduler.release("job-b")  # slot 1 released first (earlier timestamp)
        time.sleep(0.01)
        scheduler.release("job-a")  # slot 0 released second (later timestamp)

        # Next acquire should pick slot 1 (released earlier = idle longer)
        lease = scheduler.acquire("job-c", threading.Event(), map_name="VW_Poc")
        self.assertEqual(lease.slot_index, 1)
        scheduler.release("job-c")


if __name__ == "__main__":
    unittest.main()
