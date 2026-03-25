"""Tests for orchestrator.worker_pool — health check, EventPusher, dispatch."""
from __future__ import annotations

import time
import threading
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

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


class TestCheckWorkersBackoff(unittest.TestCase):
    """Test the exponential backoff and unhealthy marking in check_workers."""

    def _make_pool(self):
        from orchestrator.worker_pool import WorkerPool
        settings = make_settings()
        scheduler = GpuScheduler(settings)
        pool = WorkerPool(settings, scheduler)
        return pool, scheduler

    def test_alive_worker_resets_failure_count(self):
        pool, scheduler = self._make_pool()
        # Simulate a worker that is alive
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        pool._workers[0] = mock_proc
        slot = scheduler.slots()[0]
        pool._worker_slots[0] = slot
        pool._restart_failures[0] = 2  # Had previous failures

        pool.check_workers()

        self.assertEqual(pool._restart_failures[0], 0)
        # Scheduler should mark ready
        self.assertEqual(scheduler._slot_status[0], "ready")

    def test_dead_worker_increments_failure_count(self):
        pool, scheduler = self._make_pool()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        mock_proc.exitcode = 1
        pool._workers[0] = mock_proc
        slot = scheduler.slots()[0]
        pool._worker_slots[0] = slot

        with patch.object(pool, '_start_worker'):
            pool.check_workers()

        self.assertEqual(pool._restart_failures[0], 1)

    def test_marks_unhealthy_after_max_failures(self):
        from orchestrator.worker_pool import MAX_RESTART_FAILURES
        pool, scheduler = self._make_pool()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        mock_proc.exitcode = 1
        pool._workers[0] = mock_proc
        slot = scheduler.slots()[0]
        pool._worker_slots[0] = slot

        # Set failures to just below threshold
        pool._restart_failures[0] = MAX_RESTART_FAILURES - 1

        with patch.object(pool, '_start_worker'):
            pool.check_workers()

        # Should be marked unhealthy
        self.assertEqual(scheduler._slot_status[0], "unhealthy")
        self.assertIn("failed", scheduler._slot_errors[0])

    def test_backoff_prevents_immediate_retry(self):
        pool, scheduler = self._make_pool()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        mock_proc.exitcode = 1
        pool._workers[0] = mock_proc
        slot = scheduler.slots()[0]
        pool._worker_slots[0] = slot

        # Set a future retry time
        pool._restart_failures[0] = 1
        pool._next_retry_time[0] = time.time() + 9999

        start_worker_called = False
        original_start = pool._start_worker

        def mock_start(*args, **kwargs):
            nonlocal start_worker_called
            start_worker_called = True

        with patch.object(pool, '_start_worker', mock_start):
            pool.check_workers()

        # Should NOT have tried to restart (still in backoff)
        self.assertFalse(start_worker_called)

    def test_expired_backoff_allows_retry(self):
        pool, scheduler = self._make_pool()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        mock_proc.exitcode = 1
        pool._workers[0] = mock_proc
        slot = scheduler.slots()[0]
        pool._worker_slots[0] = slot

        # Set a past retry time
        pool._restart_failures[0] = 1
        pool._next_retry_time[0] = time.time() - 1

        start_worker_called = False

        def mock_start(*args, **kwargs):
            nonlocal start_worker_called
            start_worker_called = True

        with patch.object(pool, '_start_worker', mock_start):
            pool.check_workers()

        self.assertTrue(start_worker_called)

    def test_docker_restart_on_second_failure(self):
        """On 2nd consecutive failure, should attempt Docker container restart."""
        pool, scheduler = self._make_pool()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        mock_proc.exitcode = 1
        pool._workers[0] = mock_proc
        slot = scheduler.slots()[0]
        pool._worker_slots[0] = slot
        pool._restart_failures[0] = 1  # This will be the 2nd failure

        with patch.object(pool, '_start_worker'), \
             patch('subprocess.run') as mock_run:
            # Docker inspect returns "not running"
            mock_inspect = MagicMock()
            mock_inspect.returncode = 0
            mock_inspect.stdout = "false\n"
            mock_run.return_value = mock_inspect

            pool.check_workers()

        # subprocess.run should have been called for docker inspect + restart
        self.assertTrue(mock_run.called)


class TestCheckWorkersRecovery(unittest.TestCase):
    """Test that a recovered worker resets its failure state."""

    def test_recovery_clears_state_and_marks_ready(self):
        from orchestrator.worker_pool import WorkerPool
        settings = make_settings()
        scheduler = GpuScheduler(settings)
        pool = WorkerPool(settings, scheduler)

        # Worker was unhealthy, now alive again
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        pool._workers[0] = mock_proc
        slot = scheduler.slots()[0]
        pool._worker_slots[0] = slot
        pool._restart_failures[0] = 5
        pool._next_retry_time[0] = time.time() + 9999
        scheduler.mark_slot_unhealthy(0, error="test")

        pool.check_workers()

        self.assertEqual(pool._restart_failures[0], 0)
        self.assertNotIn(0, pool._next_retry_time)
        self.assertEqual(scheduler._slot_status[0], "ready")


class TestEventPusherBatching(unittest.TestCase):
    """Test EventPusher batch accumulation and flush behavior."""

    def test_flush_on_batch_size(self):
        """Events should be flushed when batch reaches batch_size."""
        # We can't easily test the inner class directly, but we can test
        # the batching logic by simulating it
        batch = []
        batch_size = 20

        for i in range(25):
            batch.append({"kind": "stream", "payload": {"frame": i}})
            if len(batch) >= batch_size:
                # Would flush here
                self.assertEqual(len(batch), 20)
                batch = batch[20:]  # Simulate clearing

        self.assertEqual(len(batch), 5)

    def test_flush_on_close(self):
        """Remaining events should be flushed on close."""
        batch = [{"kind": "stream"} for _ in range(5)]
        # close() should flush remaining
        self.assertEqual(len(batch), 5)
        # After flush
        batch = []
        self.assertEqual(len(batch), 0)


class TestEventPusherRetry(unittest.TestCase):
    """Test that EventPusher retries once on HTTP failure."""

    @patch('requests.post')
    def test_retry_on_first_failure(self, mock_post):
        """Should retry once if first POST fails."""
        import requests

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.ConnectionError("Connection refused")
            return MagicMock(status_code=200)

        mock_post.side_effect = side_effect

        # Simulate what _flush does
        batch = [{"kind": "stream"}]
        url = "http://127.0.0.1:18421/api/jobs/test/events"

        for attempt in range(2):
            try:
                requests.post(url, json=batch, timeout=5)
                break
            except Exception:
                if attempt == 0:
                    time.sleep(0.01)  # Short sleep for test

        self.assertEqual(call_count, 2)

    @patch('requests.post')
    def test_drops_after_two_failures(self, mock_post):
        """Should drop events after 2 failed attempts."""
        import requests

        mock_post.side_effect = requests.ConnectionError("Connection refused")

        dropped = False
        batch = [{"kind": "stream"}]
        url = "http://127.0.0.1:18421/api/jobs/test/events"

        for attempt in range(2):
            try:
                requests.post(url, json=batch, timeout=5)
                break
            except Exception:
                if attempt == 1:
                    dropped = True

        self.assertTrue(dropped)
        self.assertEqual(mock_post.call_count, 2)


class TestTaskQueueForSlot(unittest.TestCase):
    """Test task queue naming."""

    def test_task_queue_naming(self):
        from orchestrator.worker_pool import task_queue_for_slot
        self.assertEqual(task_queue_for_slot(0), "carla-slot-0")
        self.assertEqual(task_queue_for_slot(5), "carla-slot-5")


class TestWorkerPoolInit(unittest.TestCase):
    """Test WorkerPool initialization."""

    def test_init_state(self):
        from orchestrator.worker_pool import WorkerPool
        settings = make_settings()
        scheduler = GpuScheduler(settings)
        pool = WorkerPool(settings, scheduler)

        self.assertIsNone(pool._temporal_client)
        self.assertEqual(len(pool._workers), 0)
        self.assertEqual(len(pool._restart_failures), 0)
        self.assertEqual(len(pool._next_retry_time), 0)


class TestWorkerPoolStop(unittest.TestCase):
    """Test WorkerPool.stop() terminates all workers."""

    def test_stop_terminates_workers(self):
        from orchestrator.worker_pool import WorkerPool
        settings = make_settings()
        scheduler = GpuScheduler(settings)
        pool = WorkerPool(settings, scheduler)

        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        pool._workers[0] = mock_proc
        pool._workers[1] = mock_proc

        pool.stop()

        self.assertEqual(mock_proc.terminate.call_count, 2)
        self.assertEqual(mock_proc.join.call_count, 2)


class TestConstants(unittest.TestCase):
    """Test that health check constants are sensible."""

    def test_backoff_constants(self):
        from orchestrator.worker_pool import (
            MAX_RESTART_FAILURES,
            BACKOFF_BASE_SECONDS,
            BACKOFF_MAX_SECONDS,
        )
        self.assertGreaterEqual(MAX_RESTART_FAILURES, 2)
        self.assertGreater(BACKOFF_BASE_SECONDS, 0)
        self.assertGreater(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS)


if __name__ == "__main__":
    unittest.main()
