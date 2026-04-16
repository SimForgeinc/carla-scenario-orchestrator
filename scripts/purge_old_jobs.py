#!/usr/bin/env python3
"""One-time migration: purge all existing job directories from local disk.

These jobs have already completed. We don't need the local files — any that
were important (MP4s) were already uploaded to S3 by the existing upload flow.

Run with: python3 scripts/purge_old_jobs.py [--dry-run]
"""
import argparse
import shutil
import sys
from pathlib import Path

RUNS_DIR = Path("/home/ubuntu/carla-scenario-orchestrator/runs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    args = parser.parse_args()

    if not RUNS_DIR.is_dir():
        print(f"Runs directory not found: {RUNS_DIR}")
        return

    job_dirs = sorted(d for d in RUNS_DIR.iterdir() if d.is_dir())
    total_size = 0
    count = 0

    for job_dir in job_dirs:
        dir_size = sum(f.stat().st_size for f in job_dir.rglob("*") if f.is_file())
        total_size += dir_size
        count += 1

        if args.dry_run:
            print(f"  [DRY RUN] Would delete: {job_dir.name} ({dir_size / 1024 / 1024:.1f} MB)")
        else:
            try:
                shutil.rmtree(job_dir)
            except Exception as exc:
                print(f"  ERROR deleting {job_dir.name}: {exc}", file=sys.stderr)

    action = "Would delete" if args.dry_run else "Deleted"
    print(f"\n{action} {count} job directories, {total_size / 1024 / 1024 / 1024:.1f} GB freed")


if __name__ == "__main__":
    main()
