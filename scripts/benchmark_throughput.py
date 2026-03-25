#!/usr/bin/env python3
"""Benchmark: Submit N simulation jobs and measure throughput."""
from __future__ import annotations
import argparse, json, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import requests

ORCH_URL = "http://127.0.0.1:18421"
SEP = "=" * 60

@dataclass
class JobResult:
    job_id: str
    state: str
    submit_time: float
    end_time: float
    slot_index: int | None = None
    error: str | None = None
    @property
    def latency(self): return self.end_time - self.submit_time

def make_payload(map_name: str) -> dict:
    return {
        "map_name": map_name,
        "selected_roads": [{"id": "1277", "name": "Road 1277"}],
        "actors": [{
            "id": "bench-ego-1",
            "label": "Benchmark Ego",
            "kind": "vehicle",
            "role": "ego",
            "blueprint": "vehicle.tesla.model3",
            "spawn": {"road_id": "1277", "s_fraction": 0.3},
            "autopilot": True,
            "speed_kph": 60.0,
        }],
        "duration_seconds": 3.0,
        "fixed_delta_seconds": 0.05,
        "topdown_recording": False,
    }

def submit_and_wait(job_num, map_name, timeout=120):
    submit_time = time.time()
    try:
        resp = requests.post(f"{ORCH_URL}/api/jobs", json=make_payload(map_name), timeout=10)
        resp.raise_for_status()
        job_id = resp.json()["job_id"]
    except Exception as exc:
        return JobResult(job_id=f"err-{job_num}", state="submit_failed",
                        submit_time=submit_time, end_time=time.time(), error=str(exc))
    terminal = {"succeeded", "failed", "cancelled"}
    while True:
        if time.time() - submit_time > timeout:
            return JobResult(job_id=job_id, state="timeout", submit_time=submit_time,
                           end_time=time.time(), error=f"Timeout after {timeout}s")
        try:
            r = requests.get(f"{ORCH_URL}/api/jobs/{job_id}", timeout=5)
            if r.status_code == 200:
                j = r.json()
                if j["state"] in terminal:
                    slot = j.get("gpu", {}).get("slot_index") if j.get("gpu") else None
                    return JobResult(job_id=job_id, state=j["state"], submit_time=submit_time,
                                   end_time=time.time(), slot_index=slot, error=j.get("error"))
        except: pass
        time.sleep(0.5)

def pct(data, p):
    if not data: return 0.0
    k = (len(data)-1)*(p/100); f=int(k); c=min(f+1,len(data)-1)
    return data[f]+(k-f)*(data[c]-data[f])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=100)
    parser.add_argument("--map", type=str, default="VW_Poc")
    parser.add_argument("--concurrent", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    print(f"\n{SEP}")
    print(f"  CARLA Throughput Benchmark")
    print(f"{SEP}")
    print(f"  Jobs: {args.jobs}, Map: {args.map}, Concurrent: {args.concurrent}\n")

    try:
        h = requests.get(f"{ORCH_URL}/api/health", timeout=5).json()
        c = requests.get(f"{ORCH_URL}/api/capacity", timeout=5).json()
        print(f"  Status: {h['status']}, Slots: {c['total_slots']}t/{c['free_slots']}f/{c.get('unavailable_slots',0)}u")
    except Exception as e:
        print(f"  ERROR: {e}"); sys.exit(1)

    start = time.time()
    results = []
    print(f"\n  Submitting {args.jobs} jobs...")
    with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
        futures = {pool.submit(submit_and_wait, i, args.map, args.timeout): i for i in range(args.jobs)}
        done = 0
        for f in as_completed(futures):
            results.append(f.result())
            done += 1
            step = max(1, args.jobs // 10)
            if done % step == 0 or done == args.jobs:
                e = time.time() - start
                print(f"    [{done}/{args.jobs}] {e:.1f}s, {done/e*60:.1f}/min")

    total = time.time() - start
    ok = [r for r in results if r.state == "succeeded"]
    fail = [r for r in results if r.state == "failed"]
    tout = [r for r in results if r.state == "timeout"]
    lats = sorted([r.latency for r in ok])
    slots = {}
    for r in results:
        if r.slot_index is not None:
            slots[r.slot_index] = slots.get(r.slot_index, 0) + 1

    print(f"\n{SEP}")
    print(f"  RESULTS")
    print(f"{SEP}")
    print(f"  Time: {total:.1f}s | Rate: {len(results)/total*60:.1f}/min")
    print(f"  OK: {len(ok)} | Fail: {len(fail)} | Timeout: {len(tout)}")
    if lats:
        print(f"  Latency: p50={pct(lats,50):.2f}s p95={pct(lats,95):.2f}s p99={pct(lats,99):.2f}s")
        print(f"           min={min(lats):.2f}s max={max(lats):.2f}s mean={statistics.mean(lats):.2f}s")
    if slots:
        print(f"  Slot distribution:")
        mx = max(slots.values())
        for s in sorted(slots):
            bar = "#" * (slots[s] * 40 // mx)
            print(f"    slot {s}: {bar} {slots[s]}")
    if fail[:3]:
        print(f"  Errors (first 3):")
        for r in fail[:3]:
            print(f"    {r.job_id}: {(r.error or '')[:100]}")

    sr = len(ok) / len(results) * 100 if results else 0
    v = "PASS" if sr >= 95 else "MARGINAL" if sr >= 80 else "FAIL"
    print(f"\n  VERDICT: {v} ({sr:.0f}%)")
    print(SEP)

    out = f"/tmp/bench_{int(start)}.json"
    with open(out, "w") as fp:
        json.dump({
            "config": {"jobs": args.jobs, "map": args.map, "concurrent": args.concurrent},
            "total_s": total, "rate_per_min": len(results)/total*60,
            "ok": len(ok), "fail": len(fail), "timeout": len(tout),
            "p50": pct(lats, 50) if lats else None,
            "p95": pct(lats, 95) if lats else None,
            "p99": pct(lats, 99) if lats else None,
            "slots": slots,
            "results": [{"id": r.job_id, "state": r.state, "lat": round(r.latency, 2),
                        "slot": r.slot_index, "err": r.error} for r in results],
        }, fp, indent=2)
    print(f"  Results: {out}\n")

if __name__ == "__main__":
    main()
