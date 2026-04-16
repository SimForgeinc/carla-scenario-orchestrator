#!/usr/bin/env python3
"""
Benchmark map-aware slot assignment.
Runs a matrix of scenarios and captures timing metrics per job.
Usage: python3 scripts/benchmark_map_affinity.py [--label before|after] [--output FILE]
"""
import argparse, json, sys, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

ORCH_URL = "http://127.0.0.1:18421"

def make_payload(map_name, duration=5.0, topdown_recording=False):
    return {
        "map_name": map_name, "selected_roads": [],
        "actors": [{"id": "bench-ego", "label": "Bench Ego", "kind": "vehicle",
            "role": "ego", "blueprint": "vehicle.tesla.model3",
            "spawn": {"road_id": "37", "s_fraction": 0.3, "lane_id": -1},
            "route": [], "speed_kph": 60, "autopilot": True,
            "placement_mode": "road", "is_static": False, "timeline": []}],
        "duration_seconds": duration, "fixed_delta_seconds": 0.05,
        "topdown_recording": topdown_recording,
        "recording_width": 1280, "recording_height": 720, "recording_fov": 90.0,
    }

def submit_and_wait(payload, timeout=120.0):
    start = time.time()
    res = requests.post(f"{ORCH_URL}/api/jobs", json=payload, timeout=30)
    res.raise_for_status()
    job_id = res.json()["job_id"]
    while time.time() - start < timeout:
        r = requests.get(f"{ORCH_URL}/api/jobs/{job_id}", timeout=10)
        r.raise_for_status()
        data = r.json()
        if data["state"] in ("succeeded", "failed", "cancelled"):
            total_ms = (time.time() - start) * 1000
            diag = {}
            try:
                dr = requests.get(f"{ORCH_URL}/api/jobs/{job_id}/diagnostics", timeout=10)
                if dr.ok: diag = dr.json()
            except Exception: pass
            map_load_ms = 0.0
            for line in diag.get("log_excerpt", "").split("\n"):
                if "Map load:" in line:
                    try: map_load_ms = float(line.split("Map load:")[1].split("s")[0].strip()) * 1000
                    except (ValueError, IndexError): pass
                elif "Loading CARLA map" in line and map_load_ms == 0:
                    map_load_ms = -1
            gpu = data.get("gpu", {})
            return {"job_id": job_id, "map_name": payload["map_name"], "state": data["state"],
                "total_ms": round(total_ms, 1), "map_load_ms": round(map_load_ms, 1),
                "slot_index": gpu.get("slot_index", -1),
                "topdown_recording": payload["topdown_recording"],
                "skipped_actors": len(diag.get("skipped_actors", [])),
                "worker_error": diag.get("worker_error")}
        time.sleep(0.3)
    return {"job_id": job_id, "map_name": payload["map_name"], "state": "timeout",
        "total_ms": timeout * 1000, "map_load_ms": 0, "slot_index": -1,
        "topdown_recording": payload["topdown_recording"], "skipped_actors": 0, "worker_error": "timeout"}

def run_scenario(name, jobs, parallel=False):
    print(f"\n{'='*60}\nSCENARIO: {name}\n{'='*60}", file=sys.stderr)
    results = []
    if parallel:
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = {pool.submit(submit_and_wait, j): i for i, j in enumerate(jobs)}
            for fut in as_completed(futures):
                idx = futures[fut]
                r = fut.result(); r["scenario"] = name; r["job_index"] = idx; results.append(r)
                print(f"  [{idx}] {r['map_name']} slot={r['slot_index']} total={r['total_ms']:.0f}ms load={r['map_load_ms']:.0f}ms {r['state']}", file=sys.stderr)
    else:
        for i, payload in enumerate(jobs):
            r = submit_and_wait(payload); r["scenario"] = name; r["job_index"] = i; results.append(r)
            print(f"  [{i}] {r['map_name']} slot={r['slot_index']} total={r['total_ms']:.0f}ms load={r['map_load_ms']:.0f}ms {r['state']}", file=sys.stderr)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="unlabeled")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    print(f"\nBenchmark: label={args.label}", file=sys.stderr)
    r = requests.get(f"{ORCH_URL}/api/health", timeout=10); r.raise_for_status(); h = r.json()
    free = h.get("total_slots", 0) - h.get("busy_slots", 0)
    print(f"Health: status={h.get('status')} total={h.get('total_slots')} free={free}", file=sys.stderr)
    cap = requests.get(f"{ORCH_URL}/api/capacity", timeout=10).json()
    for s in cap.get("slots", []):
        if s.get("role") == "execution":
            print(f"  Slot {s['slot_index']}: busy={s['busy']} map={s.get('current_map', 'N/A')}", file=sys.stderr)

    all_results = []
    all_results.extend(run_scenario("same_map_x5_vwpoc", [make_payload("VW_Poc") for _ in range(5)]))
    all_results.extend(run_scenario("same_map_x5_town05", [make_payload("Town05") for _ in range(5)]))
    all_results.extend(run_scenario("alternating_maps", [make_payload("VW_Poc"), make_payload("Town05"), make_payload("VW_Poc"), make_payload("Town05")]))
    all_results.extend(run_scenario("concurrent_same_map", [make_payload("VW_Poc") for _ in range(3)], parallel=True))
    all_results.extend(run_scenario("concurrent_diff_maps", [make_payload("VW_Poc"), make_payload("Town05"), make_payload("Town10HD_Opt")], parallel=True))

    for r in all_results:
        r["label"] = args.label; r["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    out = open(args.output, "w") if args.output else sys.stdout
    for r in all_results:
        out.write(json.dumps(r) + "\n")
    if args.output: out.close(); print(f"\nResults: {args.output}", file=sys.stderr)

    print(f"\n{'='*80}\nSUMMARY — {args.label}\n{'='*80}", file=sys.stderr)
    print(f"{'Scenario':<30} {'Jobs':>4} {'Avg Total':>10} {'Avg Load':>10} {'Fails':>5}", file=sys.stderr)
    scenarios = {}
    for r in all_results:
        scenarios.setdefault(r["scenario"], []).append(r)
    for name, runs in scenarios.items():
        avg_t = sum(r["total_ms"] for r in runs) / len(runs)
        loads = [r["map_load_ms"] for r in runs if r["map_load_ms"] >= 0]
        avg_l = sum(loads) / max(1, len(loads))
        fails = sum(1 for r in runs if r["state"] != "succeeded")
        print(f"{name:<30} {len(runs):>4} {avg_t:>9.0f}ms {avg_l:>9.0f}ms {fails:>5}", file=sys.stderr)

    print(f"\nSlot state after:", file=sys.stderr)
    cap = requests.get(f"{ORCH_URL}/api/capacity", timeout=10).json()
    for s in cap.get("slots", []):
        if s.get("role") == "execution":
            print(f"  Slot {s['slot_index']}: map={s.get('current_map', 'N/A')}", file=sys.stderr)

if __name__ == "__main__":
    main()
