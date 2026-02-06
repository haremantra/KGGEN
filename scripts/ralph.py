"""RALPH Loop — Task orchestrator for KGGEN codebase merge.

R(ead/Reconnaissance) → A(nalyze) → L(ayout) → P(roduce) → H(alt/Handoff)

Tracks task state, enforces dependency order, and persists progress.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


STATE_FILE = Path(__file__).parent.parent / ".ralph_state.json"
DEFAULT_MANIFEST = Path(__file__).parent / "ralph_manifest.yaml"


def load_manifest(manifest_path: Path | None = None) -> dict:
    path = manifest_path or DEFAULT_MANIFEST
    with open(path) as f:
        return yaml.safe_load(f)["tasks"]


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_status(state: dict, task_id: str) -> str:
    return state.get(task_id, {}).get("status", "pending")


def set_status(state: dict, task_id: str, status: str, note: str = ""):
    if task_id not in state:
        state[task_id] = {}
    state[task_id]["status"] = status
    state[task_id][f"{status}_at"] = datetime.now(timezone.utc).isoformat()
    if note:
        state[task_id]["note"] = note
    save_state(state)


def deps_met(manifest: dict, state: dict, task_id: str) -> bool:
    deps = manifest[task_id].get("depends_on", [])
    return all(get_status(state, d) == "completed" for d in deps)


def report(manifest: dict, state: dict):
    print("\n=== RALPH Merge Report ===\n")
    layers = sorted(set(t["layer"] for t in manifest.values()))
    for layer in layers:
        print(f"--- Layer {layer} ---")
        for tid, tdef in manifest.items():
            if tdef["layer"] != layer:
                continue
            status = get_status(state, tid)
            symbol = {"pending": " ", "in_progress": "~", "completed": "+", "halted": "!"}
            marker = symbol.get(status, "?")
            deps = tdef.get("depends_on", [])
            dep_str = f"  (deps: {', '.join(deps)})" if deps else ""
            halt_str = "  [HALT]" if tdef.get("halt") else ""
            print(f"  [{marker}] {tid}: {status}{dep_str}{halt_str}")
        print()

    total = len(manifest)
    done = sum(1 for t in manifest if get_status(state, t) == "completed")
    print(f"Progress: {done}/{total} tasks completed")
    if done == total:
        print("ALL TASKS COMPLETED")


def next_task(manifest: dict, state: dict) -> str | None:
    for tid, tdef in sorted(manifest.items(), key=lambda x: x[1]["layer"]):
        status = get_status(state, tid)
        if status in ("completed", "halted"):
            continue
        if deps_met(manifest, state, tid):
            return tid
    return None


def run_task(manifest: dict, state: dict, task_id: str):
    tdef = manifest[task_id]
    print(f"\n>>> Task: {task_id}")
    print(f"    {tdef['description']}")
    print(f"    File: {tdef['file']}")
    print(f"    Action: {tdef['action']}")

    if tdef.get("halt"):
        print(f"\n    HALT: {tdef.get('halt_reason', 'Requires human input')}")
        set_status(state, task_id, "halted", tdef.get("halt_reason", ""))
        return

    set_status(state, task_id, "in_progress")
    print(f"    Status: in_progress (implement manually, then mark completed)")


def main():
    # Parse --manifest flag
    manifest_path = None
    if "--manifest" in sys.argv:
        idx = sys.argv.index("--manifest")
        if idx + 1 < len(sys.argv):
            manifest_path = Path(sys.argv[idx + 1])
            if not manifest_path.is_absolute():
                manifest_path = Path.cwd() / manifest_path

    manifest = load_manifest(manifest_path)
    state = load_state()

    if "--report" in sys.argv:
        report(manifest, state)
        return

    if "--complete" in sys.argv:
        idx = sys.argv.index("--complete")
        if idx + 1 < len(sys.argv):
            tid = sys.argv[idx + 1]
            if tid in manifest:
                set_status(state, tid, "completed")
                print(f"Marked {tid} as completed")
                report(manifest, state)
            else:
                print(f"Unknown task: {tid}")
        return

    if "--task" in sys.argv:
        idx = sys.argv.index("--task")
        if idx + 1 < len(sys.argv):
            tid = sys.argv[idx + 1]
            if tid in manifest:
                if deps_met(manifest, state, tid):
                    run_task(manifest, state, tid)
                else:
                    deps = manifest[tid].get("depends_on", [])
                    pending = [d for d in deps if get_status(state, d) != "completed"]
                    print(f"Cannot run {tid}: waiting on {pending}")
            else:
                print(f"Unknown task: {tid}")
        return

    if "--resume" in sys.argv:
        tid = next_task(manifest, state)
        if tid:
            print(f"Next available task: {tid}")
            run_task(manifest, state, tid)
        else:
            print("No tasks available (all completed or blocked)")
        return

    if "--dry-run" in sys.argv:
        print("Dry run — execution order:")
        executed = set()
        while True:
            found = False
            for tid, tdef in sorted(manifest.items(), key=lambda x: x[1]["layer"]):
                if tid in executed:
                    continue
                deps = tdef.get("depends_on", [])
                if all(d in executed for d in deps):
                    halt = " [HALT]" if tdef.get("halt") else ""
                    print(f"  {len(executed)+1}. {tid} (L{tdef['layer']}){halt}")
                    executed.add(tid)
                    found = True
                    break
            if not found:
                break
        return

    # Default: show report
    report(manifest, state)


if __name__ == "__main__":
    main()
