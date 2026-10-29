import os
import shutil
import time
from experiment.runtime.io import read_json, sha256_file

def inherit_run(store, runs_root, smoke, src_id: str) -> None:
    """Start this run as a continuation of a finished run: candidates, baselines, delta and the agent session.

    Source records are referenced, never modified: candidate code and job directories stay in the source run, the
    agent HOME (session files) is copied, and the source's base_test / formal_c000 jobs are linked for reuse."""
    src_root = os.path.join(runs_root, src_id)
    src = read_json(os.path.join(src_root, "state.json"))
    if src.get("phase") != "DONE":
        raise SystemExit(f"{src_id} is not finished (phase {src.get('phase')})")
    if src.get("smoke") != smoke:
        raise SystemExit("smoke and non-smoke runs cannot continue each other")
    import copy
    cands = copy.deepcopy(src["candidates"])
    for c in cands.values():
        c["inherited_from"] = src_id
    updates = {"candidates": cands}
    for k in ("base_dev", "delta", "estimates", "first_step_check", "agent_session", "last_memory",
              "noise", "baseline_mean_items", "baseline_sd_items"):
        if k in src:
            updates[k] = copy.deepcopy(src[k])
    rounds = [c.get("round", 0) or 0 for c in cands.values()]
    idx = [int(k[1:]) for k in cands]
    updates["round_offset"] = max(rounds) if rounds else 0
    updates["cand_offset"] = max(idx) if idx else 0
    if os.path.isdir(os.path.join(src_root, "agent_home")):
        shutil.copytree(os.path.join(src_root, "agent_home"), os.path.join(store.root, "agent_home"), symlinks=True, dirs_exist_ok=True)
    linked = []
    for name in os.listdir(os.path.join(src_root, "jobs")):
        if (name == "base_test" or name.startswith("formal_c000-")) and os.path.exists(os.path.join(src_root, "jobs", name, "DONE.json")):
            os.makedirs(os.path.join(store.root, "jobs"), exist_ok=True)
            dst = os.path.join(store.root, "jobs", name)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(src_root, "jobs", name), dst)
            linked.append(name)
    updates["inherited_from"] = {"run": src_id, "state_sha256": sha256_file(os.path.join(src_root, "state.json")),
                                    "protocol_sha256": src["run_json"]["protocol_sha256"], "runner_sha256": src["run_json"]["runner_sha256"],
                                    "candidates": sorted(cands), "agent_session": src.get("agent_session"), "linked_jobs": linked,
                                    "rounds_done": updates["round_offset"]}
    updates.update(done_upto="P1", budget_start=time.time(), phase="INHERIT")
    store.commit("continued_from", updates=updates, event_data=updates["inherited_from"])
