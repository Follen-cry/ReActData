"""CPU end-to-end test of the design-v3 state machine (protocol.yaml) without GPUs or LLM calls.

    python tests/test_v3_flow.py

Fakes: Runner.execute writes synthetic eval scores (true program effect + per-run training noise sd 0.02 + per-item
noise), call_agent writes candidate programs (one with a true +0.08 effect, one config-only copy that must be a
duplicate), audit_run never flags. Real: checks (sandboxed simulation), state machine, statistics, selection, reports.
Asserts the phase order, 10 serial candidates with lineage, the duplicate rule, confirmation of the top 3 on paired
seeds, selection of the true +0.08 program, P3 multi-seed, and the v3 report.
"""
import json
import os
import random
import shutil
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(EXP, "runner"))
import state_machine as SM  # noqa: E402
from common import MANIFESTS, RUNS, read_json, read_jsonl, write_json  # noqa: E402

RUN = "_test_v3flow"
EFFECTS = {}                                     # code sha -> true effect
DEV = [r["id"] for r in read_jsonl(os.path.join(MANIFESTS, "dev.jsonl"))]
TEST = [r["id"] for r in read_jsonl(os.path.join(MANIFESTS, "test.jsonl"))]
TASK = {r["id"]: r["task_name"] for f in ("dev.jsonl", "test.jsonl") for r in read_jsonl(os.path.join(MANIFESTS, f))}
BASE_ITEM = {i: 0.2 + 0.5 * random.Random(i).random() for i in DEV + TEST}


def effect_of(cand_dir):
    if not cand_dir:
        return -0.4                               # Base
    src = open(os.path.join(cand_dir, "databuilder.py")).read()
    for line in src.splitlines():
        if line.startswith("# EFFECT="):
            return float(line.split("=")[1])
    return 0.0


def fake_execute(self, spec):
    run = read_json(spec)
    d = run["run_dir"]
    if os.path.exists(os.path.join(d, "DONE.json")):
        return "done"
    eff = effect_of(run["candidate_dir"])
    rng = random.Random(zlib.crc32(f"{run['run_id']}".encode()))
    run_noise = rng.gauss(0, 0.02) if run["steps"] > 0 else 0.0
    cost = {"load_wall_s": 60, "train_wall_s": 600, "world_size": 4, "train_gpu_s": 2400, "probe_gpu_s": 0, "eval_gpu_s": 1200,
            "rows": 1600, "unique_clips": 1600, "repeated_rows": 0, "max_clip_uses": 1, "vit_images": 1600, "vit_tokens": 409600,
            "seq_tokens": 1, "plugin_wall_s": 1, "evals": {}}
    for ev in run["evals"]:
        ids = DEV if ev["name"] == "dev" else TEST
        for s in ev["seeds"]:
            g = random.Random(zlib.crc32(f"{run['run_id']}:{ev['name']}:{s}".encode()))
            items = [{"id": i, "task": TASK[i], "score": min(1.0, max(0.0, BASE_ITEM[i] + eff + run_noise + g.gauss(0, 0.05))),
                      "sub": {"E_cell": 0.5, "F_cell": 0.5, "F_transform": 0.5}, "valid": True, "categories": ["F_wrong"]} for i in ids]
            by_task = {}
            for t in sorted({x["task"] for x in items}):
                xs = [x for x in items if x["task"] == t]
                by_task[t] = {"n": len(xs), "mean": sum(x["score"] for x in xs) / len(xs), "n_invalid": 0,
                              "sub_mean": {"E_cell": 0.5, "F_cell": 0.5, "F_transform": 0.5}}
            out = os.path.join(d, "eval", ev["name"], f"g{s}")
            os.makedirs(os.path.join(out, "images"), exist_ok=True)
            for x in items:                         # failure-case galleries link generated images
                open(os.path.join(out, "images", x["id"] + ".png"), "wb").close()
            tm = {t: v["mean"] for t, v in by_task.items()}
            write_json(os.path.join(out, "scores.json"), {"items": items, "mean": sum(tm.values()) / len(tm), "n_invalid": 0,
                                                          "by_task": by_task, "task_means": tm, "sub_mean": {"E_cell": 0.5}})
            cost["evals"][f"{ev['name']}/g{s}"] = {"gen_wall_s": 300}
    write_json(os.path.join(d, "cost.json"), cost)
    if run["steps"] > 0:                          # a small fake checkpoint, to test the disk policy
        ck = os.path.join(d, "checkpoints", "step_000100")
        os.makedirs(ck, exist_ok=True)
        for f in ("trainable.pt", "optimizer.pt", "meta.json"):
            open(os.path.join(ck, f), "wb").write(b"0" * 1024)
    write_json(os.path.join(d, "DONE.json"), {})
    open(os.path.join(d, "exit_code"), "w").write("0")
    self.event("job_finished", job=run["run_id"], status="done", wall_s=0)
    return "done"


PLAN = {  # candidate index -> (effect, kind)
    1: (0.0, "code"), 2: (-0.03, "code"), 3: (-0.03, "config_only"), 4: (0.08, "code"), 5: (0.01, "code"),
    6: (-0.05, "code"), 7: (0.02, "code"), 8: (0.0, "code"), 9: (-0.02, "code"), 10: (0.03, "code")}
CALLS = {"n": 0}
INPUTS, REPAIRS = [], []


def fake_agent(prompt, work, inp, log_dir, **kw):
    if "Repair step" not in prompt:
        CALLS["n"] += 1
    k = CALLS["n"]
    if "Repair step" in prompt:                   # repair: only fix proposal.md
        open(os.path.join(work, "proposal.md"), "w").write(
            f"## Evidence\nfixed {k}: dev_scale_003 dev_scale_004 dev_move_005\n## Hypothesis\nh\n## Program\np\n"
            "## What analysis.py measures\nm\n## Expected result\ne\n")
        REPAIRS.append(k)
        return {"ok": True, "result": "fake repair", "models": ["fake"], "cost_usd": 0.0, "num_turns": 1, "wall_s": 0.0,
                "session_id": "fake-session", "resumed_from": kw.get("resume")}
    os.makedirs(log_dir, exist_ok=True)
    assert os.path.exists(os.path.join(inp, "report.md"))
    if k > 1:
        assert "Search budget:" in prompt and "Training noise" in open(os.path.join(inp, "report.md")).read()
    cand = sorted(x for x in os.listdir(os.path.join(inp, "candidates")))[-1] if os.path.isdir(os.path.join(inp, "candidates")) else None
    eff, kind = PLAN.get(k, (0.0, "code"))
    if kind == "config_only":                      # leave code as the workspace start (previous candidate), change config only
        open(os.path.join(work, "candidate.yaml"), "a").write(f"# salt s{k}\n")
    else:
        src = open(os.path.join(work, "databuilder.py")).read().splitlines()
        src = [l for l in src if not l.startswith("# EFFECT=") and not l.startswith("# variant")]
        src = [f"# EFFECT={eff}", f"# variant {k}"] + src
        open(os.path.join(work, "databuilder.py"), "w").write("\n".join(src) + "\n")
    INPUTS.append(sorted(os.listdir(inp)))
    if k == 2 and "Repair step" not in prompt:   # first attempt of c002: a proposal without the required sections
        open(os.path.join(work, "proposal.md"), "w").write("no sections\n")
    else:
        open(os.path.join(work, "proposal.md"), "w").write(
            f"## Evidence\nfake {k}: dev_scale_000 dev_scale_001 dev_move_002\n## Hypothesis\nh\n## Program\np\n"
            "## What analysis.py measures\nm\n## Expected result\ne\n")
        CALLS["n"] += 0
    if "Repair step" in prompt:
        REPAIRS.append(k)
    return {"ok": True, "result": "fake", "models": ["fake"], "cost_usd": 0.0, "num_turns": 1, "wall_s": 0.0,
            "session_id": "fake-session", "resumed_from": kw.get("resume")}


def main():
    shutil.rmtree(os.path.join(RUNS, RUN), ignore_errors=True)
    shutil.rmtree(os.path.join(EXP, "records", RUN), ignore_errors=True)
    SM.Runner.execute = fake_execute
    SM.call_agent = fake_agent
    SM.audit_run = lambda d: {"flag": False, "n_rows": 0}
    r = SM.Runner(RUN, False, None, os.path.join(EXP, "protocol.yaml"), None)
    r.state["done_upto"] = "P0"                    # P0 pilot needs a GPU; its estimates are faked
    r.state["budget_start"] = __import__("time").time()
    r.state["estimates"] = {"candidate_job_h": 0.3}
    r.run()
    st = r.state
    order = [e["phase"] for e in read_jsonl(os.path.join(RUNS, RUN, "events.jsonl")) if e["kind"] == "phase_start"]
    C = st["candidates"]
    print("phases", order)
    print("noise", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st["noise"].items() if k != "run_scores"})
    for k in sorted(C):
        print(k, C[k]["status"], C[k].get("parent"), round(C[k].get("dev_mean") or 0, 4), round((C[k].get("comparison") or {}).get("D", 0), 4))
    print("confirm", {k: (round(v["D"], 4), round(v["ci_lo"], 4)) for k, v in st["confirm"].items()})
    print("selected", st["selection"]["selected"], "| p3", {k: v["status"] for k, v in st["p3"].items()})
    checks = {
        "phase order": order == ["P1", "P2", "CONFIRM", "SELECT", "P3", "REPORT"],
        "10 candidates": len([k for k in C if k != "c000"]) == 10,
        "config-only c003 is a duplicate": C["c003"]["status"] == "duplicate",
        "lineage: c002 starts from c001": C["c002"].get("parent") == "c001",
        "lineage: c005 starts from c004": C["c005"].get("parent") == "c004",
        "baseline replicates": len(C["c000"]["replicates"]) == 5,
        "confirm top 3": len(st["confirm"]) == 3,
        "true +0.08 program confirmed and selected": st["selection"]["selected"] == "c004",
        "P3 3 formal seeds x 2 arms + Base": len(st["p3"]) == 7,
        "report written": os.path.exists(os.path.join(RUNS, RUN, "report.md")),
    }
    jd = os.path.join(RUNS, RUN, "jobs")
    left = sorted(n for n in os.listdir(jd) if os.path.isdir(os.path.join(jd, n, "checkpoints", "step_000100")))
    best = os.path.basename(C[r.best_so_far()]["run_dir"])
    print("checkpoints left:", left, "| best screening run:", best)
    checks["disk: only the best screening run + formal runs keep a checkpoint"] = \
        set(left) == {best} | {n for n in os.listdir(jd) if n.startswith("formal_")}
    checks["v5: screening D is against the replicate mean"] = all((C[k].get("comparison") or {}).get("reference") == "replicate_mean"
                                                                   for k in C if k != "c000" and C[k]["status"] == "trained")
    checks["v5: dev_items + analysis_traces given to the agent"] = all("dev_items" in x and "analysis_traces" in x for x in INPUTS)
    checks["v5: proposal without sections was sent back for repair (c002)"] = REPAIRS == [2] and C["c002"]["status"] == "trained"
    checks["disk: no optimizer state left"] = not any(os.path.exists(os.path.join(jd, n, "checkpoints", "step_000100", "optimizer.pt"))
                                                       for n in left)
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    print(open(os.path.join(RUNS, RUN, "report.md")).read()[:2500])
    shutil.rmtree(os.path.join(RUNS, RUN), ignore_errors=True)
    shutil.rmtree(os.path.join(EXP, "records", RUN), ignore_errors=True)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
