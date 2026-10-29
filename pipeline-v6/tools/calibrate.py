"""Pre-protocol calibration (reactdata-v6): how fast does the 8-step / 128-row fixed baseline learn at a given lr / warmup?

Runs the fixed baseline c000 (seed 1000) with each setting plus the untrained Base on dev, on two 4-GPU slots, and
prints the dev scores. Used once to choose training.overrides / warmup_steps before protocol_v6.yaml was frozen; the
outputs stay under <SCRATCH>/calibration/.

    python tools/calibrate.py
"""
import json
import os
import sys
import time
from dataclasses import asdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "runner"))
from common import API, DATA, EXP, MANIFESTS, RUNS, SCRATCH, read_json, write_json  # noqa: E402
from experiment.contracts import RunPaths  # noqa: E402
import jobs as J  # noqa: E402

ROOT = os.path.join(SCRATCH, "calibration")
SETTINGS = {  # name: (steps, overrides, warmup)
    "base": (0, [], 6),
    "lr1x_w6": (8, [], 6),
    "lr5x_w2": (8, ["optim.lr=5.0e-5", "optim.gen_decoder_lr=2.5e-4"], 2),
    "lr10x_w2": (8, ["optim.lr=1.0e-4", "optim.gen_decoder_lr=5.0e-4"], 2),
    "lr20x_w2": (8, ["optim.lr=2.0e-4", "optim.gen_decoder_lr=1.0e-3"], 2),     # stopped: lr5x_w2 already collapsed
    "lr2x_w3": (8, ["optim.lr=2.0e-5", "optim.gen_decoder_lr=1.0e-4"], 3),
    "lr1x_w2": (8, [], 2),
    # not calibration: GPU check of the probe path (tasks/reactdata/gpu_probes.py) with the chosen schedule, 8 dev rows
    "probe_test": (8, ["optim.lr=2.0e-5", "optim.gen_decoder_lr=1.0e-4"], 3),
}
CANDIDATE = {"probe_test": os.path.join(EXP, "tests", "fixtures", "v6_probe_cand")}
MAX_ROWS = {"probe_test": 8}
SLOTS = [J.Slot("P1", "torrnode12", "0,1,2,3", 29881), J.Slot("P2", "torrnode12", "4,5,6,7", 29882)]
if os.environ.get("CALIB_SLOTS") == "node14":
    SLOTS = [J.Slot("P3", "torrnode14", "4,5,6,7", 29883)]
elif os.environ.get("CALIB_SLOTS") == "node12a":
    SLOTS = [J.Slot("P1", "torrnode12", "0,1,2,3", 29881)]


def spec(name):
    steps, ov, wu = SETTINGS[name]
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    paths = RunPaths(EXP, API, DATA, MANIFESTS, RUNS, os.path.join(EXP, "protocol.yaml"))
    p = os.path.join(d, "run.json")
    write_json(p, {"run_id": name, "run_dir": d, "candidate_dir": CANDIDATE.get(name, os.path.join(EXP, "seeds", "fixed_f0_final")), "run_seed": 1000,
                   "steps": steps, "evals": [{"name": "dev", "file": os.path.join(MANIFESTS, "dev.jsonl"), "seeds": [0]}],
                   "max_eval_rows": MAX_ROWS.get(name), "gpus": 4, "resources": asdict(paths), "overrides": ov, "warmup_steps": wu})
    return p


def main():
    todo = [n for n in (sys.argv[1:] or SETTINGS) if not os.path.exists(os.path.join(ROOT, n, "DONE.json"))]
    running = []
    while todo or running:
        for s in SLOTS:
            if s.job and J.status(s.job) != "running":
                print(time.strftime("%H:%M:%S"), "finished", s.job["run_dir"], J.status(s.job), flush=True)
                running.remove(s.job["run_dir"])
                s.job = None
            if s.job is None and todo and J.gpus_idle(s.node, s.gpus):
                n = todo.pop(0)
                J.launch(s, spec(n))
                running.append(s.job["run_dir"])
                print(time.strftime("%H:%M:%S"), "launched", n, s.name, flush=True)
        time.sleep(20)
    for n in SETTINGS:
        f = os.path.join(ROOT, n, "eval", "dev", "g0", "scores.json")
        if os.path.exists(f):
            sc = read_json(f)
            print(n, round(sc["mean"], 4), {k: round(v, 4) for k, v in sc["task_means"].items()}, "invalid", sc["n_invalid"])


if __name__ == "__main__":
    main()
