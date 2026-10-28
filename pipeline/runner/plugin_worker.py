"""Runs a candidate's databuilder.py + analysis.py inside the plugin sandbox; JSON lines on stdin/stdout.

Launched by plugin_host.PluginHost as  python -I /runner/plugin_worker.py  inside bwrap, with the candidate at
/cand, the API at /api and the pool view at /data/pool_view.json. Anything the plugins print goes to stderr.
Requests:  {"op": "init"|"next_batch"|"after_step"|"state"|"load"|"ping", ...}
Responses: {"ok": true, "result": ...} | {"ok": false, "error": "..."}
"""
import importlib.util
import json
import os
import sys
import traceback

PROTO = os.fdopen(os.dup(1), "w", buffering=1)
sys.stdout = sys.stderr
sys.path.insert(0, "/api")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jsonable(x, what):
    try:
        return json.loads(json.dumps(x, allow_nan=False))
    except (TypeError, ValueError) as e:
        raise TypeError(f"{what} is not JSON-serializable: {e}") from None


def main():
    builder = analysis = None
    pool = None
    for line in sys.stdin:
        try:
            msg = json.loads(line)
            op = msg["op"]
            if op == "ping":
                res = "pong"
            elif op == "init":
                if pool is None:
                    with open("/data/pool_view.json") as f:
                        pool = json.load(f)
                B = load("cand_databuilder", "/cand/databuilder.py").Builder
                A = load("cand_analysis", "/cand/analysis.py").Analysis
                builder = B(msg["builder_config"], pool, msg["seed"], msg["total_steps"])
                analysis = A(msg["analysis_config"], msg["seed"], msg["total_steps"], msg["inner_probe_ids"])
                res = "ok"
            elif op == "next_batch":
                res = jsonable(builder.next_batch(msg["step"]), "next_batch result")
            elif op == "after_step":
                summary = jsonable(analysis.update(msg["step"], msg["rows"], msg["step_info"], msg["probe_results"]),
                                   "Analysis.update result")
                if not isinstance(summary, dict):
                    raise TypeError("Analysis.update must return a dict")
                builder.observe(msg["step"], summary)
                req = jsonable(analysis.request_probes(msg["step"], msg["budget"]), "request_probes result")
                if not isinstance(req, list):
                    raise TypeError("request_probes must return a list")
                res = {"summary": summary, "requests": req}
            elif op == "state":
                res = {"builder": jsonable(builder.state_dict(), "Builder.state_dict"),
                       "analysis": jsonable(analysis.state_dict(), "Analysis.state_dict")}
            elif op == "load":
                builder.load_state_dict(msg["state"]["builder"])
                analysis.load_state_dict(msg["state"]["analysis"])
                res = "ok"
            else:
                raise ValueError(f"unknown op {op}")
            PROTO.write(json.dumps({"ok": True, "result": res}) + "\n")
        except Exception:  # noqa: BLE001  report every plugin error to the host
            PROTO.write(json.dumps({"ok": False, "error": traceback.format_exc()[-4000:]}) + "\n")
        PROTO.flush()


if __name__ == "__main__":
    main()
