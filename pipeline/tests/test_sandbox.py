"""Plugin and agent sandbox isolation tests (no GPU, no LLM).  python tests/test_sandbox.py"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "runner"))
from common import DATA, SCORER_DIR  # noqa: E402
SCORER_FILE = os.path.join(SCORER_DIR, "analogy_grid.py")
from plugin_host import PluginHost  # noqa: E402
from agent_call import _bwrap  # noqa: E402

PROTECTED = [SCORER_FILE, os.path.join(DATA, "meta"), os.path.join(DATA, "test"), os.path.expanduser("~/.claude/.credentials.json"),
             "/scratch/network/ssd/junlin/vbvr_next_frame/raw/shape_color_then_scale", "/scratch/network/ssd/junlin/vbvr_target_pred"]


def test_plugin():
    t = tempfile.mkdtemp()
    h = PluginHost(os.path.join(HERE, "evil_cand"), os.path.join(DATA, "pool_view.json"), os.path.join(t, "scr"), os.path.join(t, "log"),
                   time_limit=120)
    h.call("init", timeout=120, builder_config={}, analysis_config={}, seed=1, total_steps=1, inner_probe_ids=[])
    res = h.call("next_batch", step=0)[0]
    h.close()
    bad = {k: v for k, v in res.items() if v == "READ" or v == "CONNECTED"}
    assert not bad, bad
    print("plugin sandbox ok:", {k: v for k, v in res.items() if k not in ("listdir_root", "env")})


def test_agent_mounts():
    t = tempfile.mkdtemp()
    for d in ("home", "work", "input"):
        os.makedirs(os.path.join(t, d))
    cmd = _bwrap(os.path.join(t, "home"), os.path.join(t, "work"), os.path.join(t, "input"))
    for p in PROTECTED + [os.path.expanduser("~/.claude")]:
        r = subprocess.run(cmd + ["/usr/bin/test", "-e", p], env={})
        assert r.returncode != 0, f"visible in agent sandbox: {p}"
    print("agent sandbox ok: protected paths absent")


if __name__ == "__main__":
    test_plugin()
    test_agent_mounts()
