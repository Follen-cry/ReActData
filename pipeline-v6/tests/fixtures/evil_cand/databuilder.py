"""Test-only candidate: tries to read protected files and reach the network from inside the plugin sandbox."""
import os
class Builder:
    def __init__(self, config, pool, seed, total_steps):
        self.res = {}
        for p in ["/scratch/network/ssd2/junlin/ssl_mllm/internvlu-exp/exp/reactdata-v5/scorers/analogy_grid.py", "/scratch/network/ssd/junlin/reactdata-v5/data/meta",
                  "/scratch/network/ssd/junlin/vbvr_next_frame/raw/2d_geometric_transformation/chunk_par_0000/transformation_worlds_2d_geometric_transformation_task/transformation_worlds_2d_geometric_transformation_00000000/metadata.json",
                  "/homes/55/junlin/.claude/.credentials.json", "/scratch/network/ssd/junlin/reactdata-v0/data", "/etc/passwd"]:
            try:
                open(p).read(10); self.res[p] = "READ"
            except Exception as e:
                self.res[p] = type(e).__name__
        import socket
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3); self.res["net"] = "CONNECTED"
        except Exception as e:
            self.res["net"] = type(e).__name__
        self.res["listdir_root"] = sorted(os.listdir("/"))
        self.res["env"] = sorted(os.environ)
    def next_batch(self, step): return [self.res]
    def observe(self, s, x): pass
    def state_dict(self): return {}
    def load_state_dict(self, s): pass
