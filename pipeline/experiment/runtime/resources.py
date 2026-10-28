"""Time reservations and disk retention, independent of task payloads."""
import os
import shutil
import threading
import time

class ResourceManager:
    PEAK_JOB_GB = 25.0

    def __init__(self, store, protocol, policy):
        self.store = store
        self.root, self.state, self.P = store.root, store.state, protocol
        self.event, self.policy = store.event, policy
        self.disk_lock = threading.Lock()

    def best_so_far(self):
        return self.policy.best_so_far(self.state)

    def elapsed_h(self) -> float:
        return (time.time() - self.state["budget_start"]) / 3600 if "budget_start" in self.state else 0.0


    def budget_left_h(self) -> float:
        return self.P["budget"]["wall_hours"] - self.elapsed_h()


    def search_allowed(self, est_h: float) -> bool:
        b = self.P["budget"]
        return self.elapsed_h() + est_h <= b["wall_hours"] - b["reserve_final_hours"] - b["buffer_hours"]


    def disk_gb(self) -> float:
        tot = 0
        for dp, dn, fn in os.walk(self.root):            # symlinks (inherited jobs) are not followed / not counted
            for f in fn:
                p = os.path.join(dp, f)
                if not os.path.islink(p):
                    try:
                        tot += os.lstat(p).st_size
                    except OSError:
                        pass
        return tot / 1e9


    def _disk_safe(self, **kw) -> None:
        try:
            self.manage_disk(**kw)
        except Exception as e:  # noqa: BLE001  disk housekeeping must not fail a job
            self.store.update_state(failures=[{"kind": "disk_manage_error", "detail": str(e)[:300]}])


    def manage_disk(self, before_launch: bool = False, protect=()) -> None:
        """Keep the final trainable weights of the current best candidate's screening run and of every formal (P3) run;
        delete the checkpoints of every other job that is final (DONE.json or FAILED.json). Running jobs, jobs waiting
        for an infrastructure retry and the dirs in ``protect`` (a job whose result is not recorded yet) are never
        touched. Called after every job (search, confirmation, P3) and before every launch; enforced against
        budget.disk_gb when the protocol sets it."""
        with self.disk_lock:
            self._manage_disk(before_launch, {os.path.abspath(x) for x in protect})


    def _manage_disk(self, before_launch: bool, protect: set) -> None:
        limit = self.P["budget"].get("disk_gb")
        jobs_dir = os.path.join(self.root, "jobs")
        if not limit or not os.path.isdir(jobs_dir):
            return
        best = self.state["candidates"].get(self.best_so_far(), {}).get("run_dir")
        freed = 0.0

        def prune(protect_best: bool):
            nonlocal freed
            for name in sorted(os.listdir(jobs_dir)):
                d = os.path.join(jobs_dir, name)
                ck = os.path.join(d, "checkpoints")
                final = os.path.exists(os.path.join(d, "DONE.json")) or os.path.exists(os.path.join(d, "FAILED.json"))
                if os.path.islink(d) or not os.path.isdir(ck) or not final or os.path.abspath(d) in protect:
                    continue                              # inherited, no checkpoints, running / retry pending, or protected
                steps = sorted(x for x in os.listdir(ck) if x.startswith("step_") and not x.endswith(".tmp"))
                keep = protect_best and (os.path.abspath(d) == os.path.abspath(best or "") or name.startswith("formal_"))
                for st in steps:
                    full = os.path.join(ck, st)
                    if keep and st == steps[-1]:
                        opt = os.path.join(full, "optimizer.pt")
                        if os.path.exists(opt):
                            freed += os.path.getsize(opt) / 1e9
                            os.remove(opt)
                        continue
                    freed += sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(full) for f in fs) / 1e9
                    shutil.rmtree(full, ignore_errors=True)

        prune(protect_best=True)
        used = self.disk_gb()
        need = used + (self.PEAK_JOB_GB if before_launch else 0.0)
        if need > limit:                                  # last resort: drop the best candidate's weights too
            prune(protect_best=False)
            used = self.disk_gb()
            self.store.update_state(failures=[{"kind": "disk", "detail": f"over budget before launch: {need:.1f} GB > {limit} GB; "
                                           f"best candidate's weights deleted, now {used:.1f} GB"}])
        self.store.update_state(updates={"disk_gb": round(used, 2)})
        if freed > 0.05:
            self.event("disk_pruned", freed_gb=round(freed, 2), used_gb=round(used, 2), limit_gb=limit)
