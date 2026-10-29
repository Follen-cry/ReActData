"""Exclusive resource leases, independent of job execution and retry state."""
from contextlib import contextmanager
import threading
import time


class SlotPool:
    def __init__(self, slots, concurrency, lock, idle):
        self.slots, self.lock, self.idle = slots, lock, idle
        self.capacity = threading.Semaphore(concurrency)

    @contextmanager
    def lease(self, attachment=None):
        self.capacity.acquire()
        slot = None
        try:
            if attachment:
                desired = next((s for s in self.slots if s.to_json() == attachment), None)
                if desired is None:
                    raise RuntimeError("persisted job slot no longer exists")
                with self.lock:
                    if desired.job is not None:
                        raise RuntimeError("persisted job slot is already reserved")
                    slot = desired
                    slot.job = {"attached": True}
            while slot is None:
                for candidate in self.slots:
                    with self.lock:
                        if candidate.job is not None:
                            continue
                        candidate.job = {"reserved": True}
                    try:
                        available = self.idle(candidate.node, candidate.gpus)
                    except BaseException:
                        with self.lock:
                            candidate.job = None
                        raise
                    if available:
                        slot = candidate
                        break
                    with self.lock:
                        candidate.job = None
                if slot is None:
                    time.sleep(60)
            yield slot
        finally:
            if slot is not None:
                with self.lock:
                    slot.job = None
            self.capacity.release()
