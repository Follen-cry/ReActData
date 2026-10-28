"""Plugin contract (fixed): what a candidate's databuilder.py and analysis.py must implement.

Both run inside a sandboxed subprocess (no network, no file system beyond a scratch dir; only the Python standard
library modules in ALLOWED_IMPORTS and numpy). They exchange JSON-serializable values with the Runner only.

databuilder.py must define

    class Builder:
        def __init__(self, config: dict, pool: dict, seed: int, total_steps: int): ...
        def next_batch(self, step: int) -> list:        # ROWS_PER_STEP recipes (api/recipe.py)
        def observe(self, step: int, summary: dict) -> None   # summary = Analysis.update(...) of this step
        def state_dict(self) -> dict                    # JSON; everything needed to resume exactly
        def load_state_dict(self, state: dict) -> None

analysis.py must define

    class Analysis:
        def __init__(self, config: dict, seed: int, total_steps: int, inner_probe_ids: list): ...
        def update(self, step: int, rows: list, step_info: dict, probe_results: list) -> dict
        def request_probes(self, step: int, budget: dict) -> list
        def state_dict(self) -> dict
        def load_state_dict(self, state: dict) -> None

Call order of one optimizer step k (k = 0 .. total_steps-1):
    recipes = builder.next_batch(k)            -> Runner checks, renders text, trains on the 16 rows
    summary = analysis.update(k, rows, step_info, probe_results)   # probe_results: results of probes requested at k-1
    builder.observe(k, summary)
    requests = analysis.request_probes(k, budget)                  # executed right after step k

``pool`` (Builder):     {"task", "n_frames", "fps", "stat_resolution", "clips": [{"clip_id", "frame_change"[59], "fg_area"[60]},
                         ...]} -- pixel statistics of the training videos (shape_color_then_scale only)
                         (manifests/frame_stats_spec.json). No generator metadata.
``config``:             the candidate's candidate.yaml "builder" / "analysis" sections (free-form JSON) plus
                        "templates": the list of template ids.
``rows`` (Analysis):    one dict per training row of the step: {"j", "clip_id", "context_frames", "template_id", "target_frame",
                        "loss" (unweighted flow-matching loss), "sigma", "seq_len", "vit_images", "cfg_drop"}.
``step_info``:          {"step", "lr", "grad_norm", "gen_weight", "sec_per_step", "total_steps"}.
``budget``:             {"loss_probe_next_allowed_step", "generate_probe_next_allowed_step", "generate_probes_left",
                         "loss_probes_left"}.

Probe requests (at most one of each kind per call; the Runner drops requests that break the frozen limits
and reports them back as {"kind", "status": "rejected", "reason"}):
    {"kind": "loss", "items": [{"clip_id": <inner probe id>, "context_frames": [...], "template_id": str,
                                "target_frame": int (optional, default final)}, ...]}
        <= PROBE_MAX_ITEMS items; no gradient; fixed noise and sigma per (clip_id) so values are comparable over
        steps. Result: {"kind": "loss", "step", "items": [{"clip_id", "context_frames", "template_id", "loss", "sigma"}]}
    {"kind": "generate", "clip_ids": [<inner probe ids>]}
        F0 + the original (evaluation-format) description -> final frame, generation seed 0, scored by the
        Runner with the task's rule scorer. Result: {"kind": "generate", "step", "items": [{"clip_id", "score",
        "sub": {"rotation_angle", "position_alignment", "shape_fidelity"}, "valid"}]}
Inner-probe clips are reserved from the training side and never trained on. dev and test are never reachable.

Losses of rows with different target frames are not comparable (a target close to F0 is easier): normalize per
target frame (and noise level) before ranking clips.

Randomness: use only random.Random / numpy Generators seeded from ``seed`` (and step). No wall-clock, no global RNG
state, no hash() of str (PYTHONHASHSEED is fixed, but do not rely on it). Equal inputs must give equal outputs.
"""
from __future__ import annotations

ALLOWED_IMPORTS = ("math", "random", "json", "collections", "itertools", "functools", "statistics", "heapq",
                   "bisect", "copy", "typing", "dataclasses", "numpy", "plugin_api", "recipe")
PROBE_MAX_ITEMS = 32
LOSS_PROBE_MIN_GAP = 10          # steps between two loss probes (budget cap)
GENERATE_PROBE_MIN_GAP = 50      # steps between two generate probes (budget cap)
MAX_LOSS_PROBES = 20
MAX_GENERATE_PROBES = 4
STEP_TIME_LIMIT_S = 10.0         # wall time of one plugin call (next_batch / update / observe / request_probes)
