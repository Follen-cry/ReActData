# reactdata-v6: multi-task data selection with the modular pipeline

Copy of `../pipeline` (reactdata-v5, modular refactor `a1ce1f3`) adapted to a multi-task setting. The architecture
(`experiment/` workflow / improvement / evidence / runtime, `tasks/reactdata/` adapter) is unchanged; see
`../pipeline/MODULARITY.md`. This file lists only what differs from v5.

**Question.** With a fixed budget of 128 datapoints seen once (8 optimizer steps x 16 rows), does a Builder Agent's
*data selection* (task mix, which clips within a task, their order, loss feedback) beat a task-balanced random subset
on three VBVR tasks?

## Setting

| item | v6 |
|---|---|
| tasks | `shape_color_then_scale`, `shape_color_then_move` (analogy-grid scorer), `2d_geometric_transformation` (VBVR-CustomEval scorer, as in v0) |
| train pool | 480 clips, 160 / task (scale: v4 pool, move: fresh raw draw chunks 0-19, geo: v0 pool) |
| inner probe | 32 (11 / 11 / 10), never trained |
| dev | 150 = 50 / task (v1 dev scale + v1 dev move + 50 of v0 dev geo) |
| test | 150 = 50 / task (50 fresh scale clips, 50 of v1 test move, 50 of v0 test geo) |
| row | F0 + evaluation text -> final frame; no context frames, no intermediate targets, fixed text |
| run | 128 rows = 8 steps x 16, each clip at most once (`api/recipe.py` `check_unique`, enforced in checks and in training) |
| schedule | LoRA lr 2e-5, gen-decoder lr 1e-4, 3-step warmup then cosine (calibrated, `tools/calibrate.py`) |
| baseline c000 | 43 / 43 / 42 random clips per task (short task rotates with the seed), shuffled |
| search | 6 serial agent steps (Sonnet, resume chain), top 2 confirmed on seeds 1001-1004, P3 formal seeds 2000-2002 on test |
| GPUs | 2 x 4 A40 slots (torrnode12), fallbacks torrnode14 4-7 / torrnode11 0-3 |

Score = equal-weight mean of the three task means. Scorer validation (`manifests/scorer_validation.json`): GT 1.0 (geo
0.997), copy of input 0 (geo 0.12). Isolation (`manifests/isolation_report.json`): no md5 overlap across splits, no
duplicate generator parameters.

## Calibration (before the protocol was frozen)

c000, seed 1000, dev, one run each:

| lr (x default 1e-5 / 5e-5) | warmup | dev | scale | move | geo |
|---|---|---|---|---|---|
| Base (no training) | - | 0.037 | 0.002 | 0.008 | 0.101 |
| 1x | 6 (v5) | 0.100 | 0.084 | 0.102 | 0.114 |
| 1x | 2 | 0.130 | 0.107 | 0.155 | 0.129 |
| **2x** | **3** | **0.149** | 0.158 | 0.170 | 0.117 |
| 5x | 2 | 0.026 | 0 | 0 | 0.077 |
| 10x | 2 | 0.023 | 0 | 0 | 0.069 |

5x and above collapse (E / F never drawn on any scale / move item).

## Files changed relative to `../pipeline`

- `protocol.yaml` (v6; `protocol_v6.yaml` is a symlink), `BUILDER_AGENT_v6.md`, `seeds/fixed_f0_final/*`.
- `runner/build_data.py` (v6 data), `runner/common.py` (v6 scratch default, tasks, env overrides).
- `api/recipe.py` (final-only, `check_unique`), `api/plugin_api.py` (pool format, probe limits for 8-step runs),
  `api/scorer_description.md` (three tasks).
- `runner/score.py` (per-task scorer dispatch, task-specific sub-scores), `runner/checks.py` (8-step simulation,
  resume at 4, one-epoch rule, fixed template), `runner/train_entry.py` (one-epoch rule, `warmup_steps` from the spec),
  `runner/audit.py` (single-template early return before any scale metadata), `runner/report.py` (per-step traces,
  `pool_view.json` in /input, F0 / final examples), `runner/final_report.py` (per-task columns),
  `tasks/reactdata/adapter.py` (protocol `training.overrides` / `warmup_steps` into job specs),
  `tasks/reactdata/contract.py` (pool summary), `tasks/reactdata/gpu_probes.py` (bug fix, also in `../pipeline`).
- `tools/calibrate.py`, `tests/fixtures/v6_probe_cand/`, `tests/unit/test_undefined_names.py`.

## Commands

```bash
python runner/build_data.py                         # once (done)
python -m pytest tests/unit -q                      # 50 tests
python -u runner/state_machine.py --run-id smoke6b --smoke --protocol protocol.yaml
nohup setsid python -u runner/state_machine.py --run-id r1 --protocol protocol.yaml \
    >> /scratch/network/ssd/junlin/reactdata-v6/runs/r1.log 2>&1 &
```

Data / runs: `/scratch/network/ssd/junlin/reactdata-v6/{data,runs,calibration}`.

## r1 outcome (2026-10-28, 2.98 h, report: runs/r1/report.md, https://claude.ai/artifact/RLCY8homw6bMfud5TqQMVS)

- P1 baseline replicates dev .149 / .179 / .134 / .068 / .102 (mean .127, sd_train .043); Base .037. Low runs are partial
  collapses (A-D cells damaged on up to 73 / 100 scale+move items); geo rotation never learns in 8 steps.
- Search (6 Sonnet steps, $3.68): mix shift c001 -.002; large-edit selection (top-50% change_px) c002 +.055, c006 +.051,
  c003 +.051, c004 +.038, c005 +.008 (screening, one run each).
- Confirmation (seeds 1001-1004): c002 D -.038 [-.107, +.032], c006 D -.020 [-.090, +.049] -> c000 selected.
- Test: baseline .101 / .115 / .109 vs Base .037; D +.071 [+.052, +.093] (scale +.073, move +.121, geo +.020).
