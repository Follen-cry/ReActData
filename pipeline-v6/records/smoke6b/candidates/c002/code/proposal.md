# c002

Started from c001 (best valid by dev mean, 0.1538 vs baseline 0.1576 on one run). Analysis.py unchanged (per-task loss, 32-clip loss probe every step, generate probe at step >= 4).

## Evidence
Read report.md, results.json, dev_items/c001.json, analysis_traces/c001.json. The dev feedback is only three items (one per task), so the per-task numbers are very noisy. c001 vs baseline replicate mean: dev_scale_000 0.35 vs 0.31 (E_cell 1.0, F_missing, F_cell 0, F_transform 0); dev_move_000 0.0 vs 0.046 (E_wrong_colour, F_wrong, all sub-scores 0; baseline had F_transform 0.367); dev_geo_000 0.1114 vs 0.056 (rotation_angle 0.0, shape_fidelity 0.75 unchanged). Scale's E_cell is solved (1.0), move's is not (0.0) although c001 gave move 56/128 of the planned quota. The c001 probe losses (mean over inner-probe clips, per task) rose from step 0 to step 2 (move 0.0088 -> 0.0163, scale 0.0052 -> 0.0137, geo 0.0075 -> 0.0161) and fell at step 3 (0.0083 / 0.0068 / 0.0073 ... per-clip values e.g. move_s12_00000317 0.0200 -> 0.0373 -> 0.0266). So early steps perturb the model and later steps recover; the order of the last steps matters. Also c001 had a bug: its fixed quota summed to 128 but the run has 64 rows (4 steps x 16), so it drew a truncated shuffle (rows per task at step 0: scale 10, move 5, geo 1).

## Hypothesis
The weak point is move (E colour wrong, F not moved). Recolouring is learnable from a handful of rows (scale E_cell went 0.744 -> 1.0), and the loss is lowest right after the recovery in the final step. Giving move 5/8 of the rows and placing the move block last, so the final updates before evaluation are all on move, should give move its best chance of learning the recolour + move rule, at the cost of geo (1/8) and scale (1/4), which are already at plateau (geo rotation 0.0 under every program so far; scale E_cell 1.0).

## Program
databuilder.py: row counts are fractions of the actual number of rows: scale 1/4, geo 1/8, move 5/8 (64 rows: 16/8/40). Clips are drawn uniformly at random without replacement within each task (seeded, valid for any seed). Order is in task blocks: scale, then geo, then move (no shuffle), so the last steps are all move. Started from c001; the change is the quota, the scaling to n and the blocked order.

## What analysis.py measures
Per-step per-task training loss and row counts, grad norm, a loss probe on all 32 inner-probe clips after each step, and a generate probe at step >= 4 for per-task scores. Read next step: does move probe loss decrease from its step-0 value by the last step, and do the scale and geo probe losses rise after the move block (forgetting)?

## Expected result
Move +0.05 to +0.12 (from 0.09 baseline / 0.0 in c001; E_cell above 0), scale -0.05 to 0 (fewer rows, forgetting risk), geo about 0 (+-0.05). Dev mean about +0.01 to +0.04 over baseline. Refuted if move stays at 0 with 40 move rows, which would show that the move rule is not learnable with this budget and rows should go elsewhere.
