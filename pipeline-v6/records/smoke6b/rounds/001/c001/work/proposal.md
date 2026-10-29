# c001

Started from c000 (fixed baseline; only candidate so far).

## Evidence
Read report.md, results.json, baseline_replicates.json, the baseline code and plugin API. Only c000 exists, so no analysis traces compare programs. c000 dev: scale 0.270 (E_cell 0.744, F_cell 0.0, F_transform 0.038), move 0.092 (E_cell 0.0, F_cell 0.0, F_transform 0.367), geo 0.111 (rotation_angle 0.0, position_alignment 0.011, shape_fidelity 0.75). Item level: dev_scale_000 baseline mean 0.31 (sd 0.057), dev_move_000 0.046 (sd 0.065), dev_geo_000 0.056 (sd 0.079). Geo rotation sub-score is exactly 0 and the geo score comes almost entirely from shape fidelity, which the untrained model already has; this suggests geo rotation is not learnable in a few rows. Move's E_cell (recolouring D as A became B) is 0, whereas scale's E_cell is already 0.74, so the recolour rule is learnable and shared by scale and move. Baseline training loss per step is flat (0.012, 0.011, 0.022, 0.010). Note the report's numbers come from a 4-step, 2-seed setup, so noise estimates are rough.

## Hypothesis
Rows spent on geo are largely wasted (rotation 0.0, little headroom within 128 rows), while scale and move share the colour-then-transform structure and have the headroom (move E_cell 0.0, scale F_cell 0.0). Moving rows from geo to scale and move should raise those two task means more than it can cost geo, which has little to lose (shape_fidelity is already 0.75).

## Program
databuilder.py: quota 56 scale / 56 move / 16 geo, clips drawn uniformly without replacement within each task (seeded, so it is valid for any seed), then one seeded shuffle so every step mixes tasks. Same as c000 except the quota. declared_class static.

## What analysis.py measures
Per-step per-task training loss and row counts, grad norm, and a loss probe on all 32 inner-probe clips after every step (8 probes), reported as the per-task mean. A generate probe on all 32 inner-probe clips at step 4 or later gives per-task scores. Next step: if probe loss on scale and move falls more than in a baseline run and the geo probe loss does not rise much, the mixture shift works. If the generate-probe score on geo drops, the shift is hurting geo.

## Expected result
Scale +0.03 to +0.08, move +0.03 to +0.06, geo -0.03 to 0. Dev mean about +0.02 to +0.04. It is refuted if the dev mean is within noise (about 0.06) or if geo falls by more than the gains elsewhere.
