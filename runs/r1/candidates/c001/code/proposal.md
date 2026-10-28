# c001 — E-state sub-goal rows, removed before the end

Started from c000 (fixed baseline; the only program so far).

## Evidence
Only the baseline exists (dev 0.391; move 0.408, scale 0.373; replicate sd 0.025). Sub-scores: move E_cell 0.64, F_cell 0.35, F_transform 0.18; scale E_cell 0.52, F_cell 0.12, F_transform 0.58. Error categories are dominated by E_wrong_colour (16 move / 20 scale) and F_wrong/F_wrong_transform (47 move / 54 scale). Items: dev_move_000 (0.519, E_cell 1.0 but F_transform 0.005), dev_move_003 (0.644, E_cell 1.0, F_transform 0.0), dev_move_001 (0.125, E_cell 0, E_wrong_colour + context_damaged, baseline mean 0.295 sd 0.18), dev_move_002 (0.029, E wrong colour). So even when E is right, F is derived wrongly, and E colour is often wrong: the recolour step is unreliable. Training loss is the final-frame loss only, with no explicit signal for the E stage.

## Hypothesis
Rows whose target is the frame where E already has its new colour and F does not exist yet (frame ~26) supervise the first analogy rule (A->B applied to D) in isolation, task-independently. This should raise E_cell on both tasks (it is shared by move). Dropping these rows after step 70 lets the last 30 steps be pure test-time mapping so the model ends on the test input distribution. Expected gain is mainly in E_cell, and in F_cell as a knock-on, since F depends on E.

## Program
databuilder.py: steps 0-69: 8 final-frame rows (template eval_format) + 8 rows with target_frame 26 (template subgoal: description + "Generate {target_spec}."), no context frames, clips by independent seeded permutations; steps 70-99: 16 final rows as in the baseline. text_templates.yaml adds `subgoal`. declared class: schedule.

## What analysis.py measures
Windowed mean loss per target frame (normalised comparison is only within a target); loss probe every 20 steps on 8 held-out inner clips for final and frame-26 targets (does sub-goal loss fall faster than final loss?); generate probes at steps 49 and 99 on the same 8 clips (rule-score of F0 -> final generation, mid-run vs. end, to see whether removing sub-goals hurts or helps).

## Expected result
Dev +0.03..+0.08, mostly via move/scale E_cell (+0.1) and some F_cell. If dev stays within noise and E_cell does not move, intermediate targets do not transfer to the final-frame mapping at this data scale (a clear negative). If E_cell drops, the extra text/target format interferes.
