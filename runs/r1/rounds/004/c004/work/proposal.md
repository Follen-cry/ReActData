# c004 — near-final (settled-state) target augmentation, 6 of 16 rows all run

Started from c003's /work copy; kept analysis.py structure, replaced builder/templates.

## Evidence
Baseline dev 0.391 (sd 0.025). Three process-based programs all scored below it: c001 sub-goal frame 26 on 8/16 rows to step 70: 0.343; c003 the same for only steps 0-39 then 60 plain steps: 0.350 (move 0.334, scale 0.366; D -0.037 CI [-0.065, -0.010]); c002 frame-27 context: 0.297. c001 and c003 are nearly identical despite a 40 vs 70-step sub-goal phase, so the damage does not scale with how long the sub-goal rows last: it is either done early or comes from the *content* of the sub-goal target (F absent; F_missing 14 and 10 on move vs 8 baseline) or from the split of rows. c003's generate probe at step 39 (0.157) shows the model produces F-less / wrong answers after the warm-up. Items: dev_move_000 (baseline mean 0.4195, 0.519 in c000, 0.0 in c001), dev_move_001 (baseline mean 0.295, sd 0.179 -> 0.0 in c001), dev_move_003 (0.644 in c000 -> 0.35 in c001; E_cell 1.0, F_cell 0.735 -> 0.0).

## Hypothesis
To separate "intermediate-target content" from "diverted rows / extra template", use process frames whose content is identical to the answer: frames 50-57 are the settled final state (E and F both at their final size), so training to generate them is equivalent to the test target under small pixel variation, with a different template ("Generate {target_spec}"). If the damage was caused by F-absent targets, this should be ~baseline or slightly better (augmentation against memorising 16 exact images, 6 extra views per step); if it is also lower, any change of the row type is costly at this data/step scale, and later candidates should stay in the baseline format.

## Program
databuilder.py: every step, 10 rows plain eval_format final-frame, 6 rows template `late` ({description} + "Generate {target_spec}.") with target_frame random in 50..57 (seeded by step), independent clip streams, no context frames. Static over the 100 steps (declared static). Always >=10 final rows.

## What analysis.py measures
Windowed loss per template (plain vs late); loss probe every 20 steps for final vs frame-54 targets on 8 held-out clips; generate probes at steps 49 and 99 (mean of every returned sub-score) to compare with c001/c003 (0.16 at the end of their sub-goal phases; c001 0.358 at 99).

## Expected result
Dev 0.37-0.41 (about baseline; no more than +0.03). Refuted (informatively) if < 0.35: row-type changes alone harm. A value >= 0.43 would be surprising and would mean settled-state views regularise the final-frame mapping.
