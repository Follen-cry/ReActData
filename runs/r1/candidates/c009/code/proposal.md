# c009 — c006 process rows, removed for the last 25 steps

Started from c008's /work copy; dropped the context rows (they helped scale but not move) and returned to the c006 mix with a schedule.

## Evidence
Baseline 0.391 (move 0.408, scale 0.373). The final test is move only. Process-row series: c006 (8 rows, frames 36-48, no ctx, all steps) 0.4147, move 0.4187, scale 0.4108; c007 (34-56) 0.4043, move 0.4152; c008 (c006-like + 4 context-conditioned rows) 0.4099 but move 0.3776 (-0.017 vs baseline CI [-0.041, +0.008]; move F_cell 0.284 vs 0.349 baseline and 0.361 in c006) and scale 0.4423 (+0.064 CI [+0.017, +0.111]; E_cell 0.704, F_transform 0.532). So the context rows specialise to the training task's scale trajectory and hurt F placement on the cross-task move set, whereas context-free F-present targets (c006/c007) gave move +0.02 (move F_cell 0.361/0.362). Items: dev_move_003 (baseline 0.409, sd 0.13), dev_move_000 (0.4195, sd 0.094), dev_move_001 (0.295, sd 0.18) are the high-variance items driving move; F_wrong_transform stays at 33 of 50 in every program (c006, c007, c008), i.e. the F-move itself is never learned from scale-only videos. c008's loss probes show late_ctx loss ~ equal to late loss, so context is not simply copied.

## Hypothesis
Because the test is the final-frame mapping on move, the end of training should be pure test-time rows. In c006 the last 25 steps still had 8 of 16 rows on process frames. Dropping the process rows after step 75 keeps the early/middle regularising effect (E-colour gain, less memorisation of the 16 exact final images) while letting the last updates fit the exact final mapping; this is the "schedule that removes them before the end" lever from the task description. Expected to retain the E_cell gain and perhaps lift F_cell/F_transform on move slightly.

## Program
databuilder.py: steps 0-74: 8 plain eval_format final rows + 8 `late` rows (target uniform 36..48, no context); steps 75-99: 16 plain final rows. Seeded independent clip streams. Declared schedule.

## What analysis.py measures
Windowed loss per template; loss probes every 20 steps (final vs frame-42 plain vs ctx probe for continuity); generate probes at 49 and 99 with mean of each returned sub-score; compare step-99 generation with c006 and the dev move/scale sub-scores.

## Expected result
Dev 0.40-0.43; move 0.41-0.44 (c006 0.419), scale ~0.40. Refuted if dev < 0.39 (the late process rows were part of the benefit) — in that case the final choice should prefer c006's constant mix.
