# c003 — short sub-goal warm-up (steps 0-39), then 60 steps of pure baseline rows

Started from c002's /work copy but reverted the context idea; builds on c001's sub-goal rows with a shorter schedule.

## Evidence
Baseline 0.391 (sd 0.025). c001 (sub-goal target frame 26 on 8/16 rows to step 70): 0.343 (move 0.336, scale 0.351; F_missing 14 / 11). c002 (frame 27 as ViT context on 8/16 rows to step 60, all targets final): 0.297, D -0.089 CI [-0.119, -0.060]; move 0.307, scale 0.288; E_cell fell on both (move 0.639 -> 0.466, scale 0.522 -> 0.379). In c002's trace the held-out loss with context equals the loss without (0.0038 vs 0.0038 at step 59), so the model ignores the context, yet dev dropped: the harm is not "copying the context" but the diversion of rows from the exact test format (plain rows were only 8/16 for 60 steps) and/or extra-image/text interference. Items: dev_move_000 baseline mean 0.4195 -> 0.0 in c001 (E_wrong_colour); dev_move_001 baseline mean 0.295; dev_move_003 0.644 (baseline c000) -> 0.35 in c001 (F_cell 0.735 -> 0). Both process uses so far lowered E_cell on both tasks, so they disrupt the recolour step rather than help it. The generate probe in c001 at step 49 was 0.163 vs 0.358 at step 99, i.e. the model recovers once the plain rows dominate.

## Hypothesis
Sub-goal rows probably help or hurt through *how long they displace the test format*. A short warm-up (40 steps) followed by 60 steps of 16 plain rows lets the model take the E-state sub-goal as an initial scaffold, then consolidate on the exact test mapping. If c001's loss came from late interference (F_missing: model stops at the E-state), this should come back to baseline or above; if the warm-up still hurts, sub-goal targets at this data size are simply harmful and later candidates should stop using intermediate targets.

## Program
databuilder.py: steps 0-39: 8 rows with template eval_format target 59 and 8 rows with template subgoal ("{description} Generate {target_spec}.") at target_frame 26, no context; steps 40-99: 16 plain eval_format final rows. Independent seeded permutation streams for the two kinds. Declared class: schedule.

## What analysis.py measures
Windowed loss per template; loss probe every 20 steps on 8 held-out clips for final vs sub-goal targets; generate probes at step 39 (end of warm-up, compare with c001's 0.163 at 49 and baseline-like level) and step 99 (compare with c001's 0.358 and c002's), logging every sub-score key returned.

## Expected result
Dev between c001 and baseline: about 0.37-0.41 (i.e. no significant gain). A result >= 0.43 would say early process supervision helps; <= 0.35 would say intermediate targets are harmful even early (refuting, and suggesting the next candidate should not use them).
