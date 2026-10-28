# c002 — E-state frame as ViT-only context for final-frame rows, removed after step 60

Started from c001 (copied files), but replaced the idea: c001 used the E-state as a *target* and hurt; here it is *context* and every target is the final frame.

## Evidence
c001 (8 of 16 rows target frame 26 for 70 steps) scored 0.343 vs baseline 0.391 (D -0.043, CI [-0.072, -0.016]); move -0.058, scale -0.028. Sub-scores: move E_cell 0.639 -> 0.546, F_cell 0.349 -> 0.265; scale F_transform 0.577 -> 0.446. F_missing rose (move 8 -> 14, scale 1 -> 11): the model sometimes stops at the E-state, as the sub-goal target looks like an F-less answer. Items: dev_move_000 fell 0.519 -> 0.0 (E_wrong_colour, baseline mean 0.4195), dev_move_001 0.125 -> 0.0 (baseline mean 0.295, sd 0.18), dev_move_003 0.644 -> 0.35 (F_cell 0.735 -> 0.0). Loss probes show sub-goal loss (~0.001) is far below final loss (~0.004-0.006) and barely changes: the sub-goal is an easy copy task teaching little. Generate probe on 8 held-out clips: 0.163 at step 49 (mid sub-goal phase), 0.358 at step 99 (several zero scores at step 49). So intermediate *targets* dilute and confuse the final-frame mapping.

## Hypothesis
The process (E first, F derived from E) may still help if shown as *input* rather than as a target: F0 + the E-state frame (frame 27) as ViT context -> final frame. The row keeps the correct target and the final-frame output distribution, but the ViT sees a correct E, so the gradient on the F-derivation (E -> F via the B->C rule) is cleaner and attributes F to E. The context is dropped after step 60, so the last 40 steps train plain test-time rows (and half the earlier rows are already plain), so the model cannot depend on the context at test time. Risk: the model learns to copy the context's E and ignore F0, which would show as a gap between ctx_final and eval_format loss.

## Program
databuilder.py: steps 0-59: 8 rows with context_frames [27], template ctx_final (wrapped_description + context_list), 8 plain rows (eval_format); steps 60-99: 16 plain rows; every row targets frame 59 (no non-final targets, no {target_spec}). Clips come from one seeded permutation stream. Declared class: schedule.

## What analysis.py measures
Windowed training loss per template (ctx_final vs eval_format: if ctx loss is much lower, the model leans on the context); loss probe every 20 steps on 8 held-out clips for both templates (final target); generate probes at 49 and 99 (plain F0 -> final) logging the mean of every sub-score key the probe returns (probe keys differ from the dev E_cell/F_cell/F_transform), comparing with c001's 0.163 (step 49) / 0.358 (step 99).

## Expected result
Small effect, roughly 0 to +0.04 dev (F_cell / F_transform on both tasks), more likely neutral than c001's drop since all targets are final and half the context rows are removed early. If dev stays <= baseline-0.05 the context also interferes (the ViT image/text difference at train vs test); if it improves F_transform on move, the process transfers across tasks.
