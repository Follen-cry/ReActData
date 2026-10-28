# c008 — c006-style process rows, half of them conditioned on an earlier F-present frame

Started from c007's /work copy (same 8 plain + 8 process-row structure); added ViT-only context on 4 of the 8 process rows.

## Evidence
Baseline 0.391. Process-row dose/coverage series: c004 (6 rows, frames 50-57) 0.3913; c005 (6, 40-47) 0.4005; c006 (8, 36-48) 0.4147 (D +0.028 CI [+0.001, +0.057]); c007 (8, 34-56) 0.4043 (D +0.018 CI [-0.007, +0.044]; move 0.4152, scale 0.3933). So F-present process targets give a stable but small gain of about +0.02, mostly in scale E_cell (0.641-0.643 vs 0.522 baseline) and fewer E_wrong_colour on scale (10-11 vs 20), while scale F_cell (0.113 / 0.151 vs 0.116) and F_transform (0.495 / 0.501 vs 0.577) did not improve; move changes: E_cell 0.641, F_cell 0.362, F_transform 0.185 vs 0.639 / 0.349 / 0.181 — flat. Widening to settled frames in c007 did not restore F_transform. The F-geometry remains the failure: items dev_move_000 (0.4195 baseline mean, sd 0.094), dev_move_003 (0.409, sd 0.13) and dev_move_001 (0.295, sd 0.18) fail through F_wrong / F_wrong_transform (33 of 50 move items in c007), while dev_move_007 (0.971) and dev_move_010 (0.981) are solved. Single-frame targets teach what a state looks like but not the *transition* of F; the F-absent context hurt earlier (c002 0.297), the F-present case is untested.

## Hypothesis
Conditioning a later process frame on an earlier F-present frame of the same clip (frame 34-42 as context, target >= 4 frames later, up to 56) trains the model to extrapolate F's own trajectory (size change applied to an existing F), the part of the rule the single-frame targets did not improve (F_transform). Final-frame rows stay context-free, so the test-time format is untouched, and the other 4 process rows keep the c006 content. The context is given only where F is present, avoiding the F-absent confusion of c002.

## Program
databuilder.py: per step 8 plain eval_format final rows; 4 `late` rows (target uniform 36..52, no context); 4 `late_ctx` rows (context frame uniform 34..42, target uniform in [ctx+4, 56], template with {context_list} and {target_spec}, no {wrapped_description}). Independent seeded clip streams, static for all 100 steps, max 20 input images per step. Declared static.

## What analysis.py measures
Windowed loss per template (eval_format / late / late_ctx — if late_ctx loss is far below late loss the context is being copied); loss probe every 20 steps on 8 held-out clips for final, frame-45 plain, and frame-50-with-ctx-38; generate probes at 49 and 99 (plain F0 -> final) with mean of every sub-score returned. Compare the step-99 probe to c006/c007.

## Expected result
Scale F_transform >= 0.55 and dev 0.41-0.45 (primarily via scale; move +0.02 to +0.04 if the trajectory knowledge transfers). Refuted if dev <= 0.40 (no gain over c006/c007) or if F_missing/blank outputs rise (context copying), in which case the remaining budget should go to c006-like programs.
