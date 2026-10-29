## Evidence
c002 (43/43/42, random from top 50% by change_px per task) scored 0.1818 vs baseline 0.1493: D=+0.055, CI [+0.032,+0.078]; per task scale 0.246 (+0.117, CI [+0.07,+0.17], F_transform 0.56 vs 0.36, F_cell 0.097 vs 0.054), move 0.172 (+0.029, not significant), geo 0.127 (+0.020; position_alignment 0.147 vs 0.050 and rotation 0.064 vs 0.036, but shape_fidelity fell to 0.297 from 0.563 and deformed_shape rose to 33 from 15). Item examples: dev_geo_000 (c002) position_alignment 0.297 (c000: 0.006) but shape_fidelity 0.056 (c000: 0.75), score 0.114 vs baseline mean 0.146 - larger rotation displacement gets the model moving the polygon but deforming it. dev_move_007 (c002) scored 0.241 (baseline mean 0.283, sd 0.41) with F_transform 0.92 but E_cell 0.0: the move is right while the recolour of E is wrong. dev_move_006 scored 0.0 with E_wrong_colour, F_missing and extra_objects (baseline 0.0 too), a persistent failure that selection by edit size did not fix. c001 (task mix only) gave nothing, so the edit-size rule is the working lever. Residual failures in c002: extra_objects 36 (scale), 23 (move). Single run, noise sd 0.043, so scale's +0.117 is the credible part.

## Hypothesis
Dose-response: the gain on the analogy tasks comes from large-edit clips. Sharpening the selection on scale and move from top 50% (80 clips) to the top 27% (exactly the 43 largest change_px clips, deterministic) should raise them further. Geo is left at the c002 rule (top 50%) as it is ambiguous (position gain, shape loss), so any difference vs c002 is attributable to scale/move.

## Program
Started from c002. `top_frac` is now per task: scale 0.27 and move 0.27 (candidate pool = max(quota, 43) = the 43 clips with largest change_px), geo 0.5. Quota 43/43/42, mixed shuffled order. The scale/move subset is therefore the same for every seed (rule-based, order only varies). analysis.py unchanged.

## What analysis.py measures
Per-task training loss, loss probes after steps 0,2,4,6 and generate probe after step 7 (per-task score). Read: whether scale/move probe loss and generate score exceed c002's; compare extra_objects (36/23).

## Expected result
Scale 0.25 -> 0.28+, move 0.17 -> 0.20, geo about unchanged (0.12 +/- 0.03); dev mean about 0.19-0.21. Refuted if scale/move fall back to baseline levels (<0.19 / <0.16), meaning the extreme top clips lack diversity and the gain of c002 was moderate edit size, not maximal.
