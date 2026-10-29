## Evidence
Read report.md, c001 analysis trace (per-task loss, 4 loss probes, generate probe) and dev summaries. c001 (quota 56/56/16) scored 0.1245 vs baseline 0.1493: D=-0.002 vs the replicate-mean comparison in results (CI [-0.026,+0.021]); geo -0.015 (0.092; position_alignment 0.005, rotation 0.016), move -0.005 (0.138), scale +0.014 (0.143). So moving 27 rows from geo to analogy tasks did not help: the task mix lever is weak or noise-dominated. Failure categories rose for analogy tasks in c001: extra_objects 37 (scale) and 32 (move) vs 17 / 19 in c000, blank_or_repainted 8 each, context_damaged 14 on scale vs 3; i.e. the model sometimes repaints or adds objects instead of making a clean edit. Probe losses: geo probe loss rose from 0.0075 (step 0) to 0.0138 (step 6) while move fell 0.0088->0.0062 and scale 0.0052->0.0045 - geo is not being learned; analogy tasks learn slowly. Generate probe after step 7: scores are bimodal (e.g. move_s10_00000357 score 1.0, scale_s20_00000221 0.78, but many 0.0 such as scale_s20_00000066 and move_s00_00000001), so success is all-or-nothing per instance. Dev items quoted from c000: dev_geo_000 0.109 (rotation 0.0, position 0.006), dev_geo_001 0.114, dev_geo_002 0.041.

## Hypothesis
Clips with a large edit (large change_px: bigger added/moved objects, bigger rotation displacement) give a stronger, less ambiguous gradient toward "the final frame differs from F0 in a specific way", and discourage the near-copy / repaint failures; small-edit clips are close to F0 and teach mostly identity. With balanced quota (the mix lever showed no gain) selecting the top 50% by change_px in each task should yield better learning per row.

## Program
Started from c001 code (the databuilder generalised with `top_frac`), but quota reset to the baseline-like 43/43/42. For each task, the clips are ranked by change_px and 43/43/42 are drawn uniformly at random from the top 50% (80 clips) of that task; mixed shuffled order. Rule-based, seed-generalising. analysis.py unchanged.

## What analysis.py measures
Per-task training loss per step, loss probes on all inner-probe clips after steps 0,2,4,6 and a generate probe after step 7 (as c001). Read: whether probe losses decrease more than in c001 per task, and whether the generate probe has fewer 0.0 failures; compare extra_objects counts with c000 (17/19) and c001 (37/32).

## Expected result
Scale and move +0.02 to +0.05 each, geo around 0 to +0.02, dev mean +0.02 to +0.03 over the baseline; fewer extra_objects/blank failures. Refuted if the dev mean is at or below baseline and failure categories do not change, which would say edit size does not matter either at this budget.
