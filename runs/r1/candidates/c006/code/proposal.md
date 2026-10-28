# c006 — dose increase of the mid-rescale target rows (8 of 16, frames 36-48)

Started from c005's /work copy; same mechanism, larger dose and a wider frame range.

## Evidence
Baseline 0.391. c005 (6/16 rows target frames 40-47, rest plain final): 0.4005, D +0.014 CI [-0.014, +0.042] — within noise overall, but the direction of the sub-scores is consistent with the mechanism: scale F_cell 0.187 vs 0.116 baseline (and 0.153 for c004), F_missing 1 on scale (baseline 1), move E_cell 0.601 / F_cell 0.345 / F_transform 0.214 (baseline 0.639 / 0.349 / 0.181), move F_transform up. c004 (settled frames 50-57): 0.3913. F-absent targets/context (c001 0.343, c003 0.350, c002 0.297) were clearly worse (F_missing up). The generate probe at step 49 was 0.396 in c005 vs 0.355 in c004 and 0.163 in c001/c003 at comparable stages. Failure items remain dominated by the same cases: dev_move_002 (baseline mean 0.0337, essentially always failed, E_wrong_colour) and dev_move_006 (baseline 0.0, sd 0.0), while dev_move_007 (baseline 0.971) and dev_move_010 (0.981) are solved; the F-stage is where the gain would have to come from.

## Hypothesis
If the F-present rescaling frames carry a real signal, a larger dose (8/16 rows, the allowed maximum of non-final rows) over a wider span of the rescale (frames 36-48, from F's first size to near its final size) should increase scale F_cell / F_transform and, if the signal transfers across tasks, also move F_cell. A dose-response (c004 -> c005 -> c006) is the cheapest check that c005's small gain is not noise; the confirmation stage will decide.

## Program
databuilder.py: each step 8 plain final-frame eval_format rows + 8 `late`-template rows ({description} + "Generate {target_spec}.") with target_frame random in 36..48, independent seeded clip streams, no context. Static for 100 steps. Differences from c005: n_late 6 -> 8, range 40-47 -> 36-48.

## What analysis.py measures
Windowed loss per template; loss probe every 20 steps for final vs frame-42 targets on 8 held-out clips; generate probes at 49 and 99 with mean of each returned sub-score. Compare the step-49 probe with c005 (0.396; E 0.49, F 0.22, F_tr 0.54) and the dev F_missing count.

## Expected result
Dev 0.40-0.44 (scale F_cell >= 0.2, F_transform >= 0.6; move unchanged to +0.03). Refuted if dev < 0.38 or F_missing >= 6 on scale, which would mean the dose starts to dilute the final mapping or wide-range frames near the F start confuse the final size.
