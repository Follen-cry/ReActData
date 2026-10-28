# c007 — F-phase targets over the whole span (frames 34-56), 8 of 16 rows

Started from c006's /work copy; only the frame range changes (the dose is already at the maximum of 8 non-final rows).

## Evidence
Dose-response so far: c004 (6 rows, frames 50-57) 0.3913; c005 (6 rows, 40-47) 0.4005; c006 (8 rows, 36-48) 0.4147, D +0.028 CI [+0.001, +0.057] vs baseline 0.391 (selection-biased; move +0.025 [-0.007, +0.060], scale +0.032 [-0.012, +0.079]). Sub-scores c006 vs baseline: scale E_cell 0.643 vs 0.522, F_cell 0.151 vs 0.116, F_transform 0.501 vs 0.577 (down); move E_cell 0.648 vs 0.639, F_cell 0.361 vs 0.349, F_transform 0.189 vs 0.181. E_wrong_colour on scale fell to 11 (baseline 20), F_missing 4 on scale (baseline 1). So the gain of F-present process targets appears mostly in E (colour) and F position/cell, and not (yet) in the scale-transform sub-score. Probe: step-49 generate score 0.340 (E 0.50, F 0.10, F_tr 0.51), lower F than c005's 0.22 at the same step, i.e. mid-training probes are noisy (8 clips). Items: dev_move_002 and dev_move_006 remain at about 0 for the baseline (means 0.034 / 0.0); dev_move_007 (0.971) and dev_move_010 (0.981) stay solved, so gains must come from the mid-range items such as dev_move_000 (0.4195, sd 0.094) and dev_move_003 (0.409, sd 0.13).

## Hypothesis
Since dose cannot go up further, vary the *coverage*: targets across the whole F phase (34-56, including settled frames that were neutral in c004) make the extra rows a continuous video-like view of F being rescaled and settling, which should keep the c006 gain and may restore F_transform (the one sub-score that fell on scale), since the end of the rescale (settled frames) contains the final size that the test asks for. A wider range also reduces the share of ambiguous early-F frames (34-36) per row relative to c006 only mildly, so I expect c007 to land near c006; the test is whether the wider range helps F_transform or whether the narrower mid-range was the active ingredient.

## Program
databuilder.py: each step 8 plain final-frame eval_format rows + 8 `late`-template rows ({description} + "Generate {target_spec}.") with target_frame uniform in 34..56, independent seeded clip streams, no context. Static for 100 steps. Differences from c006: late_lo 36 -> 34, late_hi 48 -> 56.

## What analysis.py measures
Windowed loss per template; loss probe every 20 steps for final vs frame-45 targets (8 held-out clips); generate probes at 49 and 99 with the mean of each returned sub-score; read the step-99 probe against c006's and the dev F_transform on scale (c006 0.501) and E_wrong_colour counts.

## Expected result
Dev 0.40-0.44, scale F_transform back to >= 0.55; move within +-0.03 of c006. Refuted if dev < 0.38 (then the narrower, F-rescale-only range of c006 is better) or if F_missing rises above 6 on scale.
