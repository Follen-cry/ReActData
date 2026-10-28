# c005 — mid-rescale frames (40-47) as extra targets, 6 of 16 rows, all 100 steps

Started from c004's /work copy (same structure, different target-frame range).

## Evidence
Baseline 0.391. c004 (6/16 rows target settled frames 50-57, other 10 plain): 0.3913, D +0.005 CI [-0.020, +0.031]; move 0.392 (-0.003), scale 0.391 (+0.012); sub-scores move E_cell 0.580, F_cell 0.328, F_transform 0.230 (baseline 0.639/0.349/0.181), scale F_cell 0.153 (baseline 0.116), F_transform 0.597 (0.577). So diverting 6 rows into a different template is harmless as long as the target content is F-present/final-like. Earlier programs with F-absent targets (c001 0.343, c003 0.350) or an F-absent context (c002 0.297) were worse: F_missing went up (c001 move 14, scale 11; baseline 8 / 1; c004 8 / 2). Items: dev_move_000 (0.519 baseline c000, 0.0 in c001), dev_move_003 (0.644 -> 0.35 in c001), dev_move_001 (baseline mean 0.295, 0.0 in c001), all rows where E/F content was wrong after F-absent supervision. c004's probe: generate score 0.355 at step 49, comparable to the baseline-like end level.

## Hypothesis
The informative process content with F present is the rescaling trajectory (frames ~40-47: F at intermediate size between its start size and final size). Training to generate these states, with the description naming the size change, may link the size words to a continuum of F sizes and sharpen F_transform/F_cell on scale, and (via the generic "F derived from E by the rule" structure) possibly the move task. Because F is present and E is correct in these frames, the F_missing failure mode seen with E-only targets should not arise.

## Program
databuilder.py: each step 10 plain final-frame eval_format rows + 6 `late`-template rows ({description} + "Generate {target_spec}.") with target_frame random in 40..47; independent seeded clip streams; no context. Static. Only the frame range differs from c004.

## What analysis.py measures
Windowed loss per template; loss probe every 20 steps for final vs frame-44 targets on 8 held-out clips; generate probes at 49 and 99 with the mean of each returned sub-score. Compare F-related probe sub-scores with c004 (step 49: 0.355; E 0.37, F 0.22, F_tr 0.54) and the F_missing share in the dev error categories.

## Expected result
Scale F_transform +0.05 or more and F_cell +0.03; move unchanged to slightly better; dev 0.39-0.43. Refuted if F_missing rises (>= 10 on move) or dev < 0.36, which would mean any non-final F states confuse the generated final size.
