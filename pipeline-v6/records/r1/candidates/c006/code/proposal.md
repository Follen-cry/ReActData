## Evidence
Dev means: c000 0.149; c001 0.1245; c002 0.1818; c003 0.1774; c004 0.1649; c005 0.1345 (D +0.008, CI [-0.015,+0.032]). c005 (move top-43, scale whole pool, geo small-edit half) gave move only 0.166 although the same fixed move set gave 0.260 (c003) and 0.238 (c004): so the move effect is not by itself reproducible, and single-run noise is large. A more robust pattern is the analogy-task sum (scale+move): c002 0.418, c003 0.403, c004 0.394 (large-edit selection on both tasks) vs c005 0.307 (large-edit on move only), c001 0.281, c000 0.332. Whenever both analogy tasks were drawn from large-change_px clips, the sum was ~+0.07 above baseline. Items: dev_move_007 scored 1.0 in c003 (0.241 in c002, baseline mean 0.283 sd 0.41, i.e. huge item variance); dev_scale_021 (c003) E_cell 1.0 but F_transform 0.07; dev_geo_000 scored 0.0 in c004 (deformed_shape; baseline mean 0.146). Geo: shape_fidelity was 0.25-0.35 and deformed_shape 28-36 in every change_px-selected run, including the small-edit half (c005: 0.347, 28), versus 0.563 / 15 with a plain random subset (c000), and geo scores stayed at 0.097-0.129 against 0.116 baseline: selection on geo does not pay, either direction.

## Hypothesis
Large-edit selection helps the two analogy tasks (a pooled effect of about +0.07 on their sum, +0.035 on the dev mean), while geo is best left as an unselected random subset, which keeps its shape fidelity (0.56) and baseline level (~0.116). Expect the combination to beat c000 on the dev mean without the geo penalty seen in c004/c005.

## Program
Started from c005. Quota 43/43/42. scale and move: uniform random draw of 43 from the 80 clips with the largest change_px (top 50%, as in c002). geo: uniform random from all 160 geo clips (no selection). One seeded mixed order. Rule-based, so it generalises across run seeds. analysis.py unchanged.

## What analysis.py measures
Per-task training loss, loss probes on inner-probe clips after steps 0,2,4,6, and a generate probe after step 7 (per-task score and sub-scores). Read: geo shape_fidelity in the generate probe and deformed_shape count on dev (target back near 15), analogy probe losses.

## Expected result
Scale ~0.18-0.24, move ~0.16-0.22, geo ~0.11-0.13; dev mean ~0.16-0.19 (about +0.02-0.03 over baseline, within noise of a single run). Refuted if scale+move sum is again near baseline (~0.33) or geo deformed_shape stays above 25.
