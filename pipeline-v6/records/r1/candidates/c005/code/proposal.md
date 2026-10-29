## Evidence
Dev means: c000 0.149; c001 0.1245; c002 0.1818 (scale 0.246, move 0.172, geo 0.127); c003 0.1774 (scale 0.1425, move 0.2604, geo 0.1293); c004 0.1649 (scale 0.1567, move 0.2378, geo 0.1002). The additivity test of c004 failed on scale: scale top-50% random gave 0.246 in c002 but 0.157 in c004, so c002's scale gain is likely a lucky draw (scale: no reproducible effect, c003 +0.013, c004 +0.027, CIs include 0). Move with the 43 largest-change_px clips is reproducible: 0.260 (c003, +0.117 CI [0.066,0.173]) and 0.238 (c004, +0.095 CI [0.039,0.155]); dev_move_007 went from 0.241 (c002) to 1.0 (c003). Geo: every large-edit selection (c002-c004) cut shape_fidelity to 0.25-0.30 from 0.563 and raised deformed_shape to 33-36 from 15; in c004 dev_geo_000 scored 0.0 (all sub-scores 0, deformed_shape) vs baseline mean 0.146 (sd 0.138). Position_alignment rose (0.10-0.17 vs 0.05) but not enough to offset: geo c004 is -0.007 vs baseline. dev_scale_021 (c003) had E_cell 1.0 but F_transform 0.07: scale's F-step is not learned by edit-size selection.

## Hypothesis
(1) Keep the one reproducible effect: move trained on the 43 largest-edit clips. (2) Large-edit geo clips teach big displacement but wreck the polygon's shape (deformed_shape up); selecting the small-edit half of geo clips (small rotations/displacements, closer to F0) should keep shape_fidelity near baseline (0.56) and may still improve position/rotation modestly, so geo should be at or above baseline instead of below. (3) Scale returns to the unselected random pool, since selection has no reproducible effect there.

## Program
Started from c004. Quota 43/43/42. move: random 43 from the 43 largest change_px (i.e. all of them, fixed set). scale: uniform random from the whole 160-clip scale pool. geo: uniform random from the 80 geo clips with the smallest change_px (sign flipped for geo via SMALL_FIRST). One seeded mixed order. Rule-based, generalises over seeds. analysis.py unchanged.

## What analysis.py measures
Per-task training loss per step, loss probes on inner-probe clips after steps 0,2,4,6, generate probe after step 7 with per-task sub-scores. Read next: geo probe loss (did it stop rising as in c001: 0.0075 -> 0.0138?) and geo shape_fidelity sub-score in the generate probe (target >= 0.5), move probe vs c003/c004.

## Expected result
Move ~0.23-0.26 (+0.09), scale ~0.14-0.17 (baseline level), geo ~0.12-0.15 (shape_fidelity 0.4-0.56, deformed_shape count back toward 15-20); dev mean ~0.18-0.19. Refuted if geo falls below ~0.10 again or deformed_shape stays >30, which would mean the deformation is caused by any geo training at this budget and not by edit size.
