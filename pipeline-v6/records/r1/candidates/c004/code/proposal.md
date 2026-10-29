## Evidence
Results: c000 baseline 0.149; c001 (mix 56/56/16) 0.1245; c002 (all tasks random from top-50% change_px) 0.1818 with scale 0.246 / move 0.172 / geo 0.127; c003 (scale and move = the 43 largest change_px clips, geo top 50%) 0.1774 with scale 0.1425 / move 0.2604 / geo 0.129 (D vs baseline +0.051, CI [+0.026,+0.079]; move +0.117 CI [+0.066,+0.173], scale +0.013). So the same intervention gave the gain on different tasks: scale needed the broader top-50% pool, move benefited from the extreme top-43. Items from c003: dev_move_007 scored 1.0 (c002: 0.241; baseline mean 0.283, sd 0.41); dev_move_006 stays 0.0 (E_wrong_colour, F_missing, extra_objects) in every program; dev_scale_020 scored 0.037 (baseline mean 0.075) with E_wrong_colour and F_wrong_transform, and dev_scale_021 0.368 with E_cell 1.0 but F_transform 0.07 (baseline 0.345). Scale F_transform dropped to 0.263 in c003 from 0.56 in c002, error F_wrong_transform 37 vs 20. Geo is stable at ~0.13 in both (position_alignment 0.15-0.17 vs 0.05 baseline, shape_fidelity 0.25-0.30 vs 0.56).

## Hypothesis
Per-task edit-size selection is the working lever, but its best form differs by task: scale tolerates/needs diversity (top 50%, 80 clips; the 43 largest scale clips are probably a narrow size band that hurts F-scale generalisation), move gains from the strongest edits. Combining the best observed rule per task (scale: random from top 50%; move: top 43; geo: top 50%) should give roughly additive gains. This is a test of additivity of the two task-specific effects; single run noise is large (sd 0.043), so it is the combination to carry into confirmation.

## Program
Started from c003. databuilder.py: defaults now embedded (per-task top fractions scale 0.5, move 0.27 -> the 43 largest, geo 0.5), quota 43/43/42, uniform random draw within each task's eligible pool, one seeded mixed order. Rule-based, generalises over seeds (move subset is fixed, scale/geo draws vary with seed). analysis.py unchanged.

## What analysis.py measures
Per-task training loss, loss probes after steps 0,2,4,6 and generate probe after step 7 per task (as before). Read: scale probe/generate vs c002, move vs c003; the dev per-task numbers tell whether effects are additive.

## Expected result
Scale ~0.22-0.25, move ~0.22-0.26, geo ~0.13; dev mean ~0.20-0.21 (+0.05-0.06 over baseline). Refuted if scale stays near 0.14 (then c002's scale gain was a lucky draw rather than the broad pool) or move falls to ~0.17.
