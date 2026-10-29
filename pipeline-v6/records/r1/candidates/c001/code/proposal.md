## Evidence
Only c000 (baseline) exists. Read report.md, results.json, dev_items/c000.json, baseline_replicates.json, analysis_traces/c000.json (only mean loss, no probes). Baseline dev 0.149 (replicates 0.068-0.179, sd 0.043). Per task: geo 0.116 with rotation_angle 0.036 and position_alignment 0.050 (shape_fidelity 0.563 already high); move 0.164 (E_cell 0.22, F_transform 0.20); scale 0.168 (E_cell 0.16, F_transform 0.36). Items: dev_geo_000 scored 0.109 with rotation 0.0 and position 0.006 and all of wrong_rotation/wrong_position/outline_left; dev_geo_001 0.114 (rotation 0.0, position 0.020, shape 0.75), so geo gains come only from shape fidelity, which is already learned; dev_geo_002 0.041 (deformed). Baseline sd on geo items is large (e.g. dev_geo_000 sd 0.138). Error categories on the analogy tasks (E_wrong_colour 30/23, F_wrong_transform 29/21 of 50) are the kind of discrete rule errors that more examples can fix. I did not inspect failure images in detail.

## Hypothesis
In 8 steps the geometric rotation/position skill is essentially not learned (about 0.04 sub-scores) so the 43 geo rows are mostly wasted, while the analogy tasks (colour recolouring of E, scale/move of F) have partial credit and visible learnable structure. Moving rows from geo to scale/move (56/56/16) should raise the scale and move means more than it costs geo (geo has little to lose; shape fidelity should survive with 16 rows).

## Program
Started from c000 (baseline). databuilder.py: quota scale 56 / move 56 / geo 16 (from candidate.yaml), clips drawn uniformly at random without replacement within a task, one seeded shuffle so every step mixes tasks. Rule-based, so it generalises across seeds. declared_class: static.

## What analysis.py measures
Per-step mean training loss per task, grad norm; loss probe on all 32 inner-probe clips after steps 0,2,4,6 (per-task mean loss, shows whether each task's loss falls); one generate probe on all probe clips after step 7 (per-task score and sub-scores). Next step: if the scale/move probe losses fall faster than geo and the generate probe is higher for them, the reallocation works; if geo probe score collapses, 16 rows is too few.

## Expected result
Scale +0.03 to +0.06, move +0.03 to +0.06, geo -0.02 to 0 relative to baseline; dev mean about +0.02 to +0.04 (within or near noise, sd 0.043). Refuted if scale/move do not improve despite 30% more rows, i.e. more rows of these tasks do not help in 8 steps.
