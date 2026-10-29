# c001 — E-sub-goal curriculum

## Evidence
Only the baseline c000 exists. It scores 0.0 on both tasks (dev_move_000, dev_move_001, dev_scale_000: all sub-scores 0.0; categories include E_missing, F_missing, context_damaged, extra_objects). dev_scale_001 has baseline mean 0.0976 (sd 0.138), the only non-zero item. Failures are dominated by E_missing/F_missing, i.e. the model does not even produce the recoloured E cell, let alone F. The run here has 4 steps x 16 rows, loss 0.0148 -> 0.0061 (analysis trace c000). Baseline signal is at floor, so any gain must first come from E.

## Hypothesis
Rows that target the intermediate frame (E recoloured, F not yet present) isolate the first rule (colour) as a sub-goal and give the model a cleaner gradient for the E cell; mixing them with final-frame rows (and dropping them for the last 30% of steps) should raise E_cell and then F_cell on both tasks, since the colour rule is shared with move.

## Program
Started from c000 (baseline). Per step: 6 rows target frame 26 (template `subgoal`: "Generate {target_spec}. {description}", no wrapped description) and 10 rows target the final frame with the eval text, no context frames. Sub-goal rows are used while step < 0.7*total_steps, then all 16 rows are final. Clips come from the baseline's seeded permutation. Declared class: schedule.

## What analysis.py measures
Per-step mean loss by target frame (sub-goal vs final), row counts, running means, grad norm, and a generate probe on 16 held-out clips at the last step (rule scores of generations). Compare final-frame loss against c000's trace; read probe sub-scores for F-cell quality.

## Expected result
E_cell up on both tasks (+0.05 to +0.1 dev mean at best); F_transform on move unlikely to change. Refuted if dev stays 0.0 or final-frame loss is worse than baseline's, meaning the sub-goal rows just displace final-frame training (10 vs 16 final rows).
