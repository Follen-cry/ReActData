# c002 — sub-goal chain (E-done and F-midway) with 8 final rows

## Evidence
c000 and c001 both score 0.0 on dev (c001 vs baseline D = -0.0244, CI [-0.0488, 0]; the only difference is dev_scale_001: 0.0 vs baseline mean 0.0976, sd 0.138, i.e. noise-level). dev_move_000 (E_missing, F_missing, blank_or_repainted), dev_move_001 (E_wrong, F_wrong_colour) and dev_scale_000 (E_missing, F_missing) are all 0.0 with every sub-score 0.0 under both programs. The c001 probe on 16 held-out clips scored 0.0 everywhere. Loss in c001: sub-goal rows 0.0107 at step 0 (running mean 0.028 later), final rows 0.0146 -> 0.0060; nothing but loss moved in 4 steps. The budget is 4 steps, so at most two stages are possible.

## Hypothesis
With so few steps, the sub-goal signal in c001 (6 rows, one frame) was too small and too uniform. A chain of two sub-goals (E done at t=26, F half-rescaled at t=42) at 8 rows/step covers the whole process E -> F, so the model sees every state of the analogy; the final step is final-only. If process supervision carries any signal in this budget, it should show as non-zero E_cell/F_cell.

## Program
Started from c001. Steps with step < 0.7*total: 8 rows alternate targets 26 and 42 (template `subgoal`), 8 rows final with the eval text; later steps are all final. No context frames. declared_class: schedule.

## What analysis.py measures
Unchanged from c001: loss per target frame (26, 42, 59), row counts, grad norm, and a generate probe on 16 held-out clips on the last step. Next step: compare final-frame loss and probe sub-scores with c001 (all 0.0).

## Expected result
Small at best (E_cell +0.02..0.05); most likely still 0.0 on dev because 4 optimizer steps barely move the model. Refuted if dev and probe stay 0.0, meaning that screening at this budget cannot discriminate process-frame use, and lower-noise levers (e.g. more final-frame rows) are needed.
