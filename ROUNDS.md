# Rounds of reactdata-v5 / r1

Generated from `runs/r1/state.json`, `confirm.json`, each candidate's `proposal.md` and each round's `patch.diff` (diff of the agent's `/work` against the program it started from = its parent).

Baseline (c000, fixed program) dev on run seeds 1000–1004: 0.391, 0.355, 0.396, 0.420, 0.371 (mean 0.386, sd_train = δ = 0.025). Base (untrained) dev 0.005.
Screening D is against the per-item mean of the baseline replicates; confirmation is top-3 × 4 paired seeds with Bonferroni (m = 3). Selection rule: D ≥ δ and CI lower bound > 0, otherwise the baseline.

| round | cand | parent | class | dev | D | D scale / move | confirm D [CI] |
|---|---|---|---|---|---|---|---|
| 1 | [c001](#c001) | c000 | schedule | 0.343 | -0.043 | -0.028 / -0.058 | — |
| 2 | [c002](#c002) | c001 | schedule | 0.297 | -0.089 | -0.091 / -0.087 | — |
| 3 | [c003](#c003) | c002 | schedule | 0.350 | -0.037 | -0.013 / -0.060 | — |
| 4 | [c004](#c004) | c003 | static | 0.391 | +0.005 | +0.012 / -0.003 | — |
| 5 | [c005](#c005) | c004 | static | 0.401 | +0.014 | +0.021 / +0.007 | — |
| 6 | [c006](#c006) | c005 | static | 0.415 | +0.028 | +0.032 / +0.024 | +0.030 [-0.019, +0.078] |
| 7 | [c007](#c007) | c006 | static | 0.404 | +0.018 | +0.015 / +0.021 | — |
| 8 | [c008](#c008) | c007 | static | 0.410 | +0.023 | +0.064 / -0.017 | -0.026 [-0.075, +0.023] |
| 9 | [c009](#c009) | c008 | schedule | 0.409 | +0.023 | +0.025 / +0.020 | +0.025 [-0.024, +0.074] |

**Selected**: `c000` — no confirmed candidate passed: fixed baseline

## Per candidate

### c001

- **Round** 1 · parent `c000` · class `schedule` · agent calls 1 ($0.22)
- **Screening** dev 0.343, D vs baseline mean -0.043 (95% CI [-0.072, -0.015]; scale -0.028, move -0.058)
- **Files changed vs parent**: `databuilder.py` +27/−21, `analysis.py` +37/−8, `text_templates.yaml` +1/−1, `candidate.yaml` +3/−4, `proposal.md` +18/−3, `memory.md` +1/−1
- **Hypothesis**: Rows whose target is the frame where E already has its new colour and F does not exist yet (frame ~26) supervise the first analogy rule (A->B applied to D) in isolation, task-independently. This should raise E_cell on both tasks (it is shared by move). Dropping these rows after step 70 lets the last 30 steps be pure test-time mapping so the model ends on the test input distribution.
- **What analysis.py measures**: Windowed mean loss per target frame (normalised comparison is only within a target); loss probe every 20 steps on 8 held-out inner clips for final and frame-26 targets (does sub-goal loss fall faster than final loss?); generate probes at steps 49 and 99 on the same 8 clips (rule-score of F0 -> final generation, mid-run vs. end, to see whether removing sub-goals hurts or helps).
- Records: [proposal](runs/r1/rounds/001/c001/work/proposal.md) · [patch.diff](runs/r1/rounds/001/c001/patch.diff) · [final code](runs/r1/candidates/c001/code/) · [agent input report](runs/r1/rounds/001/c001/input/report.md) · [agent output](runs/r1/rounds/001/c001/agent_0/output.json) · [checks](runs/r1/rounds/001/c001/checks.json)

### c002

- **Round** 2 · parent `c001` · class `schedule` · agent calls 3 ($1.31)
- **Screening** dev 0.297, D vs baseline mean -0.089 (95% CI [-0.119, -0.060]; scale -0.091, move -0.087)
- **Files changed vs parent**: `databuilder.py` +21/−24, `analysis.py` +18/−10, `text_templates.yaml` +1/−1, `candidate.yaml` +2/−2, `proposal.md` +7/−7, `memory.md` +2/−1
- **Hypothesis**: The process (E first, F derived from E) may still help if shown as *input* rather than as a target: F0 + the E-state frame (frame 27) as ViT context -> final frame. The row keeps the correct target and the final-frame output distribution, but the ViT sees a correct E, so the gradient on the F-derivation (E -> F via the B->C rule) is cleaner and attributes F to E. The context is dropped after step 60, so the last 40 steps train plain test-time rows (and half the earlier rows are already plain), so the model cannot depend on the context at test time.
- **What analysis.py measures**: Windowed training loss per template (ctx_final vs eval_format: if ctx loss is much lower, the model leans on the context); loss probe every 20 steps on 8 held-out clips for both templates (final target); generate probes at 49 and 99 (plain F0 -> final) logging the mean of every sub-score key the probe returns (probe keys differ from the dev E_cell/F_cell/F_transform), comparing with c001's 0.163 (step 49) / 0.358 (step 99).
- Records: [proposal](runs/r1/rounds/002/c002/work/proposal.md) · [patch.diff](runs/r1/rounds/002/c002/patch.diff) · [final code](runs/r1/candidates/c002/code/) · [agent input report](runs/r1/rounds/002/c002/input/report.md) · [agent output](runs/r1/rounds/002/c002/agent_0/output.json) · [checks](runs/r1/rounds/002/c002/checks.json)

### c003

- **Round** 3 · parent `c002` · class `schedule` · agent calls 1 ($0.60)
- **Screening** dev 0.350, D vs baseline mean -0.037 (95% CI [-0.065, -0.010]; scale -0.013, move -0.060)
- **Files changed vs parent**: `databuilder.py` +22/−21, `analysis.py` +8/−13, `text_templates.yaml` +1/−1, `candidate.yaml` +2/−2, `proposal.md` +7/−7, `memory.md` +2/−2
- **Hypothesis**: Sub-goal rows probably help or hurt through *how long they displace the test format*. A short warm-up (40 steps) followed by 60 steps of 16 plain rows lets the model take the E-state sub-goal as an initial scaffold, then consolidate on the exact test mapping. If c001's loss came from late interference (F_missing: model stops at the E-state), this should come back to baseline or above; if the warm-up still hurts, sub-goal targets at this data size are simply harmful and later candidates should stop using intermediate targets.
- **What analysis.py measures**: Windowed loss per template; loss probe every 20 steps on 8 held-out clips for final vs sub-goal targets; generate probes at step 39 (end of warm-up, compare with c001's 0.163 at 49 and baseline-like level) and step 99 (compare with c001's 0.358 and c002's), logging every sub-score key returned.
- Records: [proposal](runs/r1/rounds/003/c003/work/proposal.md) · [patch.diff](runs/r1/rounds/003/c003/patch.diff) · [final code](runs/r1/candidates/c003/code/) · [agent input report](runs/r1/rounds/003/c003/input/report.md) · [agent output](runs/r1/rounds/003/c003/agent_0/output.json) · [checks](runs/r1/rounds/003/c003/checks.json)

### c004

- **Round** 4 · parent `c003` · class `static` · agent calls 1 ($0.71)
- **Screening** dev 0.391, D vs baseline mean +0.005 (95% CI [-0.020, +0.031]; scale +0.012, move -0.003)
- **Files changed vs parent**: `databuilder.py` +10/−9, `analysis.py` +2/−2, `text_templates.yaml` +1/−1, `candidate.yaml` +3/−3, `proposal.md` +7/−7, `memory.md` +2/−2
- **Hypothesis**: To separate "intermediate-target content" from "diverted rows / extra template", use process frames whose content is identical to the answer: frames 50-57 are the settled final state (E and F both at their final size), so training to generate them is equivalent to the test target under small pixel variation, with a different template ("Generate {target_spec}"). If the damage was caused by F-absent targets, this should be ~baseline or slightly better (augmentation against memorising 16 exact images, 6 extra views per step); if it is also lower, any change of the row type is costly at this data/step scale, and later candidates should stay in the baseline format.
- **What analysis.py measures**: Windowed loss per template (plain vs late); loss probe every 20 steps for final vs frame-54 targets on 8 held-out clips; generate probes at steps 49 and 99 (mean of every returned sub-score) to compare with c001/c003 (0.16 at the end of their sub-goal phases; c001 0.358 at 99).
- Records: [proposal](runs/r1/rounds/004/c004/work/proposal.md) · [patch.diff](runs/r1/rounds/004/c004/patch.diff) · [final code](runs/r1/candidates/c004/code/) · [agent input report](runs/r1/rounds/004/c004/input/report.md) · [agent output](runs/r1/rounds/004/c004/agent_0/output.json) · [checks](runs/r1/rounds/004/c004/checks.json)

### c005

- **Round** 5 · parent `c004` · class `static` · agent calls 1 ($0.80)
- **Screening** dev 0.401, D vs baseline mean +0.014 (95% CI [-0.014, +0.042]; scale +0.021, move +0.007)
- **Files changed vs parent**: `databuilder.py` +5/−5, `candidate.yaml` +2/−2, `proposal.md` +7/−7, `memory.md` +2/−2
- **Hypothesis**: The informative process content with F present is the rescaling trajectory (frames ~40-47: F at intermediate size between its start size and final size). Training to generate these states, with the description naming the size change, may link the size words to a continuum of F sizes and sharpen F_transform/F_cell on scale, and (via the generic "F derived from E by the rule" structure) possibly the move task. Because F is present and E is correct in these frames, the F_missing failure mode seen with E-only targets should not arise.
- **What analysis.py measures**: Windowed loss per template; loss probe every 20 steps for final vs frame-44 targets on 8 held-out clips; generate probes at 49 and 99 with the mean of each returned sub-score. Compare F-related probe sub-scores with c004 (step 49: 0.355; E 0.37, F 0.22, F_tr 0.54) and the F_missing share in the dev error categories.
- Records: [proposal](runs/r1/rounds/005/c005/work/proposal.md) · [patch.diff](runs/r1/rounds/005/c005/patch.diff) · [final code](runs/r1/candidates/c005/code/) · [agent input report](runs/r1/rounds/005/c005/input/report.md) · [agent output](runs/r1/rounds/005/c005/agent_0/output.json) · [checks](runs/r1/rounds/005/c005/checks.json)

### c006

- **Round** 6 · parent `c005` · class `static` · agent calls 1 ($0.89)
- **Screening** dev 0.415, D vs baseline mean +0.028 (95% CI [+0.001, +0.057]; scale +0.032, move +0.024)
- **Confirmation** (seeds 1001–1004, paired) D +0.030 [-0.019, +0.078]
- **Files changed vs parent**: `databuilder.py` +6/−6, `candidate.yaml` +2/−2, `proposal.md` +7/−7, `memory.md` +2/−2
- **Hypothesis**: If the F-present rescaling frames carry a real signal, a larger dose (8/16 rows, the allowed maximum of non-final rows) over a wider span of the rescale (frames 36-48, from F's first size to near its final size) should increase scale F_cell / F_transform and, if the signal transfers across tasks, also move F_cell. A dose-response (c004 -> c005 -> c006) is the cheapest check that c005's small gain is not noise; the confirmation stage will decide.
- **What analysis.py measures**: Windowed loss per template; loss probe every 20 steps for final vs frame-42 targets on 8 held-out clips; generate probes at 49 and 99 with mean of each returned sub-score. Compare the step-49 probe with c005 (0.396; E 0.49, F 0.22, F_tr 0.54) and the dev F_missing count.
- Records: [proposal](runs/r1/rounds/006/c006/work/proposal.md) · [patch.diff](runs/r1/rounds/006/c006/patch.diff) · [final code](runs/r1/candidates/c006/code/) · [agent input report](runs/r1/rounds/006/c006/input/report.md) · [agent output](runs/r1/rounds/006/c006/agent_0/output.json) · [checks](runs/r1/rounds/006/c006/checks.json)

### c007

- **Round** 7 · parent `c006` · class `static` · agent calls 1 ($0.99)
- **Screening** dev 0.404, D vs baseline mean +0.018 (95% CI [-0.007, +0.044]; scale +0.015, move +0.021)
- **Files changed vs parent**: `databuilder.py` +5/−5, `candidate.yaml` +2/−2, `proposal.md` +7/−7, `memory.md` +2/−2
- **Hypothesis**: Since dose cannot go up further, vary the *coverage*: targets across the whole F phase (34-56, including settled frames that were neutral in c004) make the extra rows a continuous video-like view of F being rescaled and settling, which should keep the c006 gain and may restore F_transform (the one sub-score that fell on scale), since the end of the rescale (settled frames) contains the final size that the test asks for. A wider range also reduces the share of ambiguous early-F frames (34-36) per row relative to c006 only mildly, so I expect c007 to land near c006; the test is whether the wider range helps F_transform or whether the narrower mid-range was the active ingredient.
- **What analysis.py measures**: Windowed loss per template; loss probe every 20 steps for final vs frame-45 targets (8 held-out clips); generate probes at 49 and 99 with the mean of each returned sub-score; read the step-99 probe against c006's and the dev F_transform on scale (c006 0.501) and E_wrong_colour counts.
- Records: [proposal](runs/r1/rounds/007/c007/work/proposal.md) · [patch.diff](runs/r1/rounds/007/c007/patch.diff) · [final code](runs/r1/candidates/c007/code/) · [agent input report](runs/r1/rounds/007/c007/input/report.md) · [agent output](runs/r1/rounds/007/c007/agent_0/output.json) · [checks](runs/r1/rounds/007/c007/checks.json)

### c008

- **Round** 8 · parent `c007` · class `static` · agent calls 1 ($1.14)
- **Screening** dev 0.410, D vs baseline mean +0.023 (95% CI [-0.003, +0.051]; scale +0.064, move -0.017)
- **Confirmation** (seeds 1001–1004, paired) D -0.026 [-0.075, +0.023]
- **Files changed vs parent**: `databuilder.py` +21/−10, `analysis.py` +1/−0, `text_templates.yaml` +1/−0, `candidate.yaml` +2/−2, `proposal.md` +7/−7, `memory.md` +2/−2
- **Hypothesis**: Conditioning a later process frame on an earlier F-present frame of the same clip (frame 34-42 as context, target >= 4 frames later, up to 56) trains the model to extrapolate F's own trajectory (size change applied to an existing F), the part of the rule the single-frame targets did not improve (F_transform). Final-frame rows stay context-free, so the test-time format is untouched, and the other 4 process rows keep the c006 content. The context is given only where F is present, avoiding the F-absent confusion of c002.
- **What analysis.py measures**: Windowed loss per template (eval_format / late / late_ctx — if late_ctx loss is far below late loss the context is being copied); loss probe every 20 steps on 8 held-out clips for final, frame-45 plain, and frame-50-with-ctx-38; generate probes at 49 and 99 (plain F0 -> final) with mean of every sub-score returned. Compare the step-99 probe to c006/c007.
- Records: [proposal](runs/r1/rounds/008/c008/work/proposal.md) · [patch.diff](runs/r1/rounds/008/c008/patch.diff) · [final code](runs/r1/candidates/c008/code/) · [agent input report](runs/r1/rounds/008/c008/input/report.md) · [agent output](runs/r1/rounds/008/c008/agent_0/output.json) · [checks](runs/r1/rounds/008/c008/checks.json)

### c009

- **Round** 9 · parent `c008` · class `schedule` · agent calls 1 ($1.25)
- **Screening** dev 0.409, D vs baseline mean +0.023 (95% CI [-0.004, +0.051]; scale +0.025, move +0.020)
- **Confirmation** (seeds 1001–1004, paired) D +0.025 [-0.024, +0.074]
- **Files changed vs parent**: `databuilder.py` +8/−18, `candidate.yaml` +3/−3, `proposal.md` +7/−7, `memory.md` +2/−2
- **Hypothesis**: Because the test is the final-frame mapping on move, the end of training should be pure test-time rows. In c006 the last 25 steps still had 8 of 16 rows on process frames. Dropping the process rows after step 75 keeps the early/middle regularising effect (E-colour gain, less memorisation of the 16 exact final images) while letting the last updates fit the exact final mapping; this is the "schedule that removes them before the end" lever from the task description.
- **What analysis.py measures**: Windowed loss per template; loss probes every 20 steps (final vs frame-42 plain vs ctx probe for continuity); generate probes at 49 and 99 with mean of each returned sub-score; compare step-99 generation with c006 and the dev move/scale sub-scores.
- Records: [proposal](runs/r1/rounds/009/c009/work/proposal.md) · [patch.diff](runs/r1/rounds/009/c009/patch.diff) · [final code](runs/r1/candidates/c009/code/) · [agent input report](runs/r1/rounds/009/c009/input/report.md) · [agent output](runs/r1/rounds/009/c009/agent_0/output.json) · [checks](runs/r1/rounds/009/c009/checks.json)
