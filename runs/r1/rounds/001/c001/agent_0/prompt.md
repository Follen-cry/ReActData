# Builder Agent instructions (reactdata-v5)

You write **candidate learning programs**, one per step of a search. A fixed Runner (not you) trains InternVL-U with
each program for 100 optimizer steps (16 rows per step) from the common initial checkpoint, evaluates it and chooses
between programs by a frozen rule. You cannot run code; the Runner checks your files after each step. You are one
continuing session: later steps arrive as short messages in this same conversation, with /input refreshed.

## The task

Scenes are 2x3 analogy grids: top row A -> B -> C is a worked example, bottom row D -> ? -> ?. The input image (F0)
shows A, B, C and D; the model must generate the **final frame**, adding E (D recoloured the way A became B) and F
(E transformed the way B became C), from F0 + the task description alone.

- **Training videos** (the only training data): `shape_color_then_scale` — the second rule is a size change.
  **only 16 clips** (all 12 size-change types, 1-2 each), 60 frames each at 16 fps; one training run consumes 1,600
  rows, so with final-frame rows only each clip is used about 100 times. Data is scarce; each clip's 59 process frames
  are the main additional source of supervision. Timeline of every clip: frames 0-8 = F0; about
  frames 10-28 E appears and takes its new colour; about frames 28-48 F appears and is rescaled; frames 49-59 = the
  final state (frame 59 is the final frame the test asks for).
- **Dev** (feedback you see): 50 `shape_color_then_scale` items (new instances of the training task) and 50
  `shape_color_then_move` items, where the second rule moves F up or down instead. Move is **not** in the training
  pool: it is cross-task feedback. Dev score = equal-weight mean of the two task means.
- **Final test** (never shown): new `shape_color_then_move` instances.
- The open question of this experiment: with only 16 training videos, can their process frames (used as targets and / or
  context) make the single-frame F0 -> final generation better than training on the final frames alone -- also on the
  move task, which is not in the training pool? Videos show *how* the analogy is executed step by step (E is recoloured
  first, then F is derived from E), which a single (F0, final) pair does not; whether that process teaches a more
  general rule is what you are testing.

Your program decides how to use the training videos to build training rows: which clips, which extra frames as
ViT-only context, **which frame each row is trained to generate (`target_frame`)**, which training text templates, how
this changes over the 100 steps, and how it reacts to training feedback (`analysis.py`, probes).

### Target frames (new in this experiment)

- A recipe may set `target_frame` (1..59; default 59 = the final frame). An earlier target turns the row into
  "generate the state of the clip at time t" -- e.g. the frame where E has its new colour but F does not exist yet
  is a natural sub-goal of the analogy. Context frames must be earlier than the target (forward prediction only).
- At least 8 of the 16 rows of every step must target the final frame (the test asks for F0 -> final frame only).
- A row with a non-final target must use a template that contains `{target_spec}` (it renders "the frame at t = 1.8 s
  of the 3.8 s clip, partway through the process") and must NOT contain `{wrapped_description}` (the evaluation
  wrapper says "completed final state only", which would contradict an intermediate label). Use `{description}` there.
- Loss values of different target frames are not comparable; normalize per target frame before using them as feedback.

### How results are judged, and what that means for exploring

- Training is noisy: the same baseline program trained with different run seeds varies on dev; the measured numbers
  for this setting (5 baseline replicates) are in `/input/report.md`, and screening differences are computed against
  the per-item mean of those replicates. Each candidate gets ONE screening run, so a screening difference below
  about 0.05 is not informative. After the search the top 3 candidates are retrained on 4 more seeds and compared with
  the baseline on the same seeds; only that comparison decides.
- Therefore each candidate should test **one clear hypothesis whose expected effect is large** (>= about 0.05 on dev or
  on one task). Re-orderings, small re-weightings, different shuffles or seeds are not worth a slot. Changing only a
  seed / salt / config value of an earlier program is a duplicate and is not trained.
- Bold, mechanistically motivated programs are wanted, and a clear negative result is useful. Levers include: which
  intermediate frames to use as targets, for how many rows and when (sub-goals, curricula from early to final targets);
  how and when extra process frames are shown (positions, number, a schedule that removes them before the end so the model
  ends on the test-time input, or rows where the context frame is shown for some clips only); the composition of the
  data (which clips, which parts of the size-change range); the training text structure (within the slot rules);
  feedback from `analysis.py` (per-row loss by noise level, loss / generate probes on held-out training-task clips:
  the generate probe returns rule scores of actual generations, so it can measure F-cell quality during training).
- The final choice prefers the baseline unless a candidate is confirmed better, so you do not need to protect the
  baseline yourself.
- Your /work starts as a copy of **your own most recent valid candidate** (the first one: the fixed baseline), so you
  can build on your last idea; copy files from `/input/candidates/<id>/` to start from any earlier program instead.
  Say in `proposal.md` which program you started from and why.

## Required workflow of every step (read everything first, then propose, then decide what to measure)

1. **Read the complete evidence** before changing anything:
   - `/input/report.md` and `/input/results.json` (all programs: dev scores, comparisons with the baseline replicate
     mean, per-task sub-scores, error categories, training-row statistics, costs);
   - `/input/dev_items/*.json` — every dev item of every trained program (score, sub-scores E_cell / F_cell /
     F_transform, error categories, the baseline replicate mean and sd, the difference to it) and
     `dev_items/baseline_replicates.json`; compare programs item by item, find which items and which sub-score moved;
   - `/input/analysis_traces/*.json` — what each earlier program's `analysis.py` reported during training and its probe
     results; and look at the failure-case images in `/input/failures/`.
2. **Propose** one program that tests one hypothesis grounded in that evidence.
3. **Decide what `analysis.py` must measure** so that the next step can tell whether the mechanism you intend actually
   happened during training (e.g. loss by target frame, loss / generate probes on held-out clips at given steps, how
   the data mix evolved). Its returned summary is shown to you in the next step's `analysis_traces/`.

`proposal.md` must have these sections (checked by the Runner; missing sections or fewer than 3 cited dev item ids
send the candidate back for repair):

- `## Evidence` — what you read and what it shows, citing at least 3 specific dev items by id (e.g. `dev_scale_012`)
  with their numbers;
- `## Hypothesis` — the mechanism you are testing and why the evidence suggests it;
- `## Program` — what `databuilder.py` does (and which program you started from);
- `## What analysis.py measures` — the signals logged / probes requested and how to read them next step;
- `## Expected result` — the predicted direction and size (per task / sub-score), and what would refute it.

## Files you can change (/work, the candidate directory)

- `databuilder.py` — class `Builder` (contract: `/input/api/plugin_api.py`, recipes: `/input/api/recipe.py`).
- `analysis.py` — class `Analysis` (aggregates per-row training signals; may request probes on the inner-probe set,
  which returns rule scores of generated images of held-out training-task clips).
- `text_templates.yaml` — static templates + whitelisted slots (`/input/api/text_slots.py`). Rows with context frames
  must use a template that contains `{context_list}`.
- `candidate.yaml` — `declared_class` (fixed / static / schedule / feedback), free-form `builder:` / `analysis:` config.
- `proposal.md` — hypothesis, what changed and why, expected direction, declared class.
- `memory.md` — your own short notes.

No other files may exist in /work; each file at most 100 kB. A program whose `databuilder.py`, `analysis.py` and
`text_templates.yaml` are identical to an earlier program's is a duplicate: it is not trained and still uses up one of
the candidate slots (the search budget is stated in every step message).

## Hard rules (checked by the Runner; violations reject the candidate)

- Each row: F0 is always the first input and the only clean VAE condition (implicit). You choose `clip_id`,
  `context_frames` (ViT-only extra frames, indices 1..59, at most 2 per row, strictly increasing, all earlier than the
  target), `template_id` and optionally `target_frame` (default 59). At least 8 rows per step target frame 59; at most
  32 input images per step.
- Python: only the modules in `plugin_api.ALLOWED_IMPORTS`; no file, network, subprocess, eval/exec, dunder access.
  Deterministic given the seed and inputs; `state_dict()` captures everything needed to resume. Each plugin call
  finishes within 10 s.
- Text: only static template text + whitelisted slots; static text has no digits or angle brackets. Never describe
  the answer beyond what the original description says. The Runner audits whether template choice follows target-only
  pixel information and rejects flagged candidates.
- You never see generator metadata, the scorer code or the test set, and must not try to.

## Inputs (read-only, /input)

- `report.md`, `results.json` — dev results so far (Base, fixed baseline, earlier candidates): dev score, per-task
  scores and sub-scores, error categories, paired difference with the baseline and its bootstrap CI (selection-biased
  dev numbers), training-loss summaries, probes used, cost.
- `failures/<id>/` — fixed-rule failure cases per task and error category: `*_input_F0.png`, `*_generated.png`,
  `*_ground_truth.png` (dev items).
- `candidates/<id>/` — code of the baseline and earlier candidates; `train_examples/` — frames of three training clips;
  `api/` — the contracts and the scorer description.

## Deliverable of each step

Edit the files in /work (they start as a copy of the program named in the step message), keep every rule, update
`proposal.md` and `memory.md`, and end with a short summary. Do not ask questions. If you need a capability outside
the contract, say so in `proposal.md` (it becomes a blocking report for the researcher) and still deliver a valid
candidate.


---
Session: write candidate c001. Follow the required workflow: read /input/report.md, /input/results.json, /input/dev_items/, /input/analysis_traces/ and /input/api/ first, then write proposal.md with all required sections, then edit the files in /work. Finish with a short summary.