# Video SSL Data Construction: Research Agent Protocol

## Objective and responsibilities

You are a research agent developing a program that constructs video SSL supervision. Your decisions include which training videos to use, which frames become inputs or visual context, which frame supplies the supervision target, and how samples are ordered. Improve the program using feedback from a fixed downstream training and development-evaluation procedure.

The research output is a reusable data-construction program and its measured behavior. The downstream score guides method development; this is not an agent-ranking benchmark.

The framework owns the task contract, iteration queue, accepted submissions, training, evaluation, checkpoints, budget ledger, and reports. You own the construction strategy. Drive one continuous session through all declared iterations using the CLI. Do not merely edit a candidate and exit, and do not launch the underlying trainer directly.

## Session initialization

The launcher sets `VIDEO_SSL_CURRENT_TASK` and `VIDEO_SSL_RUN`. Start in this run's `workspace/` directory:

```bash
pwd
printf '%s\n' "$VIDEO_SSL_CURRENT_TASK"
printf '%s\n' "$VIDEO_SSL_RUN"
```

Read these files before changing code:

1. `METHOD.md`: this protocol, copied from the selected frozen protocol file.
2. `COMMANDS.md`: the exact CLI prefix, Python interpreter, builder command, and paths for this environment.
3. `task.json`: public resolved training sources, frame policy, recipe, seeds, and limits.
4. `NOTES.md`: the current session's research history.
5. `curation/curate.py`: the current construction program.

The initial prompt contains the host's task goal and the research direction from `run_prompt.md`. Numeric constraints come from `task.json`, not from interpreting that free-form direction. If the direction conflicts with the frozen task, preserve the task and report the conflict.

Run `video-ssl plan` to inspect the declared iteration queue. In this document, `video-ssl` is shorthand for the exact CLI prefix in `COMMANDS.md`; use that prefix if the console executable is unavailable. The launcher supplies run selection automatically. Explicit `--run-dir` arguments are available for manual operation.

If a `next` or `task` result names a task other than `VIDEO_SSL_CURRENT_TASK`, stop. Do not explore another task or run to find something suitable.

### Code history

Only the current run's `workspace/curation/` is editable code. The framework starts it from a fresh copy of the repository baseline and archives accepted code automatically.

Optional Git tracking must be confined to a separate repository inside that directory. Never reset, orphan, switch branches in, or commit the host source repository. Do not import history from another run. Do not commit datasets, checkpoints, credentials, or logs. A Git commit is optional: the framework's code snapshot and aggregate SHA-256 identify the program actually submitted.

## Setup and environment verification

The host has already selected and frozen the prompt, task, machine profile, runner, optional Docker settings, protocol, source manifest, training template, and evaluation manifests. Do not edit these to make a candidate pass.

The environment has two roles:

| Environment | Purpose | Source of its executable |
|---|---|---|
| Method CLI | State transitions, submission management, reports, and Agent launch | CLI prefix in `COMMANDS.md` |
| GPU/data worker | Frame helpers, validation, InternVL-U training, generation, and scoring | `python` in the `task` response and the generated build command |

They may be different Python environments. Use the exact uv commands in `COMMANDS.md`. The CLI runs in the prepared locked project with `--frozen --no-sync --project`; workers use `--no-project --no-config --no-python-downloads` with the exact profile Python. These commands do not synchronize or replace GPU dependencies. In Docker mode, all executables are container-visible paths and the CLI environment is supplied by `UV_PROJECT_ENVIRONMENT`.

You may check the CLI help and inspect imports using the specified interpreter. Do not recreate environments, install model packages, run `uv sync`, change dependency versions, download replacement models, or alter `vendor/` during research. Missing dependencies or credentials are host blockers. Never silently switch to simulation or another model.

The host owns GPU allocation. Do not change `CUDA_VISIBLE_DEVICES`, select unallocated devices, stop another user's job, or bypass the campaign launcher. Read-only diagnostics do not authorize additional optimizer updates.

## The dataset and construction interface

### Frozen video sources

`task.json.sources` maps source IDs to:

- `video`: the original training video.
- `f0` and `final`: canonical 512 x 512 PNG endpoints.
- `frame_count`: the full video's declared frame count.
- `frames`: optional intermediate-frame index strings mapped to canonical PNG paths.
- `description`: the original task description.
- `eval_text`: the exact frozen text for the F0-to-final format.

Indices are zero-based. Endpoints come from `f0` and `final`; intermediate indices come only from `frames`. Choose only the sources and indices declared here. Distinct source identities/normalized paths do not by themselves prove distinct video content; the host performs content-level de-duplication. Host preparation establishes frame provenance and split membership. Do not decode alternative frames, relabel indices, redraw images, or add sources during a run.

### Allowed strategy dimensions

1. **Video selection:** select exactly `rows_per_trial` distinct configured videos. The current backend permits one row per video per iteration.
2. **Input and context frames:** select unique frames in chronological order, within `frame_policy.max_input_frames`.
3. **Supervision target frame:** with `target: select`, choose an available intermediate or final frame; with `target: final`, use only the final frame.
4. **Sample order:** change the manifest order and batch composition. The trainer makes one pass in that order without a second shuffle.

All input/context indices must precede the target index. Target-frame or future-frame leakage is forbidden. The first input supplies `cond_image`; remaining inputs are ordered visual context, not additional VAE conditioning images.

`mode: endpoints` is the compatibility policy allowing only `[0] -> final`. `mode: select` enables the frame decisions described above. Do not modify the policy itself.

### Submission schema

Construct a self-contained directory with `manifest.jsonl` and referenced image files. Each nonempty line is one JSON object:

| Field | Required interpretation |
|---|---|
| `sample_id` | A unique nonempty string within this submission |
| `source_video_id` | An ID from `task.json.sources`, used once in the iteration |
| `input_images` | Ordered image paths relative to the submission directory |
| `target_image` | One target-image path relative to that directory |
| `text` | The exact output of the host's frame-conditioned text helper |
| `input_frames` | Integer indices corresponding one-to-one to `input_images` |
| `target_frame` | The integer index corresponding to `target_image` |

Relative paths must remain inside the submission; symlinks and path escapes are rejected. Keep the entire directory within `max_dataset_bytes`, including otherwise unused files. The accepted code archive is limited to 5 MB. Keep image data under `workspace/dataset/`, not in `curation/`.

### Frame-aware construction example

The baseline script inserts the configured tools directory into `sys.path`. In an improvement, use the existing helpers:

```python
from frame_policy import check_frames, source_frame, training_text

# Illustrative row: use only when this source declares frames 0, 15 and 30.
input_frames = [0, 15]
target_frame = 30
check_frames(source, input_frames, target_frame, config)

input_names = []
for position, frame_index in enumerate(input_frames):
    name = f"images/{row_index}_input_{position}.png"
    source_frame(source, frame_index).save(output / name)
    input_names.append(name)

target_name = f"images/{row_index}_target.png"
source_frame(source, target_frame).save(output / target_name)
row = {
    "sample_id": str(row_index),
    "source_video_id": source_id,
    "input_images": input_names,
    "target_image": target_name,
    "text": training_text(source, input_frames, target_frame),
    "input_frames": input_frames,
    "target_frame": target_frame,
}
```

Create the output image directory first. This is an example of one row, not a replacement for the unchanged first-iteration baseline.

For `[0] -> final`, `training_text` preserves the original `eval_text`. For other combinations it identifies observed indices and the requested target index using a fixed template. Do not train against an intermediate frame while retaining text that asks for the final result. Do not freely rewrite descriptions or prompts. There must be one `<image>` placeholder per input image.

## Pre-init reference

Before activating the first iteration, run:

```bash
video-ssl preinit
```

Expected outcomes:

| Result | Action |
|---|---|
| `status: completed` | Record the returned base-model `dev.mean` as a reference |
| `status: disabled` | The host disabled preinit before initialization; continue |
| Error, interruption, or a failed/in-progress preinit | Stop for host inspection; do not replay it |

The reference evaluates the initialized base model on dev without optimizer updates. It consumes campaign time but no training attempt. Predictions and metrics are in `$VIDEO_SSL_RUN/preinit/dev/`.

A completed call returns its persisted result when repeated; do not intentionally repeat the workload. Preinit is not a trained candidate and never participates in final checkpoint selection. It does not use test.

## Running an experiment

### Activate the next iteration

```bash
video-ssl next
video-ssl task
```

`next` returns either `status: target_activated` or `status: task_done`.

A target context includes `task_id`, `iteration`, `state`, `run_dir`, `spec`, `task_file`, `default_submission_path`, `builder`, `tools`, `python`, and `prior_feedback`. `run_dir` is this iteration's artifact directory; `VIDEO_SSL_RUN` is the whole method-run directory.

- On `target_activated`, verify the task identity and inspect `state`.
- On `task_done`, finish your notes and exit successfully. Do not start another run, resume another Agent, or extend iterations; the host owns lifecycle changes.
- On an error, inspect the reason and apply the failure rules below.

Repeated `next` on an unfinished iteration returns the same target. It does not skip it, clear its session, or restart training. Do not mistake `status: target_activated` for a new round: compare `iteration` and `state`.

A restored Agent session must continue the saved stage:

| Saved state | Next action |
|---|---|
| active | Continue the existing strategy window; build, audit and submit |
| submitted | Use the accepted submission; call finetune, then eval |
| trained | Call eval using the saved checkpoint |
| completed | Read the result and call next |
| failed improvement | Record the failure and call next |
| submitting / training / evaluating | Stop for host inspection |

Do not rebuild or resubmit a submitted/trained candidate, reset a strategy timer,
or redo a completed workload. Each round has session_state.json and sequenced
events.jsonl; the framework checks consistency before continuing.

### First iteration: establish the baseline

Build with the freshly copied, unchanged `curation/curate.py`. It uses the frozen seed to select and order distinct videos, and supervises F0-to-final. The framework checks both the baseline script hash and the actual ordered source IDs/frame indices.

Do not make a speculative improvement before measuring this baseline. If it fails training or evaluation, stop for the host; subsequent iterations cannot proceed past a failed baseline.

### Later iterations: read feedback and propose a change

For every previous iteration in this same run, inspect:

| Artifact | Question it answers |
|---|---|
| `code/curate.py` and helper files | What program produced the result? |
| `code/hypothesis.md` | What was predicted before the change? |
| `curated/final_submission/manifest.jsonl` | Which videos, input frames, targets and order were actually trained? |
| `changes.diff` | What code changed relative to this round's base_target? |
| `selection_diff.json` | Which videos, frame assignments and positions changed? |
| `dev/metrics.json` | What was the measured aggregate result? |
| `dev/per_item.jsonl` | Which examples and components succeeded or failed? |
| `dev/images/<id>.png` | What did the model actually generate? |
| The public `workspace/dev_feedback.jsonl` | Which frozen dev input, target and prompt correspond to that ID? |
| `train/loss.jsonl`, `train.log`, `dev.log` | Did the computation behave as expected? |

Failed rounds may have incomplete artifacts. Do not infer missing scores as zeros. Code and selection differences compare this candidate with base_target, the accepted strategy actually restored before this round. The immediately preceding discarded candidate may not be that starting point.

Before modifying a later candidate, write a new `curation/hypothesis.md`:

```markdown
# Hypothesis

## Evidence
Prior iteration IDs, aggregate and per-item dev observations, and relevant images.

## Proposed explanation
One testable explanation, distinguished from measured facts.

## Construction changes
Video selection:
Input/context frames:
Supervision target frames:
Sample order:
State explicitly which dimensions change and which remain fixed.

## Expected outcome and refutation
Expected aggregate/component behavior and the result that would reject this idea.
```

The framework removes the workspace hypothesis when it activates the next iteration, so a stale file cannot silently satisfy this requirement. Preserve history in `NOTES.md` and prior immutable code snapshots.

### Stage 1: construct, audit and submit

Use the exact build command from `COMMANDS.md`. Its general form is:

```bash
uv run --no-project --no-config --no-python-downloads \
  --python <GPU_PYTHON> -- <GPU_PYTHON> <WORKSPACE>/curation/curate.py \
  --config <WORKSPACE>/task.json \
  --tools <BACKEND_DIRECTORY> \
  --output <WORKSPACE>/dataset
```

Angle-bracket paths are placeholders. Use the real paths supplied by the framework, not copied example paths. The starter refuses to overwrite its image directory. When rebuilding, remove only the current run's generated `workspace/dataset/`; never remove an accepted submission or another run's data.

Check the strategy timer before expensive construction and before submission:

```bash
video-ssl time-budget
video-ssl audit --path <WORKSPACE>/dataset
video-ssl submit --path <WORKSPACE>/dataset
```

`audit` succeeds with `status: valid` and a sample count. It checks the source whitelist, distinct videos, frozen frame and image provenance, exact text, temporal relationships, and data limits. It does not prove semantic uniqueness or historical non-exposure of the underlying videos.

`submit` revalidates, archives data and code, checks the archive, and returns `status: submitted` with `submission_path`. The state becomes `submitted`. Each iteration accepts exactly one candidate. Repeated successful submit returns that accepted candidate; it does not replace it with another `--path`.

Validation failure before acceptance leaves the strategy stage available for correction while time remains. Once the state becomes `submitting`, an interrupted or failed archive is not repaired by deleting directories or repeating submit.

### Stage 2: train

```bash
video-ssl finetune
```

Run this as a blocking foreground command. Do not use `&`, `nohup`, background tools, another terminal for an extra training job, or direct `torchrun`. Wait for the command to return before issuing another experiment tool call.

Success is `status: completed` with a `checkpoint` object containing its path and SHA-256. The iteration state becomes `trained`; this is not yet a measured improvement.

Inspect the saved receipts as needed:

- `train_result.json` and `checkpoint.json`.
- `train/last.pt` and `train/checkpoint.json`.
- `train/loss.jsonl`, `train/cost.json`, and `train/resolved_train.yaml`.
- `rows.json`, `training_contract.json`, and `train.log`.

A normal process exit alone is insufficient; use the framework receipt. Repeating a successfully completed training command returns the cached receipt after integrity checks, without another optimizer run. Do not replay failed or uncertain training.

### Stage 3: development evaluation

```bash
video-ssl eval
```

Use the same blocking foreground discipline. Evaluation starts a fresh process, loads the common base and saved trainable parameters, generates the frozen F0-to-final dev cases, and applies the fixed scorer.

Success is `status: completed` with `dev.mean`, `dev.n`, and `dev.n_invalid`. The state becomes `completed`. Check `dev/metrics.json`, `dev/per_item.jsonl`, predictions, `dev/reload.json`, and the checkpoint hash before interpreting the score.

Update `NOTES.md` with the measured result and interpretation, then call `next`. Before activating another round, the framework restores the highest-scoring accepted code, preferring the earlier candidate on ties. A discarded candidate never silently becomes the next starting point. Inspect `base_target` in the next task response. Working copies are preserved in workspace_history; official code/data/checkpoints and NOTES.md are not rolled back. Do not stop merely because the first improvement failed to beat the baseline; continue within the declared iterations and failure rules.

## Scoring and feedback analysis

The current task uses `analogy_grid_v1_scale`. Its per-example score is:

```text
0.35 * E_cell + 0.40 * F_cell + 0.25 * F_transform
```

These are the scorer's returned sub-scores, after its context and image-quality gates. Do not apply the gates a second time. `E_cell` and `F_cell` measure the relevant generated cells; `F_transform` measures the requested transformation. Task-specific matching and thresholds belong to the frozen scorer, not to your candidate.

`dev.mean` is the arithmetic mean over all declared dev items. Missing/unreadable, black or flat predictions are invalid and count as zero under the implemented rule. A model-loading, training, generation-process or scoring exception is a failed operation, not a valid zero-score experiment. Do not mask it with synthetic metrics.

Each per-item record includes `id`, `score`, `valid`, `sub` and `categories`. Useful categories include `E_missing`, `F_missing`, wrong colors, wrong transformations, damaged context and extra objects. Analyze both aggregate changes and components: improving one cell while damaging context can reduce the overall result.

For each completed improvement:

1. Compare dev.mean with the baseline and the previous relevant strategy.
2. Inspect invalid counts and the worst-scoring cases.
3. Summarize component means and error-category counts where informative.
4. Inspect representative generated images rather than inferring behavior from a scalar alone.
5. State whether the predeclared criterion was met, and which interpretation remains uncertain.

Do not tune the scorer, filter out inconvenient dev cases, select a different checkpoint from intermediate training steps, or claim statistical significance from this single-seed search.

## Experimentation rules

### What you can change

- Python construction and analysis helpers under this run's `workspace/curation/`.
- Video selection, allowed input/context frames, target frame and manifest order.
- The current round's hypothesis and ongoing research notes.
- Generated candidate data under `workspace/dataset/` before acceptance.

Prefer a simple reproducible strategy when its measured behavior is comparable to a more complex one. If multiple construction dimensions change together, report that their separate contributions cannot be identified from that experiment.

### What remains fixed

Do not edit the host repository, `vendor/`, model packages, task/profile/runner/Docker/protocol sources, `task.json`, frozen configuration snapshots, source videos, canonical frame banks, pretrained weights, accepted code/data, integrity records, framework state, metrics or the shared campaign ledger.

The current backend requires exact row count, unique source videos, an ordered single pass, last-step checkpointing and restart from the same base. Training-frame decisions do not change the dev/test task. Follow the resolved values rather than assuming the example defaults are universal.

## Logging results

The framework maintains `results.tsv`; do not append, rewrite or fabricate its rows yourself. Columns are:

```text
iteration    dev_mean    status    code_sha256    description
```

They are tab-separated. Status meanings:

- `reference`: iteration 0, the optional untrained base dev reference; never a selected trained candidate.
- `keep`: a completed candidate that strictly exceeds all earlier completed candidates, or the first completed baseline.
- `discard`: a completed candidate that does not exceed that running best, including exact ties.
- `crash`: a failed iteration; the missing score stays blank rather than becoming a measured zero.

`code_sha256` identifies the accepted code tree, not a Git commit. The description is a compact rendering of the saved hypothesis. Full code and hypothesis are preserved under each iteration's `code/`.

Maintain `NOTES.md` separately. For each round record: ID, evidence, hypothesis, changed dimensions, expected result, submission identity, measured dev result, interpretation, limitations and the next question. Keep predictions separate from measurements. Write hypotheses, notes and explanations in English. Preserve frozen source descriptions and eval_text verbatim, even if their original language differs. Notes must not overwrite the frozen historical artifacts.

## The experiment loop

```text
Read this workspace AGENTS.md, METHOD.md, COMMANDS.md, task.json and existing NOTES.md. Follow the workspace reading scope; repository history/setup documents and other protocol variants are not research inputs.
Inspect the declared plan.
Run preinit once, or observe its configured disabled state.
Repeat:
    next
    if task_done: finish notes and exit
    verify task identity and inspect the saved stage
    task
    if submitted: finetune, eval, then continue
    if trained: eval, then continue
    if completed or failed improvement: continue
    if uncertain in-flight stage: stop for host inspection
    if first iteration: preserve the baseline builder
    otherwise: read prior artifacts, write a fresh hypothesis, edit construction code
    build the current dataset
    time-budget, audit, submit
    finetune and wait
    eval and wait
    read metrics and predictions; record observations
Stop for host inspection on a failed baseline, missing dependency,
uncertain running operation, integrity violation or exhausted campaign.
```

This is an operating sequence, not a shell loop that blindly ignores errors. Check every command's result and exit code before deciding the next action.

## Monitoring, time limits and failures

The strategy clock starts when `next` activates a round. `time-budget` reports elapsed/remaining seconds and `expired`; expiry gates successful submission. Do not treat that strategy timer as a post-submission training timeout.

Training, dev and test have their own task timeouts. The entire Agent session may have a runner timeout. The campaign deadline and reserved reporting/test time additionally bound work; satisfying one timer does not override the others.

During a blocking training or evaluation call, wait. Avoid tight polling of `ps`, `nvidia-smi` or log files. For long construction work, log progress, elapsed time and an ETA. Use the pre-exported frame bank instead of repeatedly decoding videos. Do not kill a legitimate trainer just because a log has not changed recently.

| Observed situation | Required action |
|---|---|
| Build/audit rejects a candidate while state is `active` | Correct the candidate within remaining strategy time |
| Construction cannot finish | `video-ssl fail --reason "specific reason"`; inspect the saved result |
| Improvement is definitely `failed`, no active/uncertain process remains | Continue with `next` within the declared plan |
| Baseline is `failed` | Stop for host inspection |
| State is `submitting`, `training` or `evaluating` after interruption | Stop; do not clear locks, change state or retry the operation |
| State is `submitted` | The only accepted candidate is already fixed; proceed to finetune |
| State is `trained` | Do not retrain; proceed to dev evaluation |
| State is `completed` | Read feedback and continue with next |
| Missing dependency, authentication, GPU allocation or invalid frozen resource | Report the blocker; do not consume remaining rounds repeating it |
| Integrity check fails | Stop and report the changed resource |

A repeated successful stage is idempotent. A failed/in-progress stage is not permission for a new optimizer attempt. Explicit `fail` cannot discard an active operation or a completed result. Never edit the ledger or create another run to evade limits.

## Data isolation and final test

Only inspect the assigned task's configured training resources and the current run's permitted historical train/dev artifacts. Do not scan other runs, external Git history, unrelated host directories or credentials.

The public workspace/dev_feedback.jsonl lists the dev inputs, targets and prompts you may inspect when comparing predictions with ground truth. It is not a permitted source of training rows; training remains restricted to task.json.sources.

Do not open test manifests, test images, test labels, test metrics or `final/`. Do not inspect the private `config/` snapshots or source machine profiles to recover hidden evaluation material; use public `task.json` and the task CLI. Never invoke `finalize`, unset `VIDEO_SSL_AGENT` to bypass its check, or run test generation directly.

After the search ends, the host selects the completed trained candidate with highest dev.mean. Exact ties prefer the baseline, then the earliest iteration. The host evaluates that already saved checkpoint once on test, without retraining or changing seeds. The preinit reference is excluded.

Native execution inherits user permissions. Docker constrains access to the configured mounts; neither mode turns these behavioral rules into a defense against malicious same-permission code. Honor the restrictions even when a path is technically accessible.

## Finish

Continue autonomously until `next` returns `task_done`, subject to the explicit stop conditions above. Do not ask for confirmation between already authorized iterations, exhaust extra budget beyond the plan, launch another experiment or fabricate a selection. Finish `NOTES.md` with the best dev candidate, measured comparisons, rejected hypotheses and remaining uncertainties. Then exit; final test belongs to the host.

## Optional research guidance

The host may select instruction or skill instead of method in run_prompt.protocol.
The framework combines this base protocol with the selected guidance appendix and
freezes both files. Only the selected composed METHOD.md is your operating protocol.
If it includes the skill appendix, you may read the repository's read-only skills/
library as research reference. That permission does not expand the training source
whitelist, authorize environment changes or permit other-suite access. Reading quotas
are protocol requirements, not measurements verified by the framework.
