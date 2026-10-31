# Upstream-Style Suite Lifecycle (Option B)

The comparison source is curation-bench commit
24eea1526492c00cee421f5db0793789e00aabb2. This is a task-backend adaptation of its
suite/session organization, not merely a rename of the earlier MethodRun wrapper.

## Reused interfaces and task-specific boundary

| Upstream interface | Current implementation |
|---|---|
| Target / ordered task-iteration queue | core/target.py, preserving queue identity/serialization |
| SuiteController.from_prompt | Create or load an explicit/active suite |
| SuiteController.from_state_file | Load persisted single-task or multi-task state |
| get_run_plan / begin_next_target | Summarize progress / activate a target |
| get_active_session / mark_target_completed | Restore per-target tools / persist successful dev completion |
| Session save_state / load_state / dispatch | Restore session_state.json, sequence events, dispatch stage tools |
| Task-specific submission/train/eval | tools modules delegate to backends/video_ssl.py |
| One container/Agent per task | One continuous Codex session per task, native or Docker |
| Keep/discard code search | Restore best accepted code before activating another round |

The source port retains upstream names, queue order, factory/load flow and Session
persistence, while adding result integrity and campaign constraints. It does not
reuse LLaVA/VQA-specific dataclasses, vendor environments or the original model
registry. history/upstream-readme.md and backend PROVENANCE.md retain attribution.

## Configuration and queue

run_prompt retains tasks, iterations, profile and free research direction. The
existing runner, protocol and campaign references remain. Example:

```yaml
tasks: [task_a, task_b]
iterations: 3
profile: server_a
runner: configs/runner.yaml
protocol: method
campaign: campaigns/kickoff/campaign.json
```

Both task YAMLs must exist and be supported by the video backend. Each task gets
three rounds, including its baseline; six attempts must fit the shared campaign.
Different task IDs do not create support for a different model/scorer automatically.

```text
SUITE/
  config/plan.json
  suite_state.json
  suite_result.json
  tasks/TASK_A/
    config/plan.json
    suite_state.json
    workspace/curation/
    TASK_A_iter1/session_state.json
    TASK_A_iter1/events.jsonl
    output/session_0001/
    final/
  tasks/TASK_B/...
```

A single-task suite keeps its workspace and iteration directories directly under
SUITE, preserving the existing video layout. Multi-task roots aggregate child
controllers rather than sharing mutable task context across Agent processes. Tasks
are executed in declaration order. Task A's code is never Task B's initial code.
All declared task searches finish before automatic final tests run.

## Initialize, run and load

```bash
uv run --locked video-ssl check --prompt run_prompt.md
uv run --locked video-ssl run --prompt run_prompt.md --root runs/exp-001
uv run --locked video-ssl plan --suite-dir runs/exp-001
```

--profile overrides the machine profile on check/init/run. --root is optional: the
factory can load .video_ssl_suite or create a timestamped runs/suite_* directory.
An explicitly selected/active completed suite is reused without new training;
--new-suite requests a fresh directory. Existing inputs and their paths must match
the saved declaration. Resource changes require a new suite, not silent re-resolution.
--run-dir remains an alias for --suite-dir on existing-suite commands.

## Stage-aware resume

```bash
uv run --locked video-ssl resume --suite-dir runs/exp-001 --inspect
uv run --locked video-ssl resume --suite-dir runs/exp-001
```

Inspection is read-only and does not start an Agent/GPU job. Actual resume starts a
new Agent process with the same workspace, notes, code snapshots and saved stages:
active continues construction; submitted proceeds to training; trained proceeds to
dev; completed reuses results and advances. Strategy time is not reset. Each Agent
process has a separate output/session_NNNN log and saved initial prompt.

A live/unknown Agent owner, operation lock, running/interrupted campaign attempt,
uncertain submitting/training/evaluating stage, failed baseline, changed frozen
input, inconsistent Session receipt, or failed/in-progress final test blocks resume.
The host must inspect the cause. No automatic deletion of lock files, resetting of
Session state, replacement submission or extra optimizer attempt is performed.
Multi-task resume skips completed task searches and continues unfinished tasks.

## Extend a task queue

```bash
uv run --locked video-ssl extend --suite-dir runs/exp-001 --iterations 5
uv run --locked video-ssl resume --suite-dir runs/exp-001
```

N is the new total, not an increment. It must increase the count, fit max_trials,
remaining training attempts and the campaign deadline/reserve. Existing accepted
submissions, checkpoints and reference evaluations are reused. A frozen receipt is
written under config/extensions; effective iterations is updated in the saved plan
and public task. The original prompt and historical task snapshots stay unchanged.
Do not edit run_prompt to extend a frozen experiment.

For a multi-task suite select runs/exp-001/tasks/TASK_A. Sibling tasks' pending
attempts also count against the shared budget. Extension is forbidden after any
task has begun final test. Agents cannot invoke resume/extend or create new suites.

## Strategy keep/discard

Before a new round, next selects the highest completed dev score, preferring the
earlier candidate on ties, verifies its accepted code/data/checkpoint, and restores
its code into workspace/curation. base_target identifies this starting point. Losing
candidates keep their immutable code/data/metrics and remain readable as feedback.
Code and selection differences compare the candidate against base_target, not the
immediately preceding discarded candidate. The previous worktree is preserved under
workspace_history; NOTES.md and official
results are never reset. A fresh hypothesis is required for each improvement.

This implements upstream's keep-or-revert search behavior without Git resets of the
host repository. Every trained candidate still restarts from the frozen model base.
An explicit fail records a failed improvement; unknown in-flight work cannot be
abandoned by next or fail. A failed baseline stops the task.

## Optional research guidance

protocol: method is the default. instruction adds a prescriptive research artifact;
skill adds the upstream-style card-reading and per-hypothesis grounding requirements.
The selected appendix and base are frozen and composed into workspace/METHOD.md.
Only the skill selection authorizes reading the optional skills library. All modes
share exactly the same data, trainer, scorer and budget constraints. Compliance with
card reading is a protocol requirement, not a framework-verified measurement.

## Preserved controls and validation limits

The campaign ledger remains shared and bounded. A successful submission is sealed;
checkpoint hashes and Session receipts are checked before reuse. Final selection is
per task on dev; the host tests the saved selected checkpoint once without retraining.
In Docker mode host-driven final workers use the task image, and GPU ordinals are
mapped to the container's visible devices without changing the host ledger.

CPU tests exercise real configuration/resolver/CLI operations, mocked GPU lifecycle,
resume, code restoration, task isolation, extension budgets and Docker command
construction. They do not establish GPU model quality or real Docker compatibility.
Pre-refactor frozen experiments require their original checkout; this lifecycle is
not an in-place migration of historical artifacts.
