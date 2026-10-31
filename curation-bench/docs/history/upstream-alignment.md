# Alignment with the Original Curation-Bench Design

This document maps the original configuration layout and operating protocol to the video SSL method. It records implemented behavior and deliberate differences, rather than implying that renaming files makes all original model/evaluator/Agent options available.

The comparison baseline is curation-bench commit `24eea1526492c00cee421f5db0793789e00aabb2`, the revision originally cloned for this work. It does not claim to describe every later change to upstream main. Source: [original protocol at that revision](https://github.com/feiyang-k/curation-bench/blob/24eea1526492c00cee421f5db0793789e00aabb2/BENCHMARK.md).

## Configuration roles

| Original file | Method counterpart | Alignment and explicit differences |
|---|---|---|
| run_prompt.md | run_prompt.md | Retains structured tasks/iterations/profile plus free-form research direction. Adds explicit runner/protocol/campaign references and optional tasks_dir. Supports a task-major suite queue with an isolated workspace/Agent per task. |
| profiles/<machine>.yaml | profiles/<machine>.yaml | Retains logical dataset/model IDs mapped to physical resources, and machine environment. Adds worker Python, independent eval IDs, preparation/scorer resources and explicit GPU allocation. Agent settings do not live here. |
| src/benchmark/tasks/<task>.yaml | tasks/<task>.yaml | Retains objective, dataset/model identity, data budget and train/eval settings. Adds frame policies, supervision-target decisions, seeds, explicit dev/test separation and a validated backend contract. |
| configs/docker.yaml | configs/runner.yaml plus optional configs/docker.yaml | Separates the original file's Agent settings from its container settings. Native and Docker share Agent settings; native does not load Docker configuration. |
| protocols/<variant>.md | protocols/method.md, or another explicitly selected file | Restores a full research operating manual. Default method plus optional instruction/skill appendices. The skill selection carries explicit card-reading requirements; variants do not alter the training contract. |
| Vendor training/evaluation projects | vendor/video_ssl_backend plus external InternVL-U resources | Replaces the instruction-tuning/standard VQA evaluator path with video/frame supervision, image generation and the frozen analogy-grid scorer. |

The low-level InternVL-U template, source manifest and shared campaign ledger are supporting resources. They do not replace these configuration responsibilities or create another override layer.

## Run declaration and resource lookup

Both designs separate the experiment declaration from physical machine locations. In the method, profile/runner/protocol references may be explicit paths; recognized names are resolved in the corresponding repository directory. Resource IDs in the task must match the profile maps exactly.

As upstream, initial iterations applies uniformly across the task-major queue and includes the trained baseline. Each supported task has an isolated workspace, persisted Session stages and a continuous Agent session; the suite aggregates task queues under one campaign.

Research text is sent to the Agent as a direction, not parsed as a recipe. Numeric constraints are resolved from task YAML and enforced by the backend. Unknown fields, missing resources and contradictory training-template values are rejected.

## Protocol section-by-section mapping

| Original section | Method section / behavior |
|---|---|
| Introduction | Defines a video SSL data-construction method and Agent/framework responsibilities; no agent-ranking benchmark framing. |
| Session initialization | Checks VIDEO_SSL_CURRENT_TASK/RUN and reads the selected METHOD, exact COMMANDS, public task, notes and builder. Fresh accepted snapshots provide isolation from previous runs. |
| Git history reset | Does not orphan/reset the host repository. Optional Git lives only in the run's editable curation directory; framework code-tree hashes and snapshots are authoritative. |
| Setup | Host resolves/freezes configuration and prepares dependencies. Agent verifies the supplied tools and inspects the queue before research. No repeated permission requests during authorized rounds. |
| The Dataset | Describes original videos, canonical indexed frames, descriptions/text, the manifest schema, multiple ordered inputs and a single selected target frame. |
| Venv setup | Distinguishes method CLI and GPU/data worker environments. Host prepares them; Agent does not delete environments or install new model dependencies during a run. |
| Pre-init baseline | Preserves untrained-model dev evaluation as a reference. Enabled by default in the task, explicitly disableable before initialization, never a final candidate. Its ledger row is labeled reference. |
| Activate the next iteration | Preserves next/task context and task-identity checks. Repeated next returns unfinished work rather than implicitly abandoning or reinitializing it. |
| Stage 1: Curate | Uses curate.py, declared sources and a self-contained manifest/image submission. Adds explicit input/context and supervision-target decisions, hypothesis requirements and frame/pixel/text validation. |
| Audit/submit | Preserves the validation/submission interface. The method seals one accepted submission per iteration; upstream allows replacement submissions. Audit covers provenance and construction rules, not the original question/answer overlap detector. |
| Stage 2: Train | Preserves foreground blocking training through a framework command. The receipt identifies the saved trainable checkpoint and hash rather than an instruction-tuning model directory. |
| Stage 3: Evaluate | Preserves independent evaluation and completion on success. Uses a newly loaded generation checkpoint and frozen dev, rather than the original standard VQA suite. |
| Scoring | Replaces eight-benchmark normalization with exact analogy-grid weights, gating, invalid-image policy, per-item components/categories and interpretation rules. |
| Experimentation rules | Preserves an editable construction directory and fixed trainer/evaluator. Expands strategy from subset selection to input/context frames, supervision targets and ordering. |
| Logging results | Framework owns TSV, code snapshots, metrics and differences; Agent owns research notes and hypotheses. Uses code-tree SHA-256, not an assumed Git commit. Failed scores stay blank. |
| The experiment loop | Preserves a continuous session through the declared queue. Explicitly separates construction, accepted submission, training and dev; terminates at task_done or a defined blocker. |
| Monitoring | Preserves blocking calls, sensible log checks and progress for construction. Distinguishes strategy, operation, session and shared-deadline limits. |
| Data isolation | Preserves current-run/task boundaries and prohibits other-run strategy/history access. Adds a strict final-test prohibition and public dev-only feedback paths. |

The method protocol intentionally includes commands, response states, artifact paths, a construction example, a hypothesis template, a complete loop and a failure table. These details are not delegated solely to an operator README the Agent may never read.

## Evaluation, reference and final test

Upstream evaluates each curated subset on its configured benchmark suite. This method uses a fixed development objective for construction search and reserves a separate test split for the host's selected checkpoint.

The differences are methodological requirements:

- An intermediate-frame training target must not change the F0-to-final development objective.
- The untrained base reference is descriptive; it is not selected instead of a trained checkpoint.
- The highest completed dev.mean selects the saved last-step checkpoint. Exact ties favor the baseline and earlier iterations.
- Final test runs once after search, without retraining or using test feedback for another strategy.

The protocol explains what feedback is readable. The workspace's public dev_feedback.jsonl exposes dev inputs/targets for per-case comparison. Private snapshots can contain test configuration or labels and are not research-Agent input.

## Submission and failure semantics

| Situation | Original design | Method behavior |
|---|---|---|
| Dataset representation | HuggingFace Arrow image/conversation records | JSONL frame-indexed supervision with canonical images |
| Data budget | Protocol says at most target_rows, but the actual validator requires exact equality | Exact rows_per_trial as well; adds unique videos and one row per source under the current backend |
| First trained candidate | Unchanged random selection baseline | Unchanged seeded video selection/order with F0-to-final; script and manifest decisions are checked |
| Advancing unfinished work | next can mark an active target failed; reconstructed sessions can be reset | next is idempotent until explicit terminal state; uncertain operations require inspection |
| Repeated successful stage | Some persisted training results are reused | Accepted submission, trained checkpoint and dev result are reused after integrity checks |
| Failed improvement | Later iterations may continue | Continues only when failure is explicit and no uncertain in-flight operation is being discarded |
| Failed baseline | Original loop treats iterations independently | Host inspection is required before any improvements |
| Interrupted training/test | Upstream session rebuilding can restart flow | No automatic optimizer or final-test replay; preserve ledger and operation state |

The host resume command validates the suite/campaign and continues saved stages in a new Agent process. A submitted candidate is reused, a trained checkpoint proceeds to dev, and completed operations are cached. Session files are not reset. The explicit extend command appends rounds within remaining task/campaign limits before final test.

## Environment and Docker differences

Upstream invokes its project CLI with `uv run`; this method now uses the same project workflow with committed `uv.lock` and host-side `uv sync --locked`. Agent commands pin the repository with `--project` and use `--frozen --no-sync` after initialization. Frame construction and backend workers also run through uv, using `--no-project --no-config --no-python-downloads` and the exact profile Python. The existing GPU environment remains independently prepared; the lightweight CLI dependency lock does not manage Torch/CUDA or the external training library. The worker's distributed child uses that same GPU interpreter. `uv tool` is not part of this pipeline.

The Dockerfile uses a pinned uv bootstrap and builds a separate CLI virtual environment with `uv sync --locked`. `UV_PROJECT_ENVIRONMENT` points to that prepared image environment, allowing Agent commands to use the read-only mounted repository without creating a virtual environment inside it. Host CLI uv paths are never passed as container executable paths. Dependency files and the lock are frozen in each run plan.

The original Docker configuration also stores Agent model/effort choices. The method moves those settings to runner.yaml so native execution retains the same clear Agent configuration.

Docker remains optional. When enabled, physical resource paths remain consistent through mounts, while the GPU Python/library may be supplied by the image. A host-side light resolver composes configuration without loading a GPU model. Credentials are passed as explicit files or environment names, not baked into images.

The configuration reference documents actual environment and mount behavior. Native inherits user permissions; Docker limits access only to the chosen mounts. Neither arrangement makes the protocol a malicious-user security boundary or guarantees that mounted test paths are technically unreadable.

## What is not carried over as an active feature

The following upstream assets are not implied capabilities of this method:

- Standard LLaVA/Qwen/SmolVLM instruction-tuning task families or their eight evaluation benchmarks.
- Arrow/VLMEvalKit vendor environments, VQA judge APIs and benchmark-score normalization.
- Blind reuse of original task-specific protocol wording. Instruction and skill research appendices are adapted to the video contract; card reading is not mechanically verified.
- Automatic support for every original Agent launcher; the current configured command format is Codex.
- Unlimited/null strategy time, unsupported task backends, or unsafe automatic reset of interrupted optimizer work.

The unused instruction-tuning backends, VQA code, legacy Agent images and compatibility entry points have been retired from the active tree. Recoverable copies are held in the sibling ../curation-bench-archive/legacy-curation-bench directory. The original skills library is retained as optional research reference, with quotas only when the skill appendix is selected. The public entry point follows the documented video SSL path. Unsupported settings are not papered over by prompt wording.

## Acceptance boundary

Configuration and protocol documentation are checked against the implemented parser, stage receipts and artifact layout. CPU tests exercise state persistence, one-session orchestration, actual resolve/build/audit subprocesses, frame contracts, integrity, reference logging and configured scorer loading.

These checks do not replace a real GPU acceptance run. Altered training recipes, multi-image behavior, external-library compatibility, model quality, Docker GPU mapping and credentials remain server-validation work under a fresh authorized budget.
## Detailed upstream configuration-field mapping

The following maps the original fields to their supported method equivalents. An unmapped field is not silently accepted by the new schema.

| Original field/group | Method equivalent or disposition |
|---|---|
| run_prompt.tasks / iterations / profile / Markdown body | Same conceptual roles; task-major queue, baseline included in iterations |
| Profile dataset_path_map / model_path_map | Retained logical-ID maps; the training dataset value is a directory containing the declared source manifest |
| Profile eval_data_dir | Replaced by independent eval_path_map IDs for dev/test JSONL; image paths resolve relative to each file |
| Profile CUDA_VISIBLE_DEVICES / BENCHMARK_TRAIN_GPUS / BENCHMARK_EVAL_GPUS | Replaced by explicit profile/campaign gpu_allocation, with a serial shared GPU process launcher |
| Profile judge endpoint/key environment | Not required by the rule scorer; ordinary machine environment remains configurable, secrets stay out of static files |
| task.task_id / goal / target_rows | task_id / goal / contract.rows_per_trial, with an exact one-pass budget |
| task.model_key / model_name_or_path | model_key resolves through the profile; no task-local physical snapshot path |
| task.training_stage | Not an independent switch in this backend; finetune invokes the declared video-supervision training operation |
| task.dataset.dataset_id / local_path / split / image_column / text_column | dataset_id plus sources_manifest; explicit source IDs/frame metadata replace Arrow column and split selectors |
| task.strategy_timeout_seconds | Retained as a positive per-round submission interval; null/unlimited is unsupported |
| task.default_submission_dir | Framework-managed iteration path curated/final_submission; not user-overridable |
| task.target_evals / eval_config.benchmarks | Fixed task-specific dev/test resource IDs, not a list of VQA benchmarks |
| training_config.max_steps | contract.training.max_steps and the resolved low-level template value |
| training_config.num_processes | Derived from the confirmed GPU allocation; global batch must be divisible by that count |
| training_config.save_steps / logging_steps | No arbitrary intermediate-checkpoint search; the backend saves last-step parameters and records each optimizer step |
| training_config.timeout_seconds | contract.training.timeout_seconds, bounded by the campaign reserve |
| eval_config.timeout_seconds | Independent evaluation.dev_timeout_seconds and test_timeout_seconds |
| eval_config.gpu_id | Shared confirmed allocation, rather than an independent evaluator GPU chosen by the Agent |
| eval_config.use_vllm / mode / judge_model / judge_api_base / api_nproc | Replaced by the actual generation/scoring contract; the current backend has no VQA judge or vLLM evaluator |
| Docker env | Container-specific docker.env, separate from machine profile.env; duplicate/control-variable settings are rejected |
| Docker mounts: src/dst/mode | Read-only resource mounts at consistent absolute paths, writable framework run/ledger paths, optional extras; arbitrary resource destination remapping is not implemented |
| Docker agents.codex.model / effort | runner.agent.model / reasoning_effort, shared with native execution |
| Docker agents.claude.max_turns / OpenHands.native_tools | No current counterpart; those Agent integrations are not claimed by the Codex method runner |
| Protocol plain/instruction/skill selection | Explicit method/instruction/skill selection or custom path; appendices share the same base protocol |

The separation changes file ownership, not just names: Agent settings must not be placed back in a machine profile or Docker file. Likewise, task sample counts and frame limits must not be hidden in the run-direction text. Preflight rejects cross-layer fields and mismatched low-level numeric values.

## Option B lifecycle reuse

The port reuses upstream's Target record/serialization, target-major queue, factory/load
interfaces, per-command Session persistence and sequenced tool events. Training and
evaluation now delegate to the concrete VideoSSLBackend adapter rather than the
original model registry/vendor commands. See lifecycle.md for the interface mapping.

Intentional differences remain: next does not silently fail an unfinished target;
resume never clears an uncertain Session; extension is explicit rather than editing
the frozen declaration; submitted candidates are sealed; the host owns final test;
code rollback restores hashed snapshots instead of resetting the source repository.
Native and Docker execution share the same Agent configuration, and Codex is the
current launcher. These controls preserve budget and result integrity while retaining
the original experiment organization.
