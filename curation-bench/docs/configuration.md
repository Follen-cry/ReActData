# Configuration Reference

This reference describes the configuration actually consumed by `video-ssl`. Example numbers describe the shipped task, not an unconditional rule for every future task. New suites use the layered schema below; backend.input.json is generated from those inputs and is not another hand-maintained configuration layer.

## Resolution and authority

The resolver follows this graph:

```text
run_prompt.md
  -> tasks/<task_id>.yaml
  -> profiles/<machine>.yaml
  -> configs/runner.yaml
       -> optional configs/docker.yaml, only for Docker mode
  -> protocols/<protocol>.md
  -> shared campaign ledger

task resource IDs + profile paths
  -> source JSONL + dev/test JSONL + base model + external libraries
  -> low-level training template
  -> frozen generated configuration under runs/<run-id>/config/
```

There is no arbitrary last-file-wins merge. Unknown fields and unsupported combinations fail preflight. The task owns research constraints; the profile owns physical resources; runner owns Agent execution; Docker owns container settings; protocol owns research procedure.

Generated `backend.input.json`, `backend.json`, resolved training YAML and resolved evaluation JSONL are implementation artifacts. Do not maintain a second editable copy of them as another source of truth.

## Run declaration: run_prompt.md

The file starts with YAML frontmatter between `---` delimiters. Its remaining Markdown body is research-direction text supplied to the Agent.

| Field | Required/default | Meaning |
|---|---|---|
| tasks | Required list | Currently exactly one named task ID per run |
| iterations | Default 1 | Positive integer including the trained baseline; must fit the task maximum and shared remaining attempts |
| profile | Required by the method | Machine-profile path or name |
| runner | Defaults to runner | Runner path or name under configs/ |
| protocol | Defaults to method | Markdown path or name under protocols/ |
| campaign | Required by the method | Existing shared ledger path relative to this prompt |
| tasks_dir | Optional | Task-definition directory relative to this prompt; otherwise repository tasks/ |

Example:

```yaml
---
tasks: [shape_color_video_ssl]
iterations: 3
profile: server_a
runner: configs/runner.yaml
protocol: method
campaign: campaigns/kickoff_001/campaign.json
---

Study whether earlier supervision targets improve the fixed final-state task.
```

A named task resolves to `<tasks_dir>/<task_id>.yaml`; the file's task_id must match. Task IDs are names containing letters, digits, underscores or hyphens, not path expressions.

Named profile/runner/protocol references first check the specified path relative to the declaring prompt, then their corresponding repository directory and extension. Campaign and tasks_dir are direct relative/absolute references, not registry-name lookups. No CLI override silently changes a frozen layer after initialization.

## Machine profile: profiles/<machine>.yaml

| Field | Type | Responsibility |
|---|---|---|
| python | Path string | GPU/data-worker interpreter; absolute after resolution |
| dataset_path_map | ID -> directory | Training pools; task.sources_manifest is relative to the selected directory |
| model_path_map | ID -> directory | Base snapshots keyed by task.model_key |
| eval_path_map | ID -> file | Frozen dev/test JSONL keyed by task.dev_id/test_id |
| resource_path_map | ID -> path | External model library, wrapper, scorer and optional template resources |
| gpu_allocation | Integer list | Confirmed host GPU IDs; unique, nonnegative, at most four devices, count dividing global batch |
| env | Optional string map | Machine-specific environment for Agent and worker processes |

All relative profile paths resolve against the profile file's directory. `~` is expanded. Selected resource IDs must exist in the appropriate map; similar-looking names do not trigger a fallback to some other resource.

The default task uses these IDs:

| Role | Profile map and key |
|---|---|
| Training source root | dataset_path_map.shape_color_train |
| Common base model | model_path_map.internvlu_base |
| Development manifest | eval_path_map.shape_color_dev |
| Final-test manifest | eval_path_map.shape_color_test |
| External training library | resource_path_map.internvlu_train |
| Host text-preparation wrapper | resource_path_map.prompt_wrapper |
| Rule-scoring dependency | resource_path_map.scorer_cvlib |
| Model-library compatibility metadata | resource_path_map.training_meta |

The prompt wrapper records the tool used by host data/text preparation; the research loop does not regenerate evaluation labels with it. Accepted text comes from the frozen source manifest or the fixed frame-conditioned helper. The configured scorer_cvlib is loaded as the actual cvlib file during rule scoring; it is not resolved through a historical server path.

`training_meta` fills the low-level library's train_meta setting. Submitted `rows.json` remains the actual training input; this metadata file must not be mistaken for the current selected dataset.

In native mode, the GPU Python and external model-library directory must exist on the host. In Docker mode those two may exist only in the image. Source data, weight/configuration assets, evaluation files and scoring/preparation scripts must remain host-readable for preflight and freezing.

Profile GPU IDs must exactly match the shared campaign's recorded allocation. The profile expresses the requested machine resources; the ledger records the kickoff allocation and consumption. Do not change either midway through a run.

Environment values must be strings. The resolver rejects CUDA_VISIBLE_DEVICES, PYTHONPATH and the method's VIDEO_SSL_RUN/AGENT/CURRENT_TASK control names. GPU subprocesses receive their device visibility from the ledger. A profile PATH may be used to find the native Agent executable. Keep credentials out of static configuration: configuration snapshots and `check` output are reviewable artifacts.

## Task definition: tasks/<task_id>.yaml

### Identity and resources

| Field | Meaning |
|---|---|
| task_id | Public method-task ID matching the filename |
| goal | Research objective, also supplied to the Agent |
| dataset_id | Key in dataset_path_map |
| sources_manifest | Relative path inside the selected dataset root; absolute paths and `..` are rejected |
| model_key | Key in model_path_map |
| training_config | Low-level YAML template path relative to the task file |
| resources.model_library | Resource-map key for the external InternVL-U implementation |
| resources.prompt_wrapper | Resource-map key for host preparation provenance |
| resources.scorer_cvlib | Resource-map key for the scorer dependency |
| dev_id / test_id | Independent keys in eval_path_map |

There are no physical server paths or Agent model settings in the default task. You can version a task to compare construction constraints while selecting the same machine profile. The underlying supported real task remains shape_color_then_scale; a renamed YAML does not implement a new backend.

### Strategy, framing and orchestration

| Field | Meaning |
|---|---|
| strategy_timeout_seconds | Positive finite per-round strategy interval, starting at next and gating submit; no unlimited/null mode currently |
| estimated_trial_seconds | Positive conservative train+dev estimate for campaign headroom checks |
| preinit | Boolean controlling one untrained-base dev reference before the queue |
| finalize | manual or auto; host-side final test only |
| frame_policy.mode | select enables frame decisions; endpoints is compatibility-only F0-to-final |
| frame_policy.max_input_frames | Positive maximum number of ordered input/context frames |
| frame_policy.target | select permits declared intermediate/final targets; final fixes the last frame |
| max_dataset_bytes | Positive total accepted-directory byte limit; default example 500,000,000 |
| max_text_chars | Positive per-row text limit; default example 8,000 |

Inputs are chronological, unique, available, and strictly earlier than the target. The first input is the generated image's conditioning image. Other inputs provide visual context. A target is still one image, not an output video.

### contract: training and evaluation semantics

The backend validates this contract independently of the public task metadata. It rejects unknown or missing contract fields rather than silently reverting to the old experiment's recipe.

| Field | Meaning |
|---|---|
| task | Currently shape_color_then_scale |
| rows_per_trial | Exact distinct-video row count |
| max_trials | Maximum declared rounds, 1..7 |
| model_init_seed / train_seed / build_seed / eval_seed | Integer seeds for their named roles |
| selection.metric | Currently dev.mean |
| selection.tie_break | Currently baseline_then_earliest |
| selection.retrain | Must be false |

Training fields:

| contract.training field | Meaning / supported constraint |
|---|---|
| max_steps | Positive optimizer-step count |
| global_batch | Positive number of rows consumed per step |
| epochs | One ordered pass |
| shuffle | false; preserve submitted order |
| warmup_steps | Integer from zero to less than max_steps |
| scheduler | linear_warmup_then_cosine |
| lora_lr / gen_decoder_lr | Frozen learning rates |
| lm_weight / gen_weight / gen_warmup_steps | Frozen objective weights and generation warmup |
| image_size | Currently 512 canonical image size |
| checkpoint | last_step |
| restart_from_base_each_trial | true |
| timeout_seconds | Positive training-process cap, also limited by campaign headroom |

The one-pass invariant is `rows_per_trial == max_steps * global_batch`. The world size must divide global_batch. Each GPU handles its fixed rows with accumulation as needed; no additional rows or second shuffle are inserted by the backend.

Evaluation fields:

| contract.evaluation field | Meaning |
|---|---|
| samples | Exact dev/preinit item count |
| test_samples | Exact final-test item count, independently configurable |
| diffusion_steps | Positive generation-step count |
| all_cfg_scale / part_cfg_scale | Frozen generation CFG settings |
| scorer | analogy_grid_v1_scale |
| aggregation | arithmetic_mean_including_invalid |
| invalid_image_score | Zero |
| dev_timeout_seconds / test_timeout_seconds | Positive per-process caps |

Changing a template does not change the task's score weights or allow a different evaluation target. The current scorer is task-specific. Training frame choices never alter the frozen F0-to-final dev/test format.

## Low-level training template

The provided template derives from the first real pipeline-agent train.yaml. It preserves the model architecture, adapters, preprocessing, loss/optimizer structure, generation conditioning and distributed settings used there, while removing machine-specific server paths.

Supported whole-value placeholders:

| Placeholder | Resolution |
|---|---|
| __MODEL_PATH__ | Selected model snapshot |
| __TRAIN_META__ | resource_path_map.training_meta |
| __MAX_STEPS__ / __GLOBAL_BATCH__ | Task optimizer steps and global batch |
| __LORA_LR__ / __GEN_DECODER_LR__ | Task learning rates |
| __LM_WEIGHT__ / __GEN_WEIGHT__ / __GEN_WARMUP_STEPS__ | Task objective settings |
| __WARMUP_RATIO__ | Task warmup_steps / max_steps |
| __MODEL_INIT_SEED__ / __EVAL_SEED__ | Task seeds |
| __DIFFUSION_STEPS__ / __ALL_CFG__ / __PART_CFG__ | Task generation settings |

Resolution preserves numeric types; this is not arbitrary string evaluation or a general templating language. Unknown placeholder names fail. A custom template containing explicit steps/batch/seeds/rates/objectives/generation settings must agree with the task. The resolver rejects disagreement instead of applying undocumented override precedence.

Initialization writes training.resolved.yaml. GPU jobs point their actual output directory at the current operation and write train/resolved_train.yaml for inspection. The training loop and warmup use the task contract; the external template does not introduce a separate hidden schedule.

## Source and evaluation manifests

The training manifest is JSONL, one record per source, with source_video_id, video, f0, final, frame_count, description, eval_text and optional frames. Source IDs and normalized video paths must be unique. All image/video paths resolve relative to the manifest file, not the caller's current directory.

Frame-bank entries use canonical intermediate index strings such as `15`, not `015`, zero, final, negative or out-of-range indices. The host exports and audits the images before initialization. See [training-source example](../data/train_sources.example.jsonl).

Evaluation records contain a unique string id, exactly one image, target_image and conversations. cond_image defaults to that first image and must agree with it if supplied. Relative image paths resolve against the evaluation JSONL. The host must ensure these represent F0-to-final cases and properly independent splits; structural validation is not semantic split auditing. See [evaluation example](../data/eval.example.jsonl).

The initializer generates absolute-path dev.resolved.jsonl/test.resolved.jsonl without changing text or targets. It also publishes dev-only inputs/targets as workspace/dev_feedback.jsonl for permitted prediction-to-ground-truth analysis; it does not publish a test-feedback file. Original manifests and referenced evaluation images are hashed. Training-source hashes are computed by the worker and checked against submitted canonical pixels and indices.

## Runner configuration

| Field | Meaning |
|---|---|
| mode | native or docker |
| agent.executable | Required installed CLI name/path; current command format is Codex |
| agent.model | String model override or null to use the installed CLI configuration |
| agent.reasoning_effort | String effort override or null to use CLI defaults |
| timeout_seconds | Optional whole-session cap, distinct from each strategy interval |
| cli_python | Optional uv CLI Python selector; prepare the project environment with this interpreter before research |
| docker | Docker-file reference relative to runner.yaml, resolved only in Docker mode |

The Agent model is not the downstream InternVL-U base. Changing agent.model changes the research agent; changing profile.model_path_map changes the trained model resource and must remain consistent with the selected task/template.

Explicitly pin Agent settings when reproducibility requires them; null intentionally delegates to the local CLI configuration. Authentication is prepared before launch. The framework does not log into a provider or install the Agent for you.

## Optional Docker configuration

| Field | Meaning |
|---|---|
| image | Required runnable GPU image name/tag/digest |
| cli_python | Prepared container CLI interpreter (example: /opt/video-ssl-method/.venv/bin/python); do not duplicate a conflicting runner setting |
| mount_profile_resources | Default true: mount mapped profile resources and evaluation-manifest parent directories |
| read_only_mounts | Extra paths relative to docker.yaml or absolute paths |
| env | Container-specific string environment; no duplicate profile.env keys or method-control overrides |
| credential_files | Host-file -> absolute container-file mapping, mounted read-only |
| pass_env | Environment variable names forwarded from the launching host |

With automatic profile mounting, mapped dataset/model/evaluation resources need to exist on the host. An external model library supplied only by the image is not bind-mounted as a nonexistent host path. Add parent cache directories when snapshot files resolve through blobs or other symlinks outside the mounted snapshot.

The repository is read-only; the current run and dedicated campaign directory are writable. Duplicate mount destinations are consolidated, with explicit writable run/ledger requirements taking precedence. Keep campaign.json in a dedicated directory, not the filesystem root or repository root. The ledger needs its directory writable for locks and atomic replacement, so a single-file bind mount is insufficient.

The provided Dockerfile prepares `/tmp/video-ssl-home/.codex` for the host UID/GID. Map auth.json there when appropriate. The runner passes GPU devices from the campaign, uses the host network/IPC configuration, and removes its named container on completion or failure. These settings are implementation behavior, not a claim that mounted test material is inaccessible to same-permission code.

Native execution does not inspect this file at all, even if runner.docker references a nonexistent file. Keep it only if Docker is a supported deployment option.

## Protocol configuration

`protocol` selects the Agent's stable operating procedure. It is read from the selected Markdown file, hashed, copied to workspace/METHOD.md and protected against modification during the run. A private custom protocol can be selected explicitly without changing the default method protocol.

Research direction lives in the body of run_prompt.md. Numeric contracts live in the task. Physical locations live in the profile. A protocol instruction cannot override those layers or grant extra optimizer attempts.

The shipped selections are method (base), instruction (base plus prescriptive research appendix), and skill (base plus skill-grounded appendix). Both the base and selected appendix are frozen. Only the composed protocol is copied to workspace/METHOD.md. There is no root METHOD.md duplicate; each generated workspace has its selected METHOD.md. The packaged default is kept synchronized. Custom explicit protocol paths are used verbatim.

## Campaign ledger

The mutable ledger is separate from immutable configuration:

| Field | Meaning |
|---|---|
| deadline | Fixed timezone-aware kickoff deadline |
| report_reserve_seconds | At least 1800; default example 5400 seconds reserved after train/dev/Agent work |
| gpu_allocation | Confirmed device list, identical to the profile selection |
| max_concurrent_gpus | Integer 1..4 |
| training_budget | Integer 1..7 for all runs sharing the kickoff |
| training_attempts | Initially empty for a genuinely new kickoff; never cleared to reset consumption |

A registered failed training attempt consumes a slot. Preinit/evaluation does not perform optimizer updates, but consumes wall-clock budget. Every registration checks remaining time and attempts. Initial preflight does not guarantee that every requested round will fit the final deadline.

Timeout behavior is bounded by the minimum applicable limit:

| Operation | Applicable limits |
|---|---|
| Strategy submission | Per-round strategy interval; overall Agent/campaign window |
| Training | training.timeout_seconds; campaign deadline minus report reserve |
| Dev/preinit | dev_timeout_seconds; campaign deadline minus report reserve |
| Final test | test_timeout_seconds; campaign deadline minus 1800 seconds |
| Entire Agent | Optional runner timeout; campaign deadline minus report reserve |

Do not change the ledger's identity, deadline or allocation to revive an old run. Interrupted/in-flight operations need host inspection, not an automatic retry or state-file reset.

## Generated snapshots and inspection

Initialization records original configuration/manifest copies under config/inputs, a campaign.initial.json snapshot, generated backend input, fully resolved backend configuration, training YAML and absolute evaluation manifests. plan.json records reference paths, settings and hashes.

The live campaign remains external and mutable. Its initial copy is provenance, not the ledger used for later accounting. CLI result/operation receipts and per-round accepted code/data describe what actually ran; workspace code describes only the latest editable candidate.

`check` and `init` do not prove model compatibility, GPU availability, a clean semantic split or successful multi-image training. Model assets, external-library dependencies, custom recipes and container behavior require a real server acceptance run. The method does not claim universal support for arbitrary tasks, trainers or scorers.

## uv runtime and dependency ownership

Run `uv sync --locked --python /ABS/PYTHON3.12` once during host preparation, then use `uv run --locked video-ssl check ...` and `uv run --locked video-ssl run ...`. `pyproject.toml`, `.python-version` and `uv.lock` are required frozen run inputs. The CLI lock contains only method dependencies; it does not describe or synchronize the GPU model environment.

The Agent receives absolute `--project` commands using `--frozen --no-sync --no-python-downloads`; never run dependency synchronization during research. The host resolver uses the same prepared project. Other worker actions and frame construction use `uv run --no-project --no-config --no-python-downloads --python <profile.python> -- <profile.python> ...`. The explicit command preserves that interpreter's site-packages even when the CLI environment is active. Native uv is resolved from PATH or the CLI Python's uv module; Docker commands use the image's uv.

For native mode, cli_python selects a compatible CLI interpreter for the prepared project; it does not select GPU Python. In Docker mode set env.UV_PROJECT_ENVIRONMENT to the prepared image CLI venv, as in the example. The supplied Dockerfile pins uv 0.13.6, uses the committed lock, and requires a >=3.12 METHOD_PYTHON already present in the base image. Container source paths stay mounted at their host absolute paths; the CLI venv is read from the image.

## Lifecycle configuration

The host can override the machine profile using --profile on check/init/run.
The declared task list is processed in order; initial iterations applies to every
task and includes its baseline. The sum of all tasks' attempts must fit the shared
campaign. Each task has its own config snapshot, workspace, Session records and
final checkpoint selection; a multi-task root aggregates their queues and reports.

Use --suite-dir for existing-suite commands (--run-dir is an alias). init/run load
an explicitly selected or active existing suite, preserving its frozen inputs.
Use --new-suite to request a new directory and a fresh search without resetting the
campaign. resume --inspect validates without launching or mutating saved state.
resume starts a new Agent process with the saved stage and a new log directory.
The command never resets an uncertain training operation or renews a strategy timer.

extend --iterations N changes the effective total for the selected task, writes a
frozen extension receipt, and updates the saved plan/public task metadata. It does
not edit the original run_prompt or prior submissions. Select SUITE/tasks/TASK for
multi-task extension. Pending attempts in sibling tasks count against the same
budget; no extension is allowed once any task has begun final test.
