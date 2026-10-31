# Video SSL Data Construction Method

A continuous research agent develops a program that constructs supervision from training videos. It selects videos, input/context frames, supervision target frames and sample order. A fixed downstream training-and-dev procedure supplies feedback; the host evaluates the selected saved checkpoint once on a separate test split.

The implementation follows the experiment organization of curation-bench: a run declaration, machine resource profiles, task definitions, an agent protocol, per-iteration tool dispatch, immutable submissions and persistent results. It is organized as a data-construction method rather than an agent-ranking benchmark. Source ancestry is recorded in [historical upstream README](docs/history/upstream-readme.md).

## Start here

- [Research-agent operating manual](protocols/method.md): complete initialization, baseline, construction, training, evaluation, feedback, monitoring and failure procedures.
- [Configuration reference](docs/configuration.md): every configuration layer, field ownership, path resolution, timeouts and preparation requirements.
- [Suite lifecycle](docs/lifecycle.md): task queues, saved stages, resume, extension and best-code restoration.
- [Default task](tasks/shape_color_video_ssl.yaml), [machine profile template](profiles/example.yaml), [runner](configs/runner.yaml), and optional [Docker settings](configs/docker.yaml).

The authoritative base protocol is `protocols/method.md`. The selected protocol is frozen and copied to each task workspace as `METHOD.md`; the Agent starts from that workspace's `AGENTS.md`, `METHOD.md`, `COMMANDS.md`, `task.json` and `NOTES.md`. The matching package resource is retained for installation and fallback loading.

## Configuration layout

| File | Responsibility |
|---|---|
| [run_prompt.md](run_prompt.md) | Task ID, iteration count, selected profile/runner/protocol, campaign and this experiment's research direction |
| `profiles/<machine>.yaml` | Python, dataset/model/evaluation/library paths, assigned GPUs and machine environment |
| `tasks/<task>.yaml` | Objective, resource IDs, sample budget, frame policy, training/evaluation contract and selection rules |
| [configs/runner.yaml](configs/runner.yaml) | Native/Docker execution, Agent executable/model/effort and optional whole-session timeout |
| [configs/docker.yaml](configs/docker.yaml) | Optional image, container environment, mounts, credential-file mappings and environment forwarding |
| [protocols/method.md](protocols/method.md) | Stable instructions for how the Agent conducts the experiment |
| [configs/training/internvlu.yaml](configs/training/internvlu.yaml) | Low-level InternVL-U model/adapter/preprocessing template, resolved using task/profile values |
| Training source JSONL | Frozen source IDs, videos, frame banks, descriptions and evaluation-format text |
| Campaign JSON | Shared allocation, deadline, accepted training attempts and remaining budget |

**Native mode does not read Docker configuration.** The Docker file may be absent or invalid without affecting a native run. Agent model settings belong in runner.yaml in both modes.

## Host preparation

Prepare the following before starting a research session:

1. A Linux GPU machine with the allocated devices and a compatible InternVL-U training environment.
2. A separate method CLI environment with Python >=3.12. The GPU worker may use a different interpreter.
3. The external InternVL-U library, fixed base snapshot and configured scorer file.
4. A training-source manifest and canonical 512 x 512 PNG endpoints/intermediate frames.
5. Independently prepared dev/test manifests and images, with split and duplicate-data auditing performed by the host.
6. A new campaign ledger, or the existing ledger for the same kickoff; never reset an existing budget.
7. An installed and authenticated Agent CLI for native execution, or an appropriately prepared container image and credential mapping for Docker execution.

Install uv, then prepare the locked CLI project from this repository (Python >=3.12):

```bash
uv sync --locked --python /path/to/python3.12
```

Prepare GPU/data dependencies in the **intended GPU environment**, not implicitly in the method CLI environment:

```bash
uv pip install --python /path/to/gpu/python -r vendor/video_ssl_backend/requirements.txt
```

Prepare the external training library's own dependencies according to that library. This repository does not download weights, rebuild external environments or supply all model dependencies automatically. Researchers must not change environments during a running experiment.

## Configure and launch

Copy [profiles/example.yaml](profiles/example.yaml) to a machine-specific profile and fill its actual paths:

```bash
cp profiles/example.yaml profiles/server_a.yaml
```

Prepare the source/evaluation manifests and copy [campaigns/example/campaign.json](campaigns/example/campaign.json) into a new kickoff directory. Replace the deliberately invalid deadline placeholder and confirm the GPU allocation. Set `profile: server_a` and the campaign path in `run_prompt.md`.

Check configuration without training, then initialize and launch the declared task queue (one continuous Agent session per task):

```bash
uv run --locked video-ssl check --prompt run_prompt.md
uv run --locked video-ssl run --prompt run_prompt.md --root runs/video-ssl-001
```

`check` performs static schema, reference, resource, manifest-count, recipe-consistency and budget checks. It does not load the model, prove GPU availability or establish semantic split independence. Templates contain placeholder paths and are not runnable until completed.

This follows upstream's `uv run` project workflow; `uv tool` is for standalone tools and is not the pipeline launcher. `uv.lock` pins CLI dependencies. The host synchronizes the project before research; generated Agent commands use `uv run --frozen --no-sync --project /ABS/REPO -- video-ssl ...` so they work from the workspace without modifying dependencies. GPU/data workers use `uv run --no-project --no-config --no-python-downloads --python /ABS/GPU/PYTHON -- /ABS/GPU/PYTHON ...`, preserving the exact existing interpreter and its installed model packages. Training's distributed subprocess inherits that worker interpreter. The plan freezes `pyproject.toml`, `uv.lock` and `.python-version` alongside experiment inputs.

`iterations: 3` means **one trained baseline plus two improvements**. A suite accepts multiple declared tasks and processes them in declaration order, using an isolated workspace and Agent session for each task. All tasks share the campaign budget. The underlying real task is currently `shape_color_then_scale`; adding a new task ID is not sufficient to support a different scorer or model interface.

The default task retains the first real pipeline-agent experiment's recipe: 128 rows, 8 optimizer steps, global batch 16, warmup 3, LoRA LR 2e-5, decoder LR 1e-4 and fixed initialization/training/build/evaluation seeds. These are task values, not hidden loop constants; custom tasks must satisfy the backend's supported contract.

## Native and Docker execution

Set `mode: native` in runner.yaml to launch the local Agent directly on the Linux server. Native does not require Docker. It also does not make the CUDA/POSIX pipeline a Windows-native `codex.exe` workflow.

For Docker, set `mode: docker` and configure docker.yaml. A prepared GPU image may be built from a pinned base that already contains the required model environment and Agent CLI:

```bash
docker build -f docker/VideoSSL.Dockerfile \
  --build-arg BASE_IMAGE=YOUR_PINNED_GPU_IMAGE \
  --build-arg METHOD_PYTHON=/opt/conda/bin/python \
  -t video-ssl-method:local .
```

The Docker CLI Python selected by METHOD_PYTHON must be >=3.12; profile.python still selects the existing GPU interpreter. The image builds a separate /opt/video-ssl-method/.venv with uv. Keep UV_PROJECT_ENVIRONMENT in docker.yaml pointing there when mounting the repository read-only. Set that image in docker.yaml and ensure the configured GPU Python and library paths exist inside it. The runner mounts the repository read-only, the run directory read/write, configured resources read-only and the dedicated campaign directory read/write. Linux containers use the host UID/GID and a writable `/tmp/video-ssl-home`; authentication can be mapped to `/tmp/video-ssl-home/.codex/auth.json`.

See the configuration reference for mount resolution, environment forwarding, cache/symlink considerations and the actual access boundary. Docker image/GPU/authentication behavior still needs a real server acceptance run.

## Manual staged operation

The same tools are available without an autonomous Agent:

```bash
uv run --frozen --no-sync video-ssl init --prompt run_prompt.md --root runs/manual-001
uv run --frozen --no-sync video-ssl preinit --run-dir runs/manual-001
uv run --frozen --no-sync video-ssl next --run-dir runs/manual-001
uv run --frozen --no-sync video-ssl task --run-dir runs/manual-001
```

The `task` response includes `build_command` as an exact argv list, plus `python`, `builder`, `task_file`, `tools` and `default_submission_path`. Execute that uv command to construct the dataset. Its general shell form is:

```bash
uv run --no-project --no-config --no-python-downloads \
  --python /ABS/GPU/PYTHON -- /ABS/GPU/PYTHON /ABS/RUN/workspace/curation/curate.py \
  --config /ABS/RUN/workspace/task.json --tools /ABS/REPO/vendor/video_ssl_backend \
  --output /ABS/RUN/workspace/dataset
```

Then:

```bash
uv run --frozen --no-sync video-ssl audit --run-dir runs/manual-001 --path /ABS/RUN/workspace/dataset
uv run --frozen --no-sync video-ssl submit --run-dir runs/manual-001 --path /ABS/RUN/workspace/dataset
uv run --frozen --no-sync video-ssl finetune --run-dir runs/manual-001
uv run --frozen --no-sync video-ssl eval --run-dir runs/manual-001
uv run --frozen --no-sync video-ssl next --run-dir runs/manual-001
```

For each improvement, write `workspace/curation/hypothesis.md`, modify the builder, regenerate the current workspace dataset and repeat the tools. Do not modify immutable earlier submissions. After `next` returns `task_done`, the host may perform finalization:

```bash
uv run --frozen --no-sync video-ssl finalize --run-dir runs/manual-001
```

With task `finalize: auto`, the launcher performs this host step after the Agent has exited successfully. The Agent never runs final test itself. Successful stages are idempotent; failed or in-progress optimizer/evaluation operations are not blindly replayed. Use the host resume command to continue saved stages without resetting a Session; see the lifecycle guide.

## Read the results

```text
runs/video-ssl-001/
  README.md / results.tsv / suite_state.json
  config/
    inputs/                              Original configuration/manifest snapshots
    plan.json                            Resolved settings, source paths and hashes
    backend.input.json / backend.json    Generated worker input and resolved contract
    training.resolved.yaml
    dev.resolved.jsonl / test.resolved.jsonl
    campaign.initial.json                Snapshot; the live ledger remains shared
  workspace/
    METHOD.md / AGENTS.md / COMMANDS.md / task.json / NOTES.md
    dev_feedback.jsonl                   Public dev-only inputs and targets
    curation/curate.py / hypothesis.md
    dataset/
  preinit/dev/
  shape_color_video_ssl_iter1/
    code/ / curated/final_submission/
    train/ / dev/ / checkpoint.json / events.jsonl
    REPORT.md / changes.diff / selection_diff.json
  shape_color_video_ssl_iter2/ ...
  output/initial_prompt.md / run.jsonl
  final/result.json / test/
```

Start with the run README, then each round's REPORT, code diff and video/frame/order changes. `results.tsv` is framework-owned: the optional untrained reference is labeled `reference`, measured trained candidates receive `keep`/`discard`, and failed rounds receive `crash` with a blank score.

Final selection uses the highest completed dev.mean, with exact ties preferring the baseline and then earlier rounds. The selected saved last-step checkpoint is tested once without retraining. Frozen training-frame decisions do not change the F0-to-final dev/test objective.

## Source organization and verification

The active method entry points are `src/video_ssl/cli.py`, `tools/suite.py`, `tools/session.py`, `core/layout.py` and `runner/agent_runner.py`. The Python package is `video_ssl`; the public CLI is `video-ssl`. The bundled backend runs independently of the sibling pipeline-agent checkout.

The unused VQA tools, model registries, original comparison projects and Agent images have been retired to a recoverable sibling archive outside this checkout. The original paper-derived skills remain as an optional research library; the default method protocol does not load them or impose reading quotas; selecting the skill appendix explicitly enables them.

Relevant checks:

```bash
uv sync --locked --extra dev
uv run --locked --extra dev python -m pytest -q
```

Tests cover composed configuration, native ignoring Docker, frozen inputs, a real CPU worker boundary, frame/pixel/text validation, staged idempotence, failed-round behavior, checkpoint integrity, reports and configured scorer loading. Mocked training tests are not GPU quality evidence. Real training, altered recipes, multi-image context and Docker still require a new-budget GPU smoke run. Historical upstream VQA tests are outside this acceptance claim.

The framework retains upstream's suite/session/tools organization. See [repository organization](docs/organization.md) for ownership, retired files and naming changes, and [fixed baselines](baselines/README.md) for comparison generation. `video-ssl` is the only installed CLI; the old `datacuration-bench`/`reactdata` aliases have been retired. New suites write `suite_state.json` and use `.video_ssl_suite` for manual selection. Existing frozen experiments require their original code/configuration version; this is not an automatic migration of pre-refactor experiments. The lifecycle guide describes supported resume and extension operations for new suites.

## Upstream-style suite lifecycle

See [lifecycle and reuse details](docs/lifecycle.md) for factory/load/Session interfaces,
per-task Agent isolation, stage-aware resume, queue extension and code rollback.

```bash
uv run --locked video-ssl resume --suite-dir runs/video-ssl-001 --inspect
uv run --locked video-ssl resume --suite-dir runs/video-ssl-001
uv run --locked video-ssl extend --suite-dir runs/video-ssl-001 --iterations 5
uv run --locked video-ssl resume --suite-dir runs/video-ssl-001
```

The extension example requires sufficient remaining campaign budget and must happen
before any final test starts. For a multi-task suite, extend a task using
`--suite-dir runs/video-ssl-001/tasks/TASK_ID`; all tasks' pending training attempts
are included in the budget check. Iterations is the new total, including baseline.
Use `run --new-suite` for an explicitly new experiment; it does not reset the ledger.
`--suite-dir` is the preferred existing-suite argument; `--run-dir` remains an alias.
Choose `protocol: method`, `instruction`, or `skill` for base or optional research
guidance. All variants use the same frozen task/training/evaluation contract.

## Research reading scope

The Agent starts in the assigned task workspace, not at the repository root. Its
workspace AGENTS.md lists permitted research inputs and excluded development/history
material. Framework source is inspected only for a concrete CLI error; routine
research uses the resolved task, exact commands, declared data and same-task feedback.
These are reading instructions, not an operating-system sandbox. Native permissions
and the read-only Docker repository mount remain unchanged.

Historical material is indexed under [docs/history](docs/history/README.md).
Recoverable legacy code and temporary local installation/cache artifacts have been
moved to the sibling `../curation-bench-archive/` directory. They are not needed by
the pipeline and are outside the default repository mount and package discovery.
