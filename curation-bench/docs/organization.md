# Repository Organization

The method keeps upstream's suite/session/tools separation while replacing the
task-specific data, training and evaluation backend. It is not a second parallel
framework hidden behind compatibility entry points.

```text
curation/curate.py                 Agent starter, copied into each workspace
baselines/generate_random.py       Fixed non-Agent comparison entry
run_prompt.md                     Tasks, iterations and research direction
profiles/                         Machine resource lookup
configs/runner.yaml               Agent/native/container execution settings
configs/docker.yaml               Optional container settings
configs/training/                 Frozen low-level training template
tasks/                            Task contracts
protocols/method.md               Authoritative Agent operating protocol
src/video_ssl/
  cli.py                          One public video-ssl command
  core/                           Configuration, Target, runtime and uv helpers
  tools/
    suite.py                      SuiteController and target lifecycle
    session.py                    Per-target Session and dispatch
    submission.py                 Audit, baseline checks, immutable submissions
    training.py                   Framework training stage
    evaluation.py                 Preinit, dev and host final test
    reporting.py                  TSV, per-round reports and differences
  backends/video_ssl.py           Concrete task adapter for model/data workers
  runner/agent_runner.py          Native or Docker Agent launch
vendor/video_ssl_backend/          Actual frame/model/generation/scoring backend
skills/                           Optional upstream research reference
```

## Responsibilities retained from upstream

SuiteController owns the ordered task/iteration queue, saved suite state, resource
snapshot and lifecycle. Session dispatches the active target's tools. The CLI parses
commands and delegates to these objects. The runner launches one continuous Agent
session; model-specific code remains in the backend. These names do not imply the
old VQA-specific backend is still running.

The lifecycle now exposes upstream-style from_prompt, from_state_file,
get_run_plan, begin_next_target, get_active_session and mark_target_completed.
Session persists session_state.json and sequenced events as in upstream, with added
integrity checks. VideoSSLBackend is the concrete adapter used by the framework's
submission, training and evaluation tools.

A multi-task suite aggregates isolated task controllers, each reusing the same
SuiteController/Session implementation and receiving its own Agent workspace.
This preserves upstream's task-major queue and one-Agent-per-task organization.
The host can resume stages and extend a task queue. Before the next iteration,
accepted best code is restored; candidates and working copies remain archived.
See lifecycle.md for supported commands and deliberate operation guards. Codex
remains the configured Agent integration; other launchers are not implied.

## Baselines and the Agent starter

Upstream baselines/ contains independent non-Agent comparisons; its Agent starts
from curation/curate.py. The method uses the same distinction. Our fixed random
baseline invokes the immutable repository starter, and Agent sessions receive a
workspace copy. There is no second backend starter implementation. New baselines
must produce the same submission schema and use the same audit/train/eval tools.

## Retired files and names

The former src/benchmark package, video_suite/video_session/method_cli wrappers,
reactdata compatibility API, old datacuration-bench executable, VQA task templates,
model registry, contamination evaluator, original vendor/curation-train and
vendor/VLMEvalKit projects, old Agent Dockerfiles/patches, legacy comparison scripts
and their tests have been removed from the active tree. Backend worker.py is the
single worker entry; its old cli.py wrapper and duplicate starter are retired.

Recoverable copies are stored outside the checkout in the sibling
../curation-bench-archive/legacy-curation-bench directory, preserving relative paths.
They are not installed, collected as tests, or included in the default repository
mount. Temporary local bootstrap/dependency/cache artifacts are archived beside them.
The upstream revision remains recorded in docs/history/upstream-readme.md.

The source package is video_ssl; the distribution remains video-ssl-method and the
command remains video-ssl. New artifacts use suite_state.json and .video_ssl_suite.
Old frozen experiments must use their original checkout/configuration. Renaming
artifacts in place is not a supported migration.

## Documentation and optional resources

protocols/method.md is authoritative. Root METHOD.md is retired; only generated
workspace/METHOD.md is the research entry. The matching
src/video_ssl/protocols/method.md is package data used by installed/fallback loads;
a test checks that it matches the authoritative source. BENCHMARK.md and the backend
protocol duplicate are retired. docs/history/upstream-readme.md, LICENSE and backend PROVENANCE.md
retain source history; historical names there are intentional.

skills/ is retained as optional research material to preserve the upstream research
organization. The default method protocol does not scan it or impose quotas; the selected
skill appendix explicitly enables research-card reading. Original environment/model names inside those reference cards do not describe
the current execution environment. Docker files are optional only for native mode;
frame-preparation helpers, example manifests, backend default contract and current
tests remain useful and are retained.

Research Agent discovery begins with workspace AGENTS.md and its selected METHOD.md,
COMMANDS.md, task.json and NOTES.md. It reads declared resources and same-task feedback,
not the development/history tree. This instruction scope does not change native
permissions or make mounted files technically inaccessible.
