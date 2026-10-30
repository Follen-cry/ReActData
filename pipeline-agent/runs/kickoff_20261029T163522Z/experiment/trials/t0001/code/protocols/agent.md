# Dataset coding protocol

This trusted research workflow runs Codex directly with the user's permissions,
without a container, sandbox or approval prompts. Rules here are behavioral agreements,
not access controls. Do not change this protocol or the harness to bypass them.

## Resources and commands

Read task.json, COMMANDS.md and existing NOTES.md first. COMMANDS.md contains absolute
paths and exact Python/CLI/build commands. Read the builder and relevant dev feedback.
You may read files, edit code and run shell commands. Use task, validate --dataset PATH,
run --dataset PATH --request-id UNIQUE, and status [TRIAL_ID]. run validates again.

Edit build_dataset.py and helper code in this workspace. Maintain NOTES.md with each
hypothesis, dataset change, request ID, trial ID and dev result. Keep generated data
under dataset/ (excluded from the code archive); remove the previous generated dataset
before rebuilding if needed. Completed trial data is separately copied.

## Fixed experiment rules

Do not edit harness code, task.json, resolved configs, source videos, model weights,
state files, integrity records or previous trial artifacts. Do not change the fixed
seed, training recipe, evaluation rules or budgets. Do not access unrelated host files
or credentials. One input F0 image and one final target per row; only configured
sources and the current task structure are supported.

Read trial result.json and dev artifacts before changing strategy. Record hypotheses
separately from measurements. Simulation scores are synthetic placeholders, not quality
measurements or checkpoints. Never claim real training succeeded in simulation.

## Budget and failures

Run trials serially. Each accepted attempt consumes one trial, including training
failures; validation failures do not. Reuse the request ID after a timeout. Never invent
another ID to retry an uncertain running job. Do not remove locks or modify state to
force recovery. Report interrupted jobs for host inspection. If the real backend is
unimplemented or dependencies are missing, report the blocker and stop; do not switch
to simulation, fabricate metrics, or repeat the same failure until budget exhaustion.

## Finish

Choose a completed trial using dev feedback and write only its ID to selection.txt.
If none completed, report the failure in NOTES.md instead of inventing a selection.
Never invoke finalize, read test resources or final.json/report.md, or use test feedback
to update strategy. Finalization is a host responsibility by convention only: this
version does not technically prevent invoking it or reading accessible files.

## Experiment contract (single task)

The only task is shape_color_then_scale. Read the resolved task.json: formal runs use
128 distinct videos, 8 optimizer steps, global batch 16, one pass in manifest order.
Your only levers are which videos to select and their order. Do not change image pixels,
frame choice or text: each row must use the source's exact eval_text and canonical PNGs.
The fixed seeds are model_init_seed=42, train_seed=1000, build_seed=1000, eval_seed=0.
The first submission must be the unchanged seeded random baseline. Stop for host
inspection if baseline training fails. The budget is 1 baseline + 6 improvements;
failed accepted submissions also consume these slots. Simulation may explicitly use
fewer samples/trials, but that does not represent the formal training recipe.

Before each improvement, record in NOTES.md: evidence from prior dev results, one
hypothesis, selection/order changes, expected outcome and what would refute it.
Afterward record the request/trial ID, measured result and interpretation. No dynamic
training feedback/probes, multi-seed confirmation, or statistical-significance claim.

Select the completed trial with the highest dev.mean; exact ties prefer the baseline,
then the earliest trial. In simulation all synthetic scores tie: prefer the baseline
and do not claim strategy quality. Submit its ID only. Host finalization evaluates the
already saved last-step checkpoint without retraining or selecting a new seed.
