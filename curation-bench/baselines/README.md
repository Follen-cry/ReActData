# Fixed Data-Construction Baselines

This directory contains non-Agent comparison entry points. It follows the upstream
separation between fixed baselines and the Agent's editable curation workspace.

The shipped baseline samples distinct source videos with the frozen seed, shuffles
the selected order, and uses F0 as input and the final frame as supervision target.
`generate_random.py` executes the repository's unchanged `curation/curate.py`;
there is one baseline implementation, not another independently maintained copy.

## Run the fixed baseline

Prepare configuration and initialize a manual suite using the README instructions.
Activate an iteration and inspect `video-ssl task`. Use its exact Python, task file,
backend directory and output path:

```bash
uv run --no-project --no-config --no-python-downloads \
  --python /ABS/GPU/PYTHON -- /ABS/GPU/PYTHON baselines/generate_random.py \
  --config /ABS/SUITE/workspace/task.json \
  --tools /ABS/REPO/vendor/video_ssl_backend \
  --output /ABS/SUITE/workspace/dataset
```

Then use the same `video-ssl audit`, `submit`, `finetune` and `eval` tools as Agent
candidates. Fixed baselines do not start an Agent, change the trainer/evaluator,
reset budgets or consume test feedback. A baseline comparison requiring another
training run must have its own declared budget and suite configuration.

The Agent starter remains `curation/curate.py`, copied into each new suite's
`workspace/curation/`. Agent edits affect that workspace copy only.
