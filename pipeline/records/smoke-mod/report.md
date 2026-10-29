# reactdata-v5 smoke-mod (design v3) (SMOKE: not a result)

Wall time: 0.49 h (budget 8 h).

## Training noise (P1)

Fixed baseline, run seeds [1000, 1001]: dev 0.0000, 0.0488; mean 0.0244, sd_train 0.0345 (= delta). Base (untrained) dev 0.0000.

## Screening (one run per candidate, seed = screen seed; NOT evidence on its own)

D is against the per-item mean of the baseline replicates.

| id | parent | class | status | dev | dev scale | dev move | D vs baseline | rows / unique clips | ViT images |
|---|---|---|---|---|---|---|---|---|---|
| c000 | - | fixed | trained | 0.0000 | 0.0000 | 0.0000 | - | 64 / 16 | 64 |
| c001 | c000 | schedule | trained | 0.0000 | 0.0000 | 0.0000 | -0.0244 | 64 / 16 | 64 |
| c002 | c001 | schedule | trained | 0.0000 | 0.0000 | 0.0000 | -0.0244 | 64 / 16 | 64 |

## Confirmation (top 1, seeds [1001], paired with the baseline replicates of the same seeds)

| id | D | per-seed D | SE_train | SE_item | CI (Bonferroni) | scale D | move D |
|---|---|---|---|---|---|---|---|
| c001 | -0.0229 | -0.023 | 0.0488 | 0.0162 | [-0.1237, +0.0778] | -0.0458 | +0.0000 |

**Selection**: c000 — no confirmed candidate passed: fixed baseline (delta 0.0345).

## Test (P3, formal seeds [2000]; existing set with usage history, not fresh)

- Base: 0.0000; fixed baseline runs: 0.0000

## Cost

| item | GPU-h |
|---|---|
| confirm_eval | 0.01 |
| confirm_probe | 0.05 |
| confirm_train | 0.07 |
| final_eval | 0.02 |
| final_probe | 0.00 |
| final_train | 0.03 |
| search_and_noise_eval | 0.09 |
| search_and_noise_probe | 0.11 |
| search_and_noise_train | 0.24 |

Builder Agent: 2 calls, $0.52. Disk at the end: 7.57 GB.

## Failures

- {"job": "c000-s1000-g4", "kind": "infra", "attempt": 1, "action": "resume", "log_tail": "le \"/homes/55/junlin/miniconda3/envs/internvlu/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py\", line 355, in wrapper\n    return f(*args, **kwargs)\n  File \"/homes/55/junlin/miniconda3/envs/internvlu/lib/python3.10/site-packages/torch/distributed/run.py\", line 918, in main\n    run(args)\n  File \"/homes/55/junlin/miniconda3/envs/internvlu/lib/python3.10/site-packages/torch/distributed/run.py\", line 909, in run\n    elastic_launch(\n  File \"/homes/55/junlin/
- {"job": "c001-s1001-g4", "kind": "infra", "attempt": 1, "action": "resume", "log_tail": "le \"/homes/55/junlin/miniconda3/envs/internvlu/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py\", line 355, in wrapper\n    return f(*args, **kwargs)\n  File \"/homes/55/junlin/miniconda3/envs/internvlu/lib/python3.10/site-packages/torch/distributed/run.py\", line 918, in main\n    run(args)\n  File \"/homes/55/junlin/miniconda3/envs/internvlu/lib/python3.10/site-packages/torch/distributed/run.py\", line 909, in run\n    elastic_launch(\n  File \"/homes/55/junlin/
