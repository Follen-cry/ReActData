# reactdata-v6 r1 (design v3)

Wall time: 2.98 h (budget 5.5 h).

## Training noise (P1)

Fixed baseline, run seeds [1000, 1001, 1002, 1003, 1004]: dev 0.1493, 0.1790, 0.1336, 0.0684, 0.1022; mean 0.1265, sd_train 0.0427 (= delta). Base (untrained) dev 0.0370.

## Screening (one run per candidate, seed = screen seed; NOT evidence on its own)

D is against the per-item mean of the baseline replicates.

| id | parent | class | status | dev | dev scale | dev move | dev geo | D vs baseline | rows / unique clips | task rows |
|---|---|---|---|---|---|---|---|---|---|---|
| c000 | - | fixed | trained | 0.1493 | 0.1680 | 0.1635 | 0.1162 | - | 128 / 128 | geo 42 / move 43 / scale 43 |
| c001 | c000 | static | trained | 0.1245 | 0.1430 | 0.1384 | 0.0922 | -0.0020 | 128 / 128 | geo 16 / move 56 / scale 56 |
| c002 | c001 | static | trained | 0.1818 | 0.2460 | 0.1723 | 0.1271 | +0.0553 | 128 / 128 | geo 42 / move 43 / scale 43 |
| c003 | c002 | static | trained | 0.1774 | 0.1425 | 0.2604 | 0.1293 | +0.0509 | 128 / 128 | geo 42 / move 43 / scale 43 |
| c004 | c003 | static | trained | 0.1649 | 0.1567 | 0.2378 | 0.1002 | +0.0384 | 128 / 128 | geo 42 / move 43 / scale 43 |
| c005 | c004 | static | trained | 0.1345 | 0.1414 | 0.1657 | 0.0965 | +0.0080 | 128 / 128 | geo 42 / move 43 / scale 43 |
| c006 | c005 | static | trained | 0.1778 | 0.1982 | 0.2296 | 0.1055 | +0.0513 | 128 / 128 | geo 42 / move 43 / scale 43 |

## Confirmation (top 2, seeds [1001, 1002, 1003, 1004], paired with the baseline replicates of the same seeds)

| id | D | per-seed D | SE_train | SE_item | CI (Bonferroni) | dev_scale D | dev_move D | dev_geo D |
|---|---|---|---|---|---|---|---|---|
| c002 | -0.0377 | -0.134, -0.023, -0.009, +0.015 | 0.0302 | 0.0070 | [-0.1072, +0.0318] | -0.0598 | -0.0254 | -0.0279 |
| c006 | -0.0204 | -0.100, -0.002, +0.062, -0.041 | 0.0302 | 0.0076 | [-0.0902, +0.0494] | -0.0157 | -0.0289 | -0.0165 |

**Selection**: c000 — no confirmed candidate passed: fixed baseline (delta 0.0427).

## Test (P3, formal seeds [2000, 2001, 2002]; existing set with usage history, not fresh)

- Base: 0.0367; fixed baseline runs: 0.1008, 0.1146, 0.1089

## Cost

| item | GPU-h |
|---|---|
| confirm_eval | 3.46 |
| confirm_probe | 0.84 |
| confirm_train | 1.24 |
| final_eval | 1.73 |
| final_probe | 0.00 |
| final_train | 0.15 |
| search_and_noise_eval | 5.22 |
| search_and_noise_probe | 0.64 |
| search_and_noise_train | 1.21 |

Builder Agent: 7 calls, $3.68. Disk at the end: 15.42 GB.

## Failures

- none
