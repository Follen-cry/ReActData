# reactdata-v5 r1 (design v3)

Wall time: 6.03 h (budget 8 h).

## Training noise (P1)

Fixed baseline, run seeds [1000, 1001, 1002, 1003, 1004]: dev 0.3908, 0.3548, 0.3959, 0.4197, 0.3711; mean 0.3865, sd_train 0.0247 (= delta). Base (untrained) dev 0.0049.

## Screening (one run per candidate, seed = screen seed; NOT evidence on its own)

D is against the per-item mean of the baseline replicates.

| id | parent | class | status | dev | dev scale | dev move | D vs baseline | rows / unique clips | ViT images |
|---|---|---|---|---|---|---|---|---|---|
| c000 | - | fixed | trained | 0.3908 | 0.3731 | 0.4084 | - | 1600 / 16 | 1600 |
| c001 | c000 | schedule | trained | 0.3432 | 0.3505 | 0.3360 | -0.0432 | 1600 / 16 | 1600 |
| c002 | c001 | schedule | trained | 0.2974 | 0.2876 | 0.3071 | -0.0891 | 1600 / 16 | 2080 |
| c003 | c002 | schedule | trained | 0.3500 | 0.3655 | 0.3344 | -0.0365 | 1600 / 16 | 1600 |
| c004 | c003 | static | trained | 0.3913 | 0.3910 | 0.3916 | +0.0048 | 1600 / 16 | 1600 |
| c005 | c004 | static | trained | 0.4005 | 0.3994 | 0.4017 | +0.0140 | 1600 / 16 | 1600 |
| c006 | c005 | static | trained | 0.4147 | 0.4108 | 0.4187 | +0.0283 | 1600 / 16 | 1600 |
| c007 | c006 | static | trained | 0.4043 | 0.3933 | 0.4152 | +0.0178 | 1600 / 16 | 1600 |
| c008 | c007 | static | trained | 0.4099 | 0.4423 | 0.3776 | +0.0235 | 1600 / 16 | 2000 |
| c009 | c008 | schedule | trained | 0.4091 | 0.4042 | 0.4140 | +0.0226 | 1600 / 16 | 1600 |

## Confirmation (top 3, seeds [1001, 1002, 1003, 1004], paired with the baseline replicates of the same seeds)

| id | D | per-seed D | SE_train | SE_item | CI (Bonferroni) | scale D | move D |
|---|---|---|---|---|---|---|---|
| c006 | +0.0297 | +0.045, +0.064, -0.002, +0.013 | 0.0175 | 0.0103 | [-0.0189, +0.0783] | +0.0395 | +0.0200 |
| c008 | -0.0259 | +0.011, -0.023, -0.057, -0.036 | 0.0175 | 0.0106 | [-0.0749, +0.0230] | -0.0368 | -0.0151 |
| c009 | +0.0250 | +0.040, +0.040, -0.005, +0.025 | 0.0175 | 0.0109 | [-0.0243, +0.0744] | +0.0326 | +0.0174 |

**Selection**: c000 — no confirmed candidate passed: fixed baseline (delta 0.0247).

## Test (P3, formal seeds [2000, 2001, 2002]; existing set with usage history, not fresh)

- Base: 0.0037; fixed baseline runs: 0.3567, 0.3539, 0.3949

## Cost

| item | GPU-h |
|---|---|
| confirm_eval | 3.34 |
| confirm_probe | 0.69 |
| confirm_train | 7.91 |
| final_eval | 1.11 |
| final_probe | 0.00 |
| final_train | 1.79 |
| search_and_noise_eval | 4.19 |
| search_and_noise_probe | 0.50 |
| search_and_noise_train | 8.94 |

Builder Agent: 11 calls, $7.93. Disk at the end: 15.32 GB.

## Failures

- none
