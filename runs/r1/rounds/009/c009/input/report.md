# Round input report

All dev numbers below were used for selection and are therefore selection-biased; they are not confirmatory evidence. The test set is never shown.

Files: `results.json` (all candidates: dev scores, comparisons, costs, training rows, failure cases), `dev_items/<id>.json` (every dev item of every trained program: score, sub-scores, error categories, and the baseline replicate mean / sd and the difference to it), `dev_items/baseline_replicates.json`, `analysis_traces/<id>.json` (what each program's own analysis.py reported every 10 training steps, and its probe results), `failures/`, `candidates/`, `train_examples/`, `api/`.

- You are writing candidate c009 (round 9, 1 of 1 this round). Your /work starts as a copy of c008 (the best valid program by dev mean so far).
- Search budget: this is new candidate 9 of 10 in this run (1 left after this one); round 9 of 10 (1 rounds left after this one); about 4.47 h of wall time left. The search does not stop early when scores plateau.
- Training budget per run: 100 optimizer steps x 16 rows from the common initialization; seed 1000.
- Dev generation seeds S = [0]; dev numbers are selection-biased; the test set is never shown.
- Training noise (measured): the fixed baseline trained with 5 different run seeds scored 0.391, 0.355, 0.396, 0.420, 0.371 on dev (sd 0.025). A screening result is ONE run: differences smaller than about 2 x 0.025 are indistinguishable from noise. After the search the top 3 candidates are retrained on 4 more seeds and compared with the baseline on the same seeds; only that decides. Do not spend candidates on replicates (changing only a seed / salt / shuffle is a duplicate and is not trained).

## Training pool

```json
{
 "task": "shape_color_then_scale",
 "n_clips": 16,
 "frames_per_clip": 60,
 "fps": 16,
 "inner_probe_clips": 32,
 "pixel_statistics": {
  "frame_change": "per t in 0..58: pixels of frame t+1 vs t (mp4, 256x256 INTER_AREA) whose max-channel |diff| > 40",
  "fg_area": "per frame: pixels whose BGR L2 distance from the border-median background > 25",
  "code_sha256": "8467f6a6a4019d23d2a28142790623d37b4c68b79e568bf04cb75a042c64d3c0"
 },
 "dev": "50 shape_color_then_scale (same task, new instances) + 50 shape_color_then_move (related task with a different second rule, not in the training pool: cross-task development feedback); dev score = equal-weight mean of the two task means"
}
```

Example frames of three training clips are in `train_examples/` (frame index in the file name; 16 fps, 60 frames).

## Results so far

### c000 (fixed_baseline, trained)
- dev score (equal-weight mean of the two task means), mean over generation seeds [0]: **0.3908** (per seed {0: 0.3908}); invalid images 0
  - shape_color_then_move: 0.4084; sub-scores {'E_cell': 0.6388, 'F_cell': 0.3491, 'F_transform': 0.1809}; error categories (seed 0): {'F_wrong': 14, 'F_wrong_transform': 33, 'context_damaged': 5, 'E_wrong_colour': 16, 'F_wrong_colour': 10, 'F_missing': 8, 'extra_objects': 9}
  - shape_color_then_scale: 0.3731; sub-scores {'E_cell': 0.5221, 'F_cell': 0.1155, 'F_transform': 0.5767}; error categories (seed 0): {'F_wrong': 36, 'F_wrong_transform': 18, 'F_wrong_colour': 8, 'E_wrong_colour': 20, 'extra_objects': 3, 'context_damaged': 3, 'E_wrong': 3, 'F_missing': 1}
- failure cases: `failures/c000/` (see results.json for their scores)
- training rows: 1600 rows, 16 unique clips, 0 rows with a non-final target, 0 rows with context frames
- training flow-matching loss, mean per 20 steps: [0.0095, 0.01082, 0.00558, 0.00485, 0.00505]
- cost: {'rows': 1600, 'unique_clips': 16, 'repeated_rows': 1584, 'max_clip_uses': 100, 'vit_images': 1600, 'vit_tokens': 409600, 'seq_tokens': 783330, 'train_gpu_s': 2142.3215341567993, 'probe_gpu_s': 0.0, 'eval_gpu_s': 1002.3733425140381, 'plugin_wall_s': 0.0653839111328125}
- code: `candidates/c000/`

### c001 (agent, trained)
- dev score (equal-weight mean of the two task means), mean over generation seeds [0]: **0.3432** (per seed {0: 0.3432}); invalid images 0
  - shape_color_then_move: 0.336; sub-scores {'E_cell': 0.5458, 'F_cell': 0.2652, 'F_transform': 0.1556}; error categories (seed 0): {'E_wrong_colour': 18, 'F_wrong_colour': 8, 'F_wrong_transform': 29, 'context_damaged': 4, 'F_wrong': 14, 'F_missing': 14, 'extra_objects': 9, 'blank_or_repainted': 1, 'E_missing': 1, 'E_wrong': 3}
  - shape_color_then_scale: 0.3505; sub-scores {'E_cell': 0.5495, 'F_cell': 0.1164, 'F_transform': 0.4462}; error categories (seed 0): {'F_missing': 11, 'F_wrong_colour': 4, 'F_wrong': 29, 'F_wrong_transform': 16, 'E_wrong_colour': 17, 'E_wrong': 3, 'blank_or_repainted': 2, 'context_damaged': 5, 'E_missing': 1, 'extra_objects': 3}
- training rows: 1600 rows, 16 unique clips, 560 rows with a non-final target, 0 rows with context frames
- training flow-matching loss, mean per 20 steps: [0.00851, 0.00771, 0.00724, 0.00515, 0.00627]
- vs fixed baseline: D = -0.0432, 95% bootstrap CI [-0.0721, -0.0155] (selection-biased dev numbers); per task: dev_move -0.0582 [-0.0976, -0.0221]; dev_scale -0.0283 [-0.0720, +0.0145]
- cost: {'rows': 1600, 'unique_clips': 16, 'repeated_rows': 1584, 'max_clip_uses': 100, 'vit_images': 1600, 'vit_tokens': 409600, 'seq_tokens': 775203, 'train_gpu_s': 2372.561619758606, 'probe_gpu_s': 201.26220321655273, 'eval_gpu_s': 1002.2044086456299, 'plugin_wall_s': 0.10365843772888184}
- code: `candidates/c001/`

### c002 (agent, trained)
- dev score (equal-weight mean of the two task means), mean over generation seeds [0]: **0.2974** (per seed {0: 0.2974}); invalid images 0
  - shape_color_then_move: 0.3071; sub-scores {'E_cell': 0.4664, 'F_cell': 0.248, 'F_transform': 0.1787}; error categories (seed 0): {'E_wrong_colour': 21, 'F_missing': 12, 'F_wrong_colour': 10, 'F_wrong': 14, 'F_wrong_transform': 30, 'extra_objects': 10, 'E_wrong': 6}
  - shape_color_then_scale: 0.2876; sub-scores {'E_cell': 0.3792, 'F_cell': 0.0862, 'F_transform': 0.4817}; error categories (seed 0): {'F_wrong': 35, 'F_wrong_transform': 17, 'E_wrong_colour': 22, 'F_wrong_colour': 5, 'E_wrong': 6, 'F_missing': 7, 'E_missing': 1, 'context_damaged': 2}
- training rows: 1600 rows, 16 unique clips, 0 rows with a non-final target, 480 rows with context frames
- training flow-matching loss, mean per 20 steps: [0.01352, 0.00862, 0.00541, 0.00569, 0.00567]
- vs fixed baseline: D = -0.0891, 95% bootstrap CI [-0.1189, -0.0595] (selection-biased dev numbers); per task: dev_move -0.0871 [-0.1225, -0.0556]; dev_scale -0.0911 [-0.1404, -0.0413]
- cost: {'rows': 1600, 'unique_clips': 16, 'repeated_rows': 1584, 'max_clip_uses': 100, 'vit_images': 2080, 'vit_tokens': 532480, 'seq_tokens': 906540, 'train_gpu_s': 2391.4838218688965, 'probe_gpu_s': 197.50831604003906, 'eval_gpu_s': 997.7278442382812, 'plugin_wall_s': 0.10003018379211426}
- code: `candidates/c002/`

### c003 (agent, trained)
- dev score (equal-weight mean of the two task means), mean over generation seeds [0]: **0.35** (per seed {0: 0.35}); invalid images 0
  - shape_color_then_move: 0.3344; sub-scores {'E_cell': 0.5462, 'F_cell': 0.2613, 'F_transform': 0.1547}; error categories (seed 0): {'E_wrong_colour': 19, 'F_wrong': 15, 'F_wrong_transform': 33, 'context_damaged': 3, 'extra_objects': 11, 'F_wrong_colour': 11, 'F_missing': 10, 'E_wrong': 3}
  - shape_color_then_scale: 0.3655; sub-scores {'E_cell': 0.5508, 'F_cell': 0.1196, 'F_transform': 0.4996}; error categories (seed 0): {'F_missing': 8, 'F_wrong_colour': 5, 'F_wrong': 31, 'F_wrong_transform': 15, 'E_wrong_colour': 17, 'E_wrong': 3, 'blank_or_repainted': 2, 'context_damaged': 4, 'E_missing': 1, 'extra_objects': 2}
- training rows: 1600 rows, 16 unique clips, 320 rows with a non-final target, 0 rows with context frames
- training flow-matching loss, mean per 20 steps: [0.00849, 0.00774, 0.00721, 0.0057, 0.00613]
- vs fixed baseline: D = -0.0365, 95% bootstrap CI [-0.0645, -0.0099] (selection-biased dev numbers); per task: dev_move -0.0598 [-0.0973, -0.0243]; dev_scale -0.0132 [-0.0547, +0.0270]
- cost: {'rows': 1600, 'unique_clips': 16, 'repeated_rows': 1584, 'max_clip_uses': 100, 'vit_images': 1600, 'vit_tokens': 409600, 'seq_tokens': 778737, 'train_gpu_s': 2356.6756324768066, 'probe_gpu_s': 193.34750652313232, 'eval_gpu_s': 998.5490064620972, 'plugin_wall_s': 0.09228682518005371}
- code: `candidates/c003/`

### c004 (agent, trained)
- dev score (equal-weight mean of the two task means), mean over generation seeds [0]: **0.3913** (per seed {0: 0.3913}); invalid images 0
  - shape_color_then_move: 0.3916; sub-scores {'E_cell': 0.5795, 'F_cell': 0.3283, 'F_transform': 0.2298}; error categories (seed 0): {'E_wrong_colour': 19, 'F_wrong': 17, 'F_wrong_transform': 31, 'F_wrong_colour': 8, 'extra_objects': 10, 'F_missing': 8, 'context_damaged': 2, 'E_wrong': 1}
  - shape_color_then_scale: 0.391; sub-scores {'E_cell': 0.5156, 'F_cell': 0.1531, 'F_transform': 0.5973}; error categories (seed 0): {'F_missing': 2, 'F_wrong': 34, 'F_wrong_colour': 7, 'F_wrong_transform': 15, 'E_wrong_colour': 20, 'E_wrong': 3, 'extra_objects': 2, 'context_damaged': 2}
- training rows: 1600 rows, 16 unique clips, 600 rows with a non-final target, 0 rows with context frames
- training flow-matching loss, mean per 20 steps: [0.0081, 0.0094, 0.0086, 0.00566, 0.00506]
- vs fixed baseline: D = +0.0048, 95% bootstrap CI [-0.0204, +0.0308] (selection-biased dev numbers); per task: dev_move -0.0026 [-0.0326, +0.0256]; dev_scale +0.0123 [-0.0304, +0.0552]
- cost: {'rows': 1600, 'unique_clips': 16, 'repeated_rows': 1584, 'max_clip_uses': 101, 'vit_images': 1600, 'vit_tokens': 409600, 'seq_tokens': 774660, 'train_gpu_s': 2357.247815132141, 'probe_gpu_s': 193.55410766601562, 'eval_gpu_s': 996.8511037826538, 'plugin_wall_s': 0.1049494743347168}
- code: `candidates/c004/`

### c005 (agent, trained)
- dev score (equal-weight mean of the two task means), mean over generation seeds [0]: **0.4005** (per seed {0: 0.4005}); invalid images 0
  - shape_color_then_move: 0.4017; sub-scores {'E_cell': 0.6008, 'F_cell': 0.3448, 'F_transform': 0.2138}; error categories (seed 0): {'E_wrong_colour': 18, 'F_wrong': 17, 'F_wrong_transform': 31, 'F_wrong_colour': 7, 'context_damaged': 4, 'F_missing': 8, 'extra_objects': 10}
  - shape_color_then_scale: 0.3994; sub-scores {'E_cell': 0.5183, 'F_cell': 0.1867, 'F_transform': 0.573}; error categories (seed 0): {'F_missing': 1, 'F_wrong': 36, 'E_wrong_colour': 19, 'F_wrong_transform': 18, 'E_wrong': 4, 'F_wrong_colour': 6, 'blank_or_repainted': 1, 'context_damaged': 4, 'extra_objects': 3}
- failure cases: `failures/c005/` (see results.json for their scores)
- training rows: 1600 rows, 16 unique clips, 600 rows with a non-final target, 0 rows with context frames
- training flow-matching loss, mean per 20 steps: [0.00801, 0.00936, 0.00849, 0.0056, 0.005]
- vs fixed baseline: D = +0.0140, 95% bootstrap CI [-0.0142, +0.0422] (selection-biased dev numbers); per task: dev_move +0.0075 [-0.0249, +0.0421]; dev_scale +0.0206 [-0.0240, +0.0660]
- cost: {'rows': 1600, 'unique_clips': 16, 'repeated_rows': 1584, 'max_clip_uses': 101, 'vit_images': 1600, 'vit_tokens': 409600, 'seq_tokens': 774660, 'train_gpu_s': 2354.3251876831055, 'probe_gpu_s': 193.37354278564453, 'eval_gpu_s': 998.5873308181763, 'plugin_wall_s': 0.09669160842895508}
- code: `candidates/c005/`

### c006 (agent, trained)
- dev score (equal-weight mean of the two task means), mean over generation seeds [0]: **0.4147** (per seed {0: 0.4147}); invalid images 0
  - shape_color_then_move: 0.4187; sub-scores {'E_cell': 0.6481, 'F_cell': 0.3613, 'F_transform': 0.1894}; error categories (seed 0): {'F_wrong': 17, 'F_wrong_transform': 33, 'context_damaged': 3, 'E_wrong_colour': 14, 'F_wrong_colour': 6, 'extra_objects': 9, 'F_missing': 8, 'E_wrong': 2}
  - shape_color_then_scale: 0.4108; sub-scores {'E_cell': 0.6434, 'F_cell': 0.1507, 'F_transform': 0.5012}; error categories (seed 0): {'F_wrong': 31, 'F_wrong_colour': 8, 'F_wrong_transform': 19, 'E_wrong_colour': 11, 'E_wrong': 4, 'F_missing': 4, 'blank_or_repainted': 2, 'context_damaged': 6, 'extra_objects': 2}
- failure cases: `failures/c006/` (see results.json for their scores)
- training rows: 1600 rows, 16 unique clips, 800 rows with a non-final target, 0 rows with context frames
- training flow-matching loss, mean per 20 steps: [0.00953, 0.00845, 0.0078, 0.00505, 0.0046]
- vs fixed baseline: D = +0.0283, 95% bootstrap CI [+0.0013, +0.0574] (selection-biased dev numbers); per task: dev_move +0.0245 [-0.0066, +0.0596]; dev_scale +0.0321 [-0.0124, +0.0792]
- cost: {'rows': 1600, 'unique_clips': 16, 'repeated_rows': 1584, 'max_clip_uses': 100, 'vit_images': 1600, 'vit_tokens': 409600, 'seq_tokens': 771731, 'train_gpu_s': 2343.9556379318237, 'probe_gpu_s': 193.26519012451172, 'eval_gpu_s': 998.0954465866089, 'plugin_wall_s': 0.10416650772094727}
- code: `candidates/c006/`

### c007 (agent, trained)
- dev score (equal-weight mean of the two task means), mean over generation seeds [0]: **0.4043** (per seed {0: 0.4043}); invalid images 0
  - shape_color_then_move: 0.4152; sub-scores {'E_cell': 0.6408, 'F_cell': 0.3617, 'F_transform': 0.185}; error categories (seed 0): {'F_wrong': 17, 'F_wrong_transform': 33, 'context_damaged': 3, 'E_wrong_colour': 15, 'F_wrong_colour': 6, 'extra_objects': 9, 'F_missing': 8, 'E_wrong': 2}
  - shape_color_then_scale: 0.3933; sub-scores {'E_cell': 0.6411, 'F_cell': 0.1127, 'F_transform': 0.4952}; error categories (seed 0): {'F_wrong': 33, 'F_wrong_transform': 19, 'F_wrong_colour': 7, 'E_wrong_colour': 10, 'E_wrong': 4, 'F_missing': 4, 'context_damaged': 5, 'extra_objects': 3, 'blank_or_repainted': 1}
- failure cases: `failures/c007/` (see results.json for their scores)
- training rows: 1600 rows, 16 unique clips, 800 rows with a non-final target, 0 rows with context frames
- training flow-matching loss, mean per 20 steps: [0.00961, 0.00854, 0.00787, 0.0051, 0.00469]
- vs fixed baseline: D = +0.0178, 95% bootstrap CI [-0.0071, +0.0443] (selection-biased dev numbers); per task: dev_move +0.0210 [-0.0049, +0.0497]; dev_scale +0.0146 [-0.0272, +0.0586]
- cost: {'rows': 1600, 'unique_clips': 16, 'repeated_rows': 1584, 'max_clip_uses': 100, 'vit_images': 1600, 'vit_tokens': 409600, 'seq_tokens': 771731, 'train_gpu_s': 2351.8807945251465, 'probe_gpu_s': 193.37922954559326, 'eval_gpu_s': 1000.6990203857422, 'plugin_wall_s': 0.10087156295776367}
- code: `candidates/c007/`

### c008 (agent, trained)
- dev score (equal-weight mean of the two task means), mean over generation seeds [0]: **0.4099** (per seed {0: 0.4099}); invalid images 0
  - shape_color_then_move: 0.3776; sub-scores {'E_cell': 0.6211, 'F_cell': 0.2838, 'F_transform': 0.1868}; error categories (seed 0): {'F_wrong_colour': 12, 'F_wrong_transform': 33, 'context_damaged': 2, 'E_wrong_colour': 16, 'F_wrong': 16, 'extra_objects': 10, 'F_missing': 8, 'E_wrong': 1}
  - shape_color_then_scale: 0.4423; sub-scores {'E_cell': 0.7041, 'F_cell': 0.1569, 'F_transform': 0.5324}; error categories (seed 0): {'F_wrong': 35, 'F_wrong_transform': 21, 'F_wrong_colour': 7, 'E_wrong_colour': 10, 'extra_objects': 2, 'E_wrong': 4, 'context_damaged': 2, 'F_missing': 1}
- failure cases: `failures/c008/` (see results.json for their scores)
- training rows: 1600 rows, 16 unique clips, 800 rows with a non-final target, 400 rows with context frames
- training flow-matching loss, mean per 20 steps: [0.00823, 0.00626, 0.00833, 0.00539, 0.00496]
- vs fixed baseline: D = +0.0235, 95% bootstrap CI [-0.0025, +0.0508] (selection-biased dev numbers); per task: dev_move -0.0166 [-0.0409, +0.0076]; dev_scale +0.0636 [+0.0174, +0.1109]
- cost: {'rows': 1600, 'unique_clips': 16, 'repeated_rows': 1584, 'max_clip_uses': 100, 'vit_images': 2000, 'vit_tokens': 512000, 'seq_tokens': 878697, 'train_gpu_s': 2380.0636310577393, 'probe_gpu_s': 211.77276134490967, 'eval_gpu_s': 1000.0447120666504, 'plugin_wall_s': 0.10583233833312988}
- code: `candidates/c008/`
