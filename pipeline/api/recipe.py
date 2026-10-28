"""Recipe contract (fixed, reactdata-v4). A recipe is one training row chosen by a databuilder.

    {"clip_id": str, "context_frames": [int, ...], "template_id": str, "target_frame": int (optional, default FINAL)}

* The first input image is always the clip's F0 (PNG endpoint); it is also the only clean VAE condition.
* ``context_frames`` are extra input images that go through the ViT only: frame indices 1..59 of the clip's
  mp4 (16 fps, 60 frames), strictly increasing, at most MAX_CONTEXT_PER_ROW of them.
* ``target_frame`` (v4): the frame the row is trained to generate, 1..FINAL (= 59, the clip's final frame, the default).
  The final frame is the PNG endpoint; earlier frames are colour-corrected mp4 frames. Context frames must be earlier
  than the target (forward prediction only). At least MIN_FINAL_ROWS_PER_STEP rows of every step target FINAL, so the
  test-time mapping (F0 -> final frame) is always trained. Rows with a non-final target must use a template with
  {target_spec} and without {wrapped_description} (text_slots.py).
* ``template_id`` names a template of the candidate's text_templates.yaml (see text_slots.py).
* One optimizer step consumes exactly ROWS_PER_STEP recipes; all input images of a step (F0s included) are
  at most MAX_VIT_IMAGES_PER_STEP.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

N_FRAMES = 60
FPS = 16
ROWS_PER_STEP = 16
MAX_CONTEXT_PER_ROW = 2
MAX_VIT_IMAGES_PER_STEP = 32
FINAL = N_FRAMES - 1
MIN_FINAL_ROWS_PER_STEP = 8
RECIPE_KEYS = {"clip_id", "context_frames", "template_id"}
OPTIONAL_KEYS = {"target_frame"}


class RecipeError(ValueError):
    pass


def check_recipe(r: Any, clip_ids, template_ids) -> Dict[str, Any]:
    """Returns a normalized copy or raises RecipeError."""
    if not isinstance(r, dict) or not RECIPE_KEYS <= set(r) or set(r) - RECIPE_KEYS - OPTIONAL_KEYS:
        raise RecipeError(f"recipe must be a dict with the keys {sorted(RECIPE_KEYS)} (+ optional {sorted(OPTIONAL_KEYS)}): {r!r}"[:300])
    cid, ctx, tid = r["clip_id"], r["context_frames"], r["template_id"]
    tgt = r.get("target_frame", FINAL)
    if isinstance(tgt, bool) or not isinstance(tgt, int) or not 1 <= tgt <= FINAL:
        raise RecipeError(f"target_frame must be an int in 1..{FINAL}: {tgt!r}"[:200])
    if not isinstance(cid, str) or cid not in clip_ids:
        raise RecipeError(f"unknown clip_id {cid!r}"[:200])
    if not isinstance(tid, str) or tid not in template_ids:
        raise RecipeError(f"unknown template_id {tid!r}"[:200])
    if not isinstance(ctx, list) or any(isinstance(t, bool) or not isinstance(t, int) for t in ctx):
        raise RecipeError(f"context_frames must be a list of int: {ctx!r}"[:200])
    if len(ctx) > MAX_CONTEXT_PER_ROW:
        raise RecipeError(f"at most {MAX_CONTEXT_PER_ROW} context frames per row, got {len(ctx)}")
    if any(not 1 <= t <= N_FRAMES - 1 for t in ctx) or any(b <= a for a, b in zip(ctx, ctx[1:])):
        raise RecipeError(f"context_frames must be strictly increasing in 1..{N_FRAMES - 1}: {ctx}")
    if any(t >= tgt for t in ctx):
        raise RecipeError(f"context frames must be earlier than the target frame {tgt} (forward prediction only): {ctx}")
    return {"clip_id": cid, "context_frames": list(ctx), "template_id": tid, "target_frame": tgt}


def check_batch(batch: Any, clip_ids, template_ids) -> List[Dict[str, Any]]:
    if not isinstance(batch, list) or len(batch) != ROWS_PER_STEP:
        raise RecipeError(f"next_batch must return a list of {ROWS_PER_STEP} recipes, got "
                          f"{type(batch).__name__} of length {len(batch) if isinstance(batch, list) else '?'}")
    out = [check_recipe(r, clip_ids, template_ids) for r in batch]
    n_img = sum(1 + len(r["context_frames"]) for r in out)
    if n_img > MAX_VIT_IMAGES_PER_STEP:
        raise RecipeError(f"{n_img} input images in one step > {MAX_VIT_IMAGES_PER_STEP}")
    n_final = sum(r["target_frame"] == FINAL for r in out)
    if n_final < MIN_FINAL_ROWS_PER_STEP:
        raise RecipeError(f"only {n_final} rows target the final frame; at least {MIN_FINAL_ROWS_PER_STEP} per step")
    return out


def vit_images(batch: Iterable[Dict[str, Any]]) -> int:
    return sum(1 + len(r["context_frames"]) for r in batch)
