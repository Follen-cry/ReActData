# How dev outputs are scored (description only; the scorers themselves are not accessible)

reactdata-v6 has three tasks, each with its own deterministic computer-vision rule. Every item: F0 + the original task description (evaluation wrapper) -> one generated 512x512 image, compared with the ground-truth final frame.

## shape_color_then_scale and shape_color_then_move (dev_scale_*, dev_move_*)

Both tasks show a 2x3 analogy grid: top row A -> B -> C (worked example), bottom row D -> E -> F. The input F0 shows
A, B, C and D; the model must generate the full final frame, i.e. add **E** (D recoloured the way A became B) and
**F** (E transformed the way B became C: **rescaled** in shape_color_then_scale, **moved up/down** in
shape_color_then_move), keeping A, B, C, D unchanged. A deterministic computer-vision rule compares the generated
512x512 image with the ground-truth final frame cell by cell.

**Total score** (0..1) = 0.35 x E_cell + 0.40 x F_cell + 0.25 x F_transform, multiplied by gates.

- **E_cell** — an object at E's cell with GT E's glyph and size (centred-mask overlap) and GT E's colour.
- **F_cell** — the same for F (its position, glyph + size and colour against GT F).
- **F_transform** — F relative to the generated E: the size ratio (scale task) or the vertical offset (move task)
  must match the ground truth; leaving F the same size as E / at E's height scores 0 when the task asks for a change.

**Gates**: A, B, C, D must still be there (they earn no credit, but if they are damaged the score is ramped down to
0); an (almost) blank image or very little foreground is capped; a repainted background is capped at 0.15.
Missing, unreadable, black or flat images score 0 and count as invalid. A copy of the input scores 0.

Error categories reported per item: `E_missing`, `F_missing`, `E_wrong_colour`, `F_wrong_colour`, `E_wrong`, `F_wrong`
(position / glyph / size), `F_wrong_transform`, `context_damaged`, `blank_or_repainted`, `extra_objects`.

## 2d_geometric_transformation (dev_geo_*)

**Total score** (0..1) = 0.50 x rotation_angle + 0.357 x position_alignment + 0.143 x shape_fidelity,
multiplied by global gates.

- **rotation_angle** — the rotation that best maps the generated polygon's silhouette onto the ground-truth polygon
  (exhaustive search over 360 degrees). Full credit within about 8 degrees, zero at 45 degrees or more.
- **position_alignment** — distance between the generated polygon's centroid and the ground-truth centroid. Full
  credit within about a tenth of the polygon's side; zero when the polygon is no closer to the target than it was in
  F0 (leaving the polygon where it started scores zero).
- **shape_fidelity** — the generated polygon must be the same rigid shape as the polygon in F0 (best-fit silhouette
  overlap and area ratio): reaching the target by deforming the shape is penalized.

**Gates** (multiplicative): an (almost) blank image or one with very little foreground is capped; a repainted
background is capped; a polygon of the wrong colour halves the score; an object that is not the polygon (for
example only the dashed outline) gets no pose credit; a dashed target outline that is still drawn costs up to 25 %.
Missing, unreadable, black or flat images score 0 and are counted as invalid.

Error categories reported per item (from the sub-scores and gates): `no_shape`, `blank_or_sparse`, `wrong_colour`,
`wrong_position` (< 0.5), `wrong_rotation` (< 0.5), `deformed_shape` (< 0.5), `outline_left`.

Known limits: near-symmetric shapes make the rotation score lenient; the rotation centre (pivot) is not checked, only
the end pose.

## Dev and test

Dev = 50 items per task (new instances of the three training tasks). The dev score is the equal-weight mean of the three task means. A copy of the input scores 0 on scale / move and about 0.12 on geo; the ground truth scores 1.0 (geo 0.997). The final test (never shown) has the same three-task composition.
