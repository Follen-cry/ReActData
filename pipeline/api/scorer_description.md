# How dev outputs are scored (description only; the scorer itself is not accessible)

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

Dev = 50 shape_color_then_scale items (same task as training, new instances) + 50 shape_color_then_move items
(a related task with a different second rule, absent from training: cross-task development feedback). The dev score
is the equal-weight mean of the two task means. The final test (never shown) is shape_color_then_move only.
