# Video SSL backend provenance

Bundled from the sibling ReActData pipeline-agent implementation after the first real
InternVL-U experiment and subsequent frame-policy extension. The source is now local
to this method repository; runtime does not import or invoke the sibling checkout.

worker.py separates resolve/audit/train/dev/test/preinit operations. The real
backend preserves ordered training and independent checkpoint reload. The active
task contract now supplies rows, steps, batch, warmup, learning rates, inference
settings and process timeouts. The base-only evaluation branch provides preinit.
Default values retain the first experiment's recipe. Dependencies on the external InternVL-U library, base weights and scorer
are configured explicitly; these assets are not vendored here.

analogy_grid.py retains its original scoring implementation and source attribution.
The method's broader source ancestry is documented in docs/history/upstream-readme.md and LICENSE.
