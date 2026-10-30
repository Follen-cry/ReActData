# Host instructions and experiment notes

This campaign began 2026-10-29T16:35:22Z, hard deadline 2026-10-30T00:35:22Z. All runs share campaign.json with at most 7 accepted training attempts. Baseline t0001 is the unchanged starter random selection/order. Do not submit an improvement until it is completed and dev has been verified by host.

Search scope once host launches the Agent: perform exactly TWO real improvement submissions, one hypothesis per trial, based on baseline/prior dev feedback. Then select the highest completed dev.mean (tie baseline, then earliest), write selection.txt and stop. Do not exhaust remaining slots. Do not run finalize or inspect test manifests/images/metrics, campaign-level test_sources, or any files under trial/test. Use only task.json sources, existing dev feedback and workspace builder. Do not modify harness, frozen configuration or historical trial artifacts. Only selection and order may change. Keep every hypothesis, request ID, trial ID, actual outcome here. Training seeds and recipe stay fixed.

Host scorer selfcheck: 20 training examples GT >= .95 and copy-F0 <= .05. No test feedback has been obtained. Candidate pool has 160 training sources, each formal run uses 128 distinct rows. All CLI submissions must use the same experiment and ledger; no diagnostic optimizer steps outside CLI.

## Agent continuation: improvement 1
Read protocol, commands, resolved task, original builder, baseline result and all 50 dev records. Host launch is treated as the release to proceed after its baseline verification; baseline is completed and real. Baseline request `baseline`, trial `t0001`, dev.mean=0.23550062360762994, invalid=0/50, F_cell zero=46/50.
Hypothesis: larger final shapes provide more supervised foreground pixels and improve final-cell accuracy in the fixed eight steps. Select 128 of 160 sources by descending target size, seeded random ties, then seeded shuffle. Canonical endpoints and exact text remain unchanged. Expected: higher dev.mean and F_cell mean; refuted for this run if dev.mean does not exceed baseline. This is a heuristic, not evidence that size causes baseline errors.
Request ID: `large-targets-v1`. Original builder preserved as baseline_builder.py. Exactly two improvements authorized; no test access or finalization.
