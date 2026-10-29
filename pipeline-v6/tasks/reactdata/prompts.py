"""Frozen reactdata prompt wording and evidence workflow."""
from experiment.contracts import PromptBundle


class PromptBuilder:
    def build(self, context, cid, rnd, idx, start, siblings, instructions):
        b = context.budget_info or {}
        budget_line = (f"Search budget: this is new candidate {b['new_index']} of {b['new_total']} in this run ({b['left_after_this']} left after "
                       f"this one); round {b['round']} of {b['last_round']} ({b['rounds_left_after_this']} rounds left after this one); about "
                       f"{b['wall_h_left']} h of wall time left. The search does not stop early when scores plateau.") if b else ""
        notes = [f"You are writing candidate {cid} (round {rnd}, {idx + 1} of {context.per_round} this round). Your /work starts "
                 f"as a copy of {start} (the best valid program by dev mean so far).", budget_line,
                 f"Training budget per run: {context.steps} optimizer steps x 16 rows from the common initialization; seed {context.P['training'].get('screen_seed', context.P['training'].get('search_run_seed'))}.",
                 f"Dev generation seeds S = {context.S}; dev numbers are selection-biased; the test set is never shown."]
        if context.state.get("noise"):
            nz = context.state["noise"]
            notes.append(f"Training noise (measured): the fixed baseline trained with {nz['k']} different run seeds scored "
                         f"{', '.join(f'{x:.3f}' for x in nz['run_scores'])} on dev (sd {nz['sd_train']:.3f}). A screening result is ONE run: "
                         f"differences smaller than about 2 x {nz['sd_train']:.3f} are indistinguishable from noise. After the search the "
                         f"top {context.top_k} candidates are retrained on {len(context.confirm_seeds)} more seeds and compared with the baseline "
                         "on the same seeds; only that decides. Do not spend candidates on replicates (changing only a seed / salt / "
                         "shuffle is a duplicate and is not trained).")
        notes = [x for x in notes if x]
        if siblings:
            notes.append("Other candidates already written this round (their code is in candidates/): " + ", ".join(siblings)
                         + ". Propose a clearly different idea.")
        full_prompt = (instructions +
                       f"\n\n---\nSession: write candidate {cid}. Follow the required workflow: read /input/report.md, "
                       "/input/results.json, /input/dev_items/, /input/analysis_traces/ and /input/api/ first, then write "
                       "proposal.md with all required sections, then edit the files in /work. Finish with a short summary.")
        cont_prompt = (f"Next step of the same search: write candidate {cid} (round {rnd}). {budget_line} The Runner has refreshed /input "
                       f"(new results, failure cases and candidates) and /work now starts as a copy of {start}. Follow the required "
                       "workflow of BUILDER_AGENT.md: first read /input/report.md, /input/results.json, /input/dev_items/ and "
                       "/input/analysis_traces/ completely, then write proposal.md with all required sections (Evidence citing "
                       "dev items by id), then edit the files in /work; finish with a short summary.")

        return PromptBundle(notes, full_prompt, cont_prompt)

    def repair(self, prompt, checks, resumed):
        import json
        detail = json.dumps(checks["errors"], indent=1)[:6000] + ("\nPlugin log tail:\n" + checks.get("plugin_log_tail", ""))[:3000]
        if resumed:
            return "Repair step: the Runner's checks rejected the files now in /work. Fix every error below and deliver a valid candidate (same idea if possible). Errors:\n" + detail
        return prompt + "\n\n---\nThe Runner's checks rejected the files now in /work. Fix every error below and deliver a valid candidate (same idea if possible). Errors:\n" + detail
