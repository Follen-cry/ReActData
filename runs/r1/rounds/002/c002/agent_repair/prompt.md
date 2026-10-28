Repair step: the Runner's checks rejected the files now in /work. Fix every error below and deliver a valid candidate (same idea if possible). Errors:
[
 "simulation: PluginError: plugin error in after_step:\nTraceback (most recent call last):\n  File \"/runner/plugin_worker.py\", line 54, in main\n    summary = jsonable(analysis.update(msg[\"step\"], msg[\"rows\"], msg[\"step_info\"], msg[\"probe_results\"]),\n  File \"/cand/analysis.py\", line 27, in update\n    \"E\": sum(it[\"sub\"][\"E_cell\"] for it in p[\"items\"]) / n,\n  File \"/cand/analysis.py\", line 27, in <genexpr>\n    \"E\": sum(it[\"sub\"][\"E_cell\"] for it in p[\"items\"]) / n,\nKeyError: 'E_cell'\n"
]
Plugin log tail:
