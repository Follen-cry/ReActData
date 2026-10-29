"""Compatibility statistical API; reactdata supplies the historical item grouping."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment.workflow.statistics import B, BOOT_SEED, ORDER, _z, item_means, run_means, Statistics

def task_of(item_id):
    return item_id.rsplit("_", 1)[0]

_statistics = Statistics(task_of)
agg = _statistics.agg
paired = _statistics.paired
null_delta = _statistics.null_delta
select = _statistics.select
training_noise = _statistics.training_noise
confirm_compare = _statistics.confirm_compare
