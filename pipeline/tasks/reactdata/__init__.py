"""Reactdata task entry point; GPU dependencies load only during training."""
from .adapter import ReactDataTask, BACKUP_EXT
from .contract import CAND_FILES

__all__ = ["ReactDataTask", "CAND_FILES", "BACKUP_EXT"]
