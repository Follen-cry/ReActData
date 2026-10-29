"""Trusted source inventory; independent of experiment data and output paths."""
import os
from .io import sha256_file, sha256_json


def implementation_files(source_root):
    return sorted(os.path.join(directory, name)
                  for package in ("runner", "experiment", "tasks")
                  for directory, _, names in os.walk(os.path.join(source_root, package))
                  for name in names if name.endswith(".py"))


def implementation_manifest(source_root):
    return {os.path.relpath(path, source_root).replace(os.sep, "/"): sha256_file(path)
            for path in implementation_files(source_root)}


def implementation_hash(source_root):
    return sha256_json(implementation_manifest(source_root))
