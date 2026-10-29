"""Render only explicitly authorized outputs, without access to run records."""
from dataclasses import dataclass
import os
import shutil
from experiment.runtime.io import write_json


@dataclass(frozen=True)
class OutputFile:
    name: str
    kind: str
    value: object


@dataclass(frozen=True)
class AuthorizedEvidence:
    destination: str
    files: tuple[OutputFile, ...]


def render_authorized(bundle):
    root = os.path.abspath(bundle.destination)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    for item in bundle.files:
        path = os.path.abspath(os.path.join(root, item.name))
        if os.path.commonpath([root, path]) != root or path == root:
            raise ValueError("evidence output escapes the input directory")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if item.kind == "json":
            write_json(path, item.value)
        elif item.kind == "text":
            with open(path, "w") as f:
                f.write(item.value)
        elif item.kind == "copy":
            shutil.copy2(item.value, path)
        elif item.kind == "link":
            try:
                os.link(item.value, path)
            except OSError:
                shutil.copy2(item.value, path)
        else:
            raise ValueError(f"unknown evidence output kind {item.kind}")
    return root
