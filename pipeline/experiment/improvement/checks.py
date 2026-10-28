import ast
import os
from typing import List


def file_errors(candidate, required, optional, max_bytes):
    errors = []
    for f in required:
        path = os.path.join(candidate, f)
        if not os.path.isfile(path):
            errors.append(f"missing {f}")
        elif os.path.getsize(path) > max_bytes:
            errors.append(f"{f} larger than {max_bytes} bytes")
    extra = sorted(set(os.listdir(candidate)) - set(required) - set(optional))
    if extra:
        errors.append(f"unexpected files in the candidate directory: {extra}")
    return errors

BANNED_CALLS = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input", "breakpoint",
                "help", "memoryview", "exit", "quit"}

def static_scan(path: str, allowed_imports) -> List[str]:
    errs = []
    try:
        tree = ast.parse(open(path).read(), path)
    except SyntaxError as e:
        return [f"{os.path.basename(path)}: syntax error: {e}"]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for n in names:
                if n.split(".")[0] not in allowed_imports:
                    errs.append(f"{os.path.basename(path)}:{node.lineno}: import {n} not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_CALLS:
            errs.append(f"{os.path.basename(path)}:{node.lineno}: use of {node.id} not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in ("__init__", "__name__"):
            errs.append(f"{os.path.basename(path)}:{node.lineno}: dunder attribute {node.attr} not allowed")
    return errs
