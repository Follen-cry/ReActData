"""Filesystem operations for a candidate submission, separate from search decisions."""
import difflib
import os
import shutil
from experiment.runtime.io import sha256_tree, sha256_json, write_json
from experiment.runtime.journal import OperationJournal


def prepare_workspace(root, rnd, cid, parent, source, files, last_memory):
    directory = os.path.join(root, "rounds", f"{rnd:03d}", cid)
    start, work = os.path.join(directory, "work_start"), os.path.join(directory, "work")
    if not os.path.exists(start):
        staging = start + ".tmp"
        os.makedirs(staging, exist_ok=True)
        for name in files:
            path = os.path.join(source, name)
            if os.path.isfile(path):
                shutil.copy2(path, os.path.join(staging, name))
        if last_memory and os.path.isfile(last_memory):
            shutil.copy2(last_memory, os.path.join(staging, "memory.md"))
        os.replace(staging, start)
    if not os.path.exists(work):
        shutil.copytree(start, work)
    return directory, work


def validate_submission(contract, work, directory):
    digest = sha256_json({"entries": [(name, os.lstat(os.path.join(work, name)).st_mode) for name in sorted(os.listdir(work))],
                         "content": sha256_tree([os.path.join(work, f) for f in contract.files
                                                 if os.path.isfile(os.path.join(work, f)) and not os.path.islink(os.path.join(work, f))])})
    def check():
        result = contract.check_candidate(work, os.path.join(directory, "check_work"))
        result["errors"] += contract.proposal_errors(work)
        return result
    cacheable = all(os.path.isfile(os.path.join(work, name)) and not os.path.islink(os.path.join(work, name)) for name in os.listdir(work))
    result = OperationJournal().call(os.path.join(directory, "validation", digest), check, retry_safe=True) if cacheable else check()
    write_json(os.path.join(directory, "checks.json"), result)
    return result


def write_diff(directory, work, files, parent, cid):
    diff = []
    for name in files:
        a, b = os.path.join(directory, "work_start", name), os.path.join(work, name)
        before = []
        if os.path.isfile(a):
            with open(a) as f:
                before = f.readlines()
        after = []
        if os.path.isfile(b):
            with open(b) as f:
                after = f.readlines()
        diff.extend(difflib.unified_diff(before, after, f"{parent}/{name}", f"{cid}/{name}"))
    with open(os.path.join(directory, "patch.diff"), "w") as f:
        f.writelines(diff)


def publish_submission(root, cid, work, files):
    destination = os.path.join(root, "candidates", cid, "code")
    hashes = lambda d: sha256_tree([os.path.join(d, f) for f in files if os.path.isfile(os.path.join(d, f))])
    if os.path.isdir(destination):
        if hashes(destination) != hashes(work):
            raise RuntimeError("published candidate differs from submission on recovery")
        return destination
    staging = destination + ".tmp"
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    shutil.copytree(work, staging, ignore=shutil.ignore_patterns("*.pyc", "__pycache__"))
    os.replace(staging, destination)
    return destination
