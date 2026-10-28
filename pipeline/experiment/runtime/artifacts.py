"""Freeze a validated candidate without interpreting its program format."""
import os
import shutil


def freeze_candidate(store, contract, selected):
    st = store.snapshot().data
    previous = st.get("frozen")
    if previous:
        if previous["cand"] != selected or not contract.verify_frozen(previous):
            raise RuntimeError("frozen program changed; refusing to replace its recorded hash")
        return previous["dir"]
    code = st["candidates"][selected]["code_dir"]
    frozen = os.path.join(store.root, "frozen", selected)
    if not os.path.exists(frozen):
        shutil.copytree(code, frozen)
    digest = contract.frozen_hash(frozen)
    if digest != contract.frozen_hash(code):
        raise RuntimeError("existing freeze directory does not match the selected program")
    record = {"cand": selected, "dir": frozen, "code_sha256": digest}
    store.commit("frozen", updates={"phase": "FROZEN", "frozen": record}, **record)
    return frozen
