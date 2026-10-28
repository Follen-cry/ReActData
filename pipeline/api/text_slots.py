"""Training text contract (fixed): static templates + Runner-filled whitelist slots.

A candidate ships ``text_templates.yaml``: a mapping ``template_id -> template string``. A databuilder can only
choose a template_id per row; it never emits free text. The Runner renders the human turn as

    "<image>\\n" + template.format(**slots)

where the leading <image> is F0. Every slot value is computed by the Runner from allowed sources only:

  {description}          the clip's original task description (prompt.txt), verbatim (it names the colours and the
                         size change of the analogy, exactly as the dev/test descriptions do)
  {wrapped_description}  the evaluation text: VBVR target_pred wrapper around {description}
                         (the fixed baseline and dev/test use exactly "{wrapped_description}")
  {context_list}         one line per context frame: "Frame at t = X.X s: <image>" (the only way to place
                         context images; required whenever the row has context frames)
  {n_context}            number of context frames
  {t_target}             time of the row's target frame, e.g. "3.7 s"
  {target_spec}          (v4) what to generate: "the final frame (t = 3.7 s) of the 3.8 s clip" or "the frame at
                         t = 1.8 s of the 3.8 s clip, partway through the process" for an earlier target
  {duration}             clip duration, e.g. "4.7 s"

Static template text (outside the braces) must not contain digits, "<", ">" or braces other than the slots
above, and is at most MAX_TEMPLATE_CHARS long. Static text is the same for every row, so sample-specific
information can enter a row's text only through slots and through *which* template the databuilder picks;
the latter is audited after each run (runner/audit.py), which can reveal but never prove the absence of leakage.

v4 rule: a row whose target is not the final frame must use a template that contains {target_spec} and does NOT contain
{wrapped_description} -- the evaluation wrapper says "output one image representing the completed final state only",
which would contradict an intermediate-frame label.
"""
from __future__ import annotations

import re
import string
from typing import Dict, List

SLOTS = ("description", "wrapped_description", "context_list", "n_context", "t_target", "duration", "target_spec")
MAX_TEMPLATE_CHARS = 2000
MAX_TEMPLATES = 16
_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]{0,40}$")


class TemplateError(ValueError):
    pass


def placeholders(template: str) -> List[str]:
    out = []
    for _, field, spec, conv in string.Formatter().parse(template):
        if field is None:
            continue
        if spec or conv:
            raise TemplateError(f"format spec / conversion not allowed: {{{field}{'!' + conv if conv else ''}{':' + spec if spec else ''}}}")
        out.append(field)
    return out


def check_templates(templates) -> Dict[str, str]:
    if not isinstance(templates, dict) or not templates:
        raise TemplateError("text_templates.yaml must be a non-empty mapping template_id -> string")
    if len(templates) > MAX_TEMPLATES:
        raise TemplateError(f"at most {MAX_TEMPLATES} templates")
    out = {}
    for tid, t in templates.items():
        if not isinstance(tid, str) or not _ID.match(tid):
            raise TemplateError(f"bad template id {tid!r}")
        if not isinstance(t, str) or not t.strip():
            raise TemplateError(f"template {tid}: must be a non-empty string")
        if len(t) > MAX_TEMPLATE_CHARS:
            raise TemplateError(f"template {tid}: longer than {MAX_TEMPLATE_CHARS} characters")
        fields = placeholders(t)
        bad = [f for f in fields if f not in SLOTS]
        if bad:
            raise TemplateError(f"template {tid}: unknown slots {bad}; allowed {list(SLOTS)}")
        if fields.count("context_list") > 1:
            raise TemplateError(f"template {tid}: {{context_list}} at most once")
        static = re.sub(r"\{[a-z_]+\}", "", t)
        if re.search(r"[0-9<>{}]", static):
            raise TemplateError(f"template {tid}: static text must not contain digits, '<', '>' or stray braces")
        out[tid] = t
    return out


def _t(frame: int, fps: int) -> str:
    return f"{frame / fps:.1f} s"


def render(template: str, *, description: str, wrapped_description: str, context_frames: List[int], n_frames: int,
           fps: int, target_frame: int = None) -> str:
    fields = placeholders(template)
    final = n_frames - 1
    target_frame = final if target_frame is None else target_frame
    if context_frames and "context_list" not in fields:
        raise TemplateError("row has context frames but the template has no {context_list}")
    if target_frame != final and ("target_spec" not in fields or "wrapped_description" in fields):
        raise TemplateError("a row with a non-final target needs a template with {target_spec} and without {wrapped_description}")
    dur = f"{n_frames / fps:.1f} s"
    spec = (f"the final frame (t = {_t(final, fps)}) of the {dur} clip" if target_frame == final else
            f"the frame at t = {_t(target_frame, fps)} of the {dur} clip, partway through the process")
    slots = {
        "description": description,
        "wrapped_description": wrapped_description,
        "context_list": "\n".join(f"Frame at t = {_t(f, fps)}: <image>" for f in context_frames),
        "n_context": str(len(context_frames)),
        "t_target": _t(target_frame, fps),
        "target_spec": spec,
        "duration": f"{n_frames / fps:.1f} s",
    }
    text = "<image>\n" + template.format(**slots)
    if text.count("<image>") != 1 + len(context_frames):
        raise TemplateError("rendered text has the wrong number of <image> tokens")
    return text
