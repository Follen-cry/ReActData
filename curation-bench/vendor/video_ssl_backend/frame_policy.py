"""Frozen frame-selection contract; legacy experiments default to endpoints."""


DEFAULT = {"mode": "endpoints", "max_input_frames": 1, "target": "final"}


def policy(config):
    raw = config.get("frame_policy", DEFAULT)
    if not isinstance(raw, dict) or set(raw) != set(DEFAULT):
        raise ValueError("frame_policy requires mode, max_input_frames, target")
    if raw["mode"] not in ("endpoints", "select") or raw["target"] not in ("final", "select"):
        raise ValueError("invalid frame policy")
    if type(raw["max_input_frames"]) is not int or raw["max_input_frames"] < 1:
        raise ValueError("max_input_frames must be a positive integer")
    if raw["mode"] == "endpoints" and raw != DEFAULT:
        raise ValueError("endpoints requires one input and final target")
    return dict(raw)


def available_frames(source):
    return {0: source["f0"], source["frame_count"] - 1: source["final"],
            **{int(k): v for k, v in source.get("frames", {}).items()}}


def check_frames(source, inputs, target, config):
    p = policy(config)
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= p["max_input_frames"]:
        raise ValueError("input frame count outside frozen policy")
    if any(type(i) is not int for i in inputs) or type(target) is not int:
        raise ValueError("frame indices must be integers")
    if inputs != sorted(set(inputs)):
        raise ValueError("input frames must be unique and chronological")
    allowed = available_frames(source)
    if any(i not in allowed for i in inputs + [target]):
        raise ValueError("frame not in frozen source bank")
    if any(i >= target for i in inputs):
        raise ValueError("input/context frames must precede target; target leakage forbidden")
    final = source["frame_count"] - 1
    if p["mode"] == "endpoints" and (inputs != [0] or target != final):
        raise ValueError("legacy policy supports only F0 -> final")
    if p["target"] == "final" and target != final:
        raise ValueError("target must be final under this policy")


def source_frame(source, index):
    from PIL import Image
    if type(index) is not int or index not in available_frames(source):
        raise ValueError("unknown frame index")
    with Image.open(available_frames(source)[index]) as image:
        image.load()
        if image.format != "PNG" or image.size != (512, 512):
            raise ValueError("canonical frame must be a 512x512 PNG")
        return image.convert("RGB").copy()


def training_text(source, inputs, target):
    """Host-defined text; baseline remains byte-identical to the original protocol."""
    if inputs == [0] and target == source["frame_count"] - 1:
        return source["eval_text"]
    context = "\n".join(f"<image>\nObserved video frame {i} (zero-based)." for i in inputs)
    return (context + f"\nGenerate a single image of video frame {target} (zero-based) "
            f"from this {source['frame_count']}-frame video. Output only the requested frame, "
            "not a video or a sequence.\n\nTask: " + source["description"])
