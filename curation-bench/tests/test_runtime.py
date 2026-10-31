"""Runtime integrity and dataset decision differences."""
import pytest
from video_ssl.core import runtime as r


def test_selection_tracks_membership_and_order():
    row = lambda x: {"source_video_id": x}
    delta = r.selection_delta(list(map(row, ["a", "b", "c"])), list(map(row, ["c", "b", "d"])))
    assert delta["added"] == ["d"]
    assert delta["removed"] == ["a"]
    assert delta["moved"] == [{"source": "c", "from": 2, "to": 0}]



def test_frame_delta_detects_changed_context_and_target_without_changing_video():
    before = [{'source_video_id':'a', 'input_frames':[0], 'target_frame':59}]
    after = [{'source_video_id':'a', 'input_frames':[0,15], 'target_frame':30}]
    delta = r.selection_delta(before, after)
    assert not delta['added'] and not delta['removed'] and not delta['moved']
    assert delta['frame_changes'] == [{'source':'a', 'before':{'input_frames':[0], 'target_frame':59},
                                        'after':{'input_frames':[0,15], 'target_frame':30}}]



def test_frozen_inputs_detect_changes(tmp_path):
    file = tmp_path / "config.json"
    file.write_text("original")
    plan = {"hashes": {str(file): r.digest(file)}}
    r.verify(plan)
    file.write_text("changed")
    with pytest.raises(ValueError, match="Frozen input changed"):
        r.verify(plan)
