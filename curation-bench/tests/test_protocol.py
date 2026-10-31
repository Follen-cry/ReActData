from pathlib import Path
from video_ssl.core.protocol import protocol_text
from video_ssl.core.target import build_target_queue


def test_packaged_protocol_matches_authoritative_source():
    repo=Path(__file__).resolve().parents[1]
    assert protocol_text()==(repo/'protocols/method.md').read_text(encoding='utf-8')


def test_target_queue_groups_iterations_by_task():
    targets=build_target_queue(['first','second'],2)
    assert [t.key for t in targets]==['first_iter1','first_iter2','second_iter1','second_iter2']
