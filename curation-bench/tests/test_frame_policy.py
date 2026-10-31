"""CPU tests for frame-aware contracts; no Linux/GPU imports required."""
import json
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor' / 'video_ssl_backend'))
from frame_policy import check_frames, source_frame, training_text
from task import resolve
from dataset import validate
from training import row_specs


@pytest.fixture
def setup(tmp_path):
    bank = {}
    for i, color in enumerate(['red', 'green', 'blue', 'yellow']):
        path = tmp_path / f'{i}.png'
        Image.new('RGB', (512, 512), color).save(path)
        bank[i] = str(path)
    video = tmp_path / 'source.gif'
    Image.new('RGB', (4, 4), 'red').save(video)
    source = dict(video=str(video), f0=bank[0], final=bank[3], frame_count=4,
                  description='Change color then scale', eval_text='<image>\nGenerate the final result.',
                  frames={'1': bank[1], '2': bank[2]})
    raw = dict(mode='simulation', task='shape_color_then_scale', simulation_samples=1,
               sources={'v': source}, frame_policy=dict(mode='select', max_input_frames=3, target='select'))
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(raw))
    config = resolve(config_path)
    directory = tmp_path / 'dataset'
    directory.mkdir()
    return directory, config, config_path


def build(directory, config, inputs, target):
    source = config['sources']['v']
    row = dict(sample_id='0', source_video_id='v', input_frames=inputs, target_frame=target,
               input_images=[f'input_{i}.png' for i in inputs], target_image='target.png',
               text=training_text(source, inputs, target))
    for name, index in zip(row['input_images'] + [row['target_image']], inputs + [target]):
        source_frame(source, index).save(directory / name)
    (directory / 'manifest.jsonl').write_text(json.dumps(row))
    return row


def test_intermediate_target_and_multiple_context_frames(setup):
    directory, config, _ = setup
    row = build(directory, config, [0, 1], 2)
    assert validate(directory, config) == [row]
    spec = row_specs(directory)[0]
    assert len(spec['images']) == 2
    assert spec['cond_image'] == spec['images'][0]
    assert spec['target_image'].endswith('target.png')
    assert row['text'].count('<image>') == 2
    assert 'frame 2' in row['text']


def test_baseline_unchanged_and_nonzero_input_allowed(setup):
    directory, config, _ = setup
    baseline = build(directory, config, [0], 3)
    assert baseline['text'] == config['sources']['v']['eval_text']
    validate(directory, config)
    build(directory, config, [1], 2)
    validate(directory, config)


@pytest.mark.parametrize('inputs,target', [([0, 2], 2), ([2], 1), ([1, 0], 3), ([0, 0], 2),
                                          ([], 2), ([True], 2), ([0], True), ([0], 9)])
def test_rejects_leakage_order_duplicates_and_unknown_frames(setup, inputs, target):
    _, config, _ = setup
    with pytest.raises(ValueError):
        check_frames(config['sources']['v'], inputs, target, config)


@pytest.mark.parametrize('kind', ['text', 'pixels', 'source', 'video', 'input_count'])
def test_validation_detects_tampering(setup, kind):
    directory, config, _ = setup
    row = build(directory, config, [0, 1], 2)
    if kind == 'text':
        row['text'] = config['sources']['v']['eval_text']
    elif kind == 'pixels':
        Image.new('RGB', (512, 512), 'black').save(directory / 'target.png')
    elif kind == 'source':
        Path(config['sources']['v']['frames']['2']).write_bytes(b'changed')
    elif kind == 'video':
        Path(config['sources']['v']['video']).write_bytes(b'changed')
    else:
        row['input_images'].pop()
    (directory / 'manifest.jsonl').write_text(json.dumps(row))
    with pytest.raises(ValueError):
        validate(directory, config)


def test_legacy_configs_and_final_only_policy(setup):
    directory, config, path = setup
    raw = json.loads(path.read_text())
    del raw['frame_policy']
    path.write_text(json.dumps(raw))
    legacy = resolve(path)
    assert legacy['frame_policy']['mode'] == 'endpoints'
    build(directory, legacy, [0], 3)
    validate(directory, legacy)
    with pytest.raises(ValueError):
        check_frames(legacy['sources']['v'], [0], 2, legacy)
    config['frame_policy']['target'] = 'final'
    with pytest.raises(ValueError):
        check_frames(config['sources']['v'], [0], 2, config)
    check_frames(config['sources']['v'], [0, 1], 3, config)


def test_context_limit_is_enforced(setup):
    _, config, _ = setup
    config['frame_policy']['max_input_frames'] = 1
    with pytest.raises(ValueError, match='count'):
        check_frames(config['sources']['v'], [0, 1], 2, config)


@pytest.mark.parametrize('index', ['0', '3', '-1', '01', '99'])
def test_init_rejects_bad_bank_indices(setup, index):
    _, _, path = setup
    raw = json.loads(path.read_text())
    raw['sources']['v']['frames'][index] = raw['sources']['v']['f0']
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        resolve(path)
