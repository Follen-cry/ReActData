"""Host utility: freeze intermediate frames as indexed 512 PNGs before init."""
import argparse
import json
from pathlib import Path


def prepare(video, output, frame_count):
    import cv2
    import numpy as np
    from video import frames
    if type(frame_count) is not int or frame_count < 2:
        raise ValueError('frame_count must be an integer >= 2')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    bank = {}
    count = 0
    for index, image in enumerate(frames(video)):
        count += 1
        if 0 < index < frame_count - 1:
            array = cv2.resize(np.asarray(image), (512, 512), interpolation=cv2.INTER_AREA)
            path = output / f'{index:06d}.png'
            if not cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR)):
                raise OSError(f'Cannot write frame {index}')
            bank[str(index)] = str(path)
    if count != frame_count:
        raise ValueError(f'frame_count mismatch: decoded {count}, declared {frame_count}; discard incomplete output')
    result = {'frame_count': count, 'frames': bank}
    (output / 'frames.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frame-count', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.video, args.output, args.frame_count), indent=2))
