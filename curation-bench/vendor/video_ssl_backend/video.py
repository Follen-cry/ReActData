"""CPU frame decoding. Frames are zero-based; only endpoints are currently supported."""
from pathlib import Path
from PIL import Image

def frames(path):
    # Animated GIF supports a portable CPU test; production videos use OpenCV.
    if Path(path).suffix.lower()==".gif":
        with Image.open(path) as im:
            for i in range(im.n_frames):
                im.seek(i); yield im.convert("RGB").copy()
    else:
        import cv2
        cap=cv2.VideoCapture(str(path))
        try:
            while True:
                ok,frame=cap.read()
                if not ok: break
                yield Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        finally: cap.release()
def endpoints(path):
    first=last=None; count=0
    for im in frames(path):
        if first is None: first=im
        last=im; count+=1
    if count<2: raise ValueError("video needs at least two readable frames")
    return first,last,count


def source_endpoints(source):
    """Canonical PNGs prepared by the host, rather than lossy MP4 endpoints."""
    result = []
    for key in ('f0', 'final'):
        with Image.open(source[key]) as im:
            im.load()
            if im.format != 'PNG' or im.size != (512, 512):
                raise ValueError('canonical source must be a 512x512 PNG')
            result.append(im.convert('RGB').copy())
    return *result, source['frame_count']


def prepare_png(source, output):
    """Host utility: match the 512x512 INTER_AREA endpoint normalization."""
    import cv2
    im = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if im is None: raise ValueError('unreadable source PNG')
    if im.shape[:2] != (512, 512):
        im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(output), im): raise ValueError('PNG write failed')
