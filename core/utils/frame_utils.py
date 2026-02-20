# frame_utils.py
import numpy as np
from PIL import Image
from os.path import splitext
import re
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)


def readFlow(fn):
    """Read .flo file in Middlebury format."""
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic.size == 0 or magic[0] != 202021.25:
            raise ValueError(f"Invalid .flo file (bad magic): {fn}")

        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
        return np.resize(data, (int(h), int(w), 2))


def readPFM(file):
    file = open(file, 'rb')

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise ValueError('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if not dim_match:
        raise ValueError('Malformed PFM header.')

    width, height = map(int, dim_match.groups())
    scale = float(file.readline().rstrip())
    endian = '<' if scale < 0 else '>'
    if scale < 0:
        scale = -scale

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid


def read_gen(file_name):
    ext = splitext(file_name)[-1].lower()
    if ext in ('.png', '.jpeg', '.ppm', '.jpg'):
        return Image.open(file_name)
    elif ext in ('.bin', '.raw', '.npy'):
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        return flow if flow.ndim == 2 else flow[:, :, :-1]

    raise ValueError(f"Unknown file extension: {ext} for {file_name}")
