from typing import Iterable

from vcap import DetectionNode, BoundingBox
import numpy as np


def process(frame: np.ndarray, results: Iterable[DetectionNode]):
    h, w, _ = frame.shape
    black_frame = np.zeros((h, w, 3), dtype=np.uint8)
    for result in results:
        bbox = BoundingBox(*map(int, result.bbox.rect))
        crop = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2, :]
        ch, cw, _ = crop.shape
        black_frame[bbox.y1:bbox.y1 + ch, bbox.x1:bbox.x1 + cw, :] = crop

    return black_frame
