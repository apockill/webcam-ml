from typing import Iterable

from vcap import DetectionNode, BoundingBox
import numpy as np
import cv2


def process(frame: np.ndarray, results: Iterable[DetectionNode]):
    h, w, _ = frame.shape
    black_frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Create green screen to draw onto
    green_screen = black_frame.copy()
    green_screen[:, :, 1] = 255

    # Convert to OpenCV contours
    contours = [
        np.expand_dims(np.asarray(result.coords, dtype=np.int32), axis=1)
        for result in results
    ]

    if not len(contours):
        return green_screen

    # Only show the largest object on the screen
    largest_contour = max(contours, key=cv2.contourArea)
    mask: np.ndarray = black_frame.copy()
    cv2.drawContours(mask, [largest_contour], -1,
                     color=(255, 255, 255), thickness=-1)
    mask = mask == (255, 255, 255)
    green_screen[mask] = frame[mask]

    return green_screen
