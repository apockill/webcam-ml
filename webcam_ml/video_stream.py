from threading import Thread
from typing import Callable, List, Tuple, Any
from time import time

import cv2
import numpy as np

CALLBACK_TYPE = Callable[[Tuple[np.ndarray, float]], Any]


class Capture:
    """
    Calls a list of callbacks with a tuple of (frame, timestamp), with the
    frame in BGR format.
    """

    def __init__(self, device: str,
                 frame_callbacks: List[CALLBACK_TYPE]):
        self.cap = cv2.VideoCapture(device)
        self.callbacks = list(frame_callbacks)
        self._running = True
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        frame: np.ndarray

        while self._running:
            ret, frame = self.cap.read()
            timestamp: float = time()

            for callback in list(self.callbacks):
                callback((frame.copy(), timestamp))

    def close(self):
        self._running = False
        self.cap.release()
