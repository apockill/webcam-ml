from argparse import ArgumentParser
from queue import Queue
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pyfakewebcam

from webcam_ml import Capture, CapsuleRuntime
from extensions import extension as frame_callable


class Environment:
    """A context manager for all of the services used by the script,
    to ensure everything is closed if an error occurs"""

    def __init__(self,
                 input_device,
                 output_device,
                 capsules_dir,
                 frame_queue,
                 desired_resolution):
        self.runtime: Optional[CapsuleRuntime] = None
        self.output_fakecam: Optional[pyfakewebcam.FakeWebcam] = None
        self.input_capture: Optional[Capture] = None

        # Inputs
        self._input_device = input_device
        self._output_device = output_device
        self._capsules_dir = capsules_dir
        self._frame_queue = frame_queue
        self._desired_resolution = desired_resolution

    def __enter__(self, *args, **kwargs):
        # Create the input camera
        print("Starting camera")
        self.input_camera = Capture(
            device=self._input_device,
            frame_callbacks=[self._frame_queue.put],
            try_resolution=self._desired_resolution)

        # Create the capsule runtime
        print("Starting runtime")
        self.runtime = CapsuleRuntime(directory=self._capsules_dir)

        # Create the fake webcam
        first_frame, _ = self._frame_queue.get()
        h, w, _ = first_frame.shape
        print("Starting camera")
        self.camera = pyfakewebcam.FakeWebcam(self._output_device, w, h)

    def __exit__(self, *args, **kwargs):
        print("Closing", args, kwargs)
        if self.input_camera:
            print("Closing camera")
            self.input_camera.close()
        if self.runtime:
            print("Closing runtime")
            self.runtime.close()
        self.input_camera = None
        self.runtime = None
        self.camera = None


def main(input_device: str,
         output_device: str,
         capsules_dir: Path,
         desired_resolution: Tuple[int, int]):
    frame_queue = Queue(maxsize=1)
    frame: np.ndarray

    env = Environment(input_device, output_device, capsules_dir, frame_queue,
                      desired_resolution)

    with env:
        print("Starting")
        while True:
            frame, timestamp = frame_queue.get()

            display_frame = frame_callable(
                frame=frame,
                results=env.runtime.process_frame(frame))

            rgb_display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            env.camera.schedule_frame(rgb_display_frame)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Do background removal and other CV algorithms and output "
                    "as a webcam.")
    parser.add_argument("-i", "--input-device", type=str, required=True,
                        help="For example, '/dev/video1/'")
    parser.add_argument("-o", "--output-device", type=str, required=True,
                        help="For example, '/dev/video2/'")
    parser.add_argument("-c", "--capsules-dir", type=Path, required=True)
    parser.add_argument("-r", "--try-resolution", type=int, nargs=2,
                        default=(1920, 1080), help="e.g. (width, height)")
    args = parser.parse_args()

    main(
        input_device=args.input_device,
        output_device=args.output_device,
        capsules_dir=args.capsules_dir,
        desired_resolution=args.try_resolution
    )
