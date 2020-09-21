from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from pathlib import Path
from vcap import (
    load_capsule,
    package_capsule,
    CAPSULE_EXTENSION,
    NodeDescription,
    DetectionNode
)


class CapsuleRuntime:
    """Load all capsules in a directory and make them available,
    with an easy way to close them all"""

    def __init__(self, directory: Path):
        self.capsules = []
        self.executor = ThreadPoolExecutor()
        self.directory = directory

        # Package and load capsules
        self.package_capsules(directory)
        for capsule_file in directory.glob(f"*{CAPSULE_EXTENSION}"):
            capsule = load_capsule(capsule_file)
            if capsule.input_type.size is not NodeDescription.Size.NONE:
                raise NotImplemented("Capsules that require input are not "
                                     "yet supported!")
            self.capsules.append(capsule)

    @staticmethod
    def package_capsules(directory):
        # Package any capsule directories in a directory
        for capsule_path in directory.iterdir():
            if not capsule_path.is_dir():
                continue
            output_file_name = f"{capsule_path.name}{CAPSULE_EXTENSION}"
            output_file = capsule_path.with_name(output_file_name)
            package_capsule(
                unpackaged_dir=capsule_path,
                output_file=output_file
            )

    def process_frame(self, frame) -> Iterable[DetectionNode]:
        futures = []
        for capsule in self.capsules:
            future = self.executor.submit(
                lambda frame=frame, capsule=capsule: capsule.process_frame(
                    frame=frame,
                    detection_node=None,
                    options=capsule.default_options,
                    state=capsule.get_state(0)
                )
            )
            futures.append(future)
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            elif isinstance(result, DetectionNode):
                yield result
            else:
                for node in result:
                    yield node

    def close(self):
        for capsule in self.capsules:
            capsule.close()
        self.executor.shutdown()
