from enum import Enum

class ProcessorBackends(Enum):
    NUMPY = 'numpy'
    CUPY = 'cupy'

class Processor:
    def __init__(self, backend=ProcessorBackends.CUPY, output_color: str = "RGB", nvidia_gpu: bool = False, rotation_angle: int = 0):
        self.color_mode = output_color
        # Automatically select CuPy if NVIDIA GPU is enabled, else use the specified backend
        selected_backend = ProcessorBackends.CUPY if nvidia_gpu else backend
        self.backend = self._initialize_backend(selected_backend)
        if selected_backend == ProcessorBackends.CUPY:
            self.set_rotation_CPfunction(rotation_angle)
        else:
            self.set_rotation_function(rotation_angle)

    def set_rotation_function(self, angle):
        # Map the angle to the processing function in self.backend
        angle_to_function = {
            0: self.backend.processA0,
            90: self.backend.processA90,
            180: self.backend.processA180,
            270: self.backend.processA270
        }
        self.process = angle_to_function.get(angle)
        if not self.process:
            raise ValueError(f"Unsupported rotation angle: {angle}")

    def set_rotation_CPfunction(self, angle):
        # Map the angle to the processing function in self.backend (assuming this method is for a GPU context)
        angle_to_function = {
            0: self.backend.processCPA0,
            90: self.backend.processCPA90,
            180: self.backend.processCPA180,
            270: self.backend.processCPA270
        }
        self.process = angle_to_function.get(angle)
        if not self.process:
            raise ValueError(f"Unsupported rotation angle: {angle}")

    def process(self, rect, width, height, region):
        # This method will be dynamically set to the appropriate processing function
        pass

    def _initialize_backend(self, backend):
        print(f"Initializing backend: {backend}")
        if backend == ProcessorBackends.NUMPY:
            from bettercam.processor.numpy_processor import NumpyProcessor
            return NumpyProcessor(self.color_mode)
        elif backend == ProcessorBackends.CUPY:
            from bettercam.processor.cupy_processor import CupyProcessor
            return CupyProcessor(self.color_mode)
        else:
            raise ValueError(f"Unknown backend: {backend}")



