from enum import Enum

class ProcessorBackends(Enum):
    NUMPY = 'numpy'
    CUPY = 'cupy'

class Processor:
    def __init__(self, backend=ProcessorBackends.NUMPY, output_color: str = "RGB", nvidia_gpu: bool = False, rotation_angle: int = 0):
        self.color_mode = output_color
        # Automatically select CuPy if NVIDIA GPU is enabled, else use the specified backend
        selected_backend = ProcessorBackends.CUPY if nvidia_gpu else backend
        self.backend = self._initialize_backend(selected_backend)
        self.set_rotation_function(rotation_angle)  # Select the processing function based on rotation angle

    def set_rotation_function(self, angle):
        # Map the angle to the processing function
        angle_to_function = {
            0: self.backend.processA0,
            90: self.backend.processA90,
            180: self.backend.processA180,
            270: self.backend.processA270
        }
        self.process = angle_to_function.get(angle)
        if not self.process:
            raise ValueError(f"Unsupported rotation angle: {angle}")

    def process(self, rect, width, height, region):
        # This method will be dynamically set to the appropriate processing function
        pass

    def _initialize_backend(self, backend):
        # Initialize the appropriate backend based on the ProcessorBackends enum
        if backend == ProcessorBackends.NUMPY:
            from bettercam.processor.numpy_processor import NumpyProcessor
            return NumpyProcessor(self.color_mode)
        elif backend == ProcessorBackends.CUPY:
            from bettercam.processor.cupy_processor import CupyProcessor
            return CupyProcessor(self.color_mode)
        else:
            raise ValueError(f"Unknown backend: {backend}")

#####USAGE
#processor = Processor(output_color="RGB", nvidia_gpu=True, rotation_angle=90)
