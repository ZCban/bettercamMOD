import enum


class ProcessorBackends(enum.Enum):
    PIL = 0
    NUMPY = 1
    CUPY = 2


class Processor:
    def __init__(self, backend=ProcessorBackends.NUMPY, output_color: str = "RGB", nvidia_gpu: bool = False):
        self.color_mode = output_color
        if nvidia_gpu:
            backend = ProcessorBackends.CUPY
        self.backend = self._initialize_backend(backend)

    def process(self, rect, width, height, region, rotation_angle):
        return self.backend.process(rect, width, height, region, rotation_angle)

    def _initialize_backend(self, backend):
        if backend == ProcessorBackends.NUMPY:
            from bettercam.processor.numpy_processor import NumpyProcessor

            return NumpyProcessor(self.color_mode)
        
        elif backend == ProcessorBackends.CUPY:
            from bettercam.processor.cupy_processor import CupyProcessor

            return CupyProcessor(self.color_mode)
        
        else:
            print(f"Unknown backend: {backend}")
