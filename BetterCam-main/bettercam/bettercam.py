import time
import ctypes
from typing import Tuple
from threading import Thread, Event, Lock
import comtypes
import numpy as np
from bettercam.core import Device, Output, StageSurface, Duplicator
from bettercam.processor import Processor
from bettercam.util.timer import (
    create_high_resolution_timer,
    set_periodic_timer,
    wait_for_timer,
    cancel_timer,
    INFINITE,
    WAIT_FAILED,
)


class BetterCam:
    def __init__(
        self,
        output: Output,
        device: Device,
        region: Tuple[int, int, int, int],
        output_color: str = "RGB",
        nvidia_gpu: bool = False,
        max_buffer_len=64,
    ) -> None:
        self._output: Output = output
        self._device: Device = device
        self._stagesurf: StageSurface = StageSurface(
            output=self._output, device=self._device
        )
        self._duplicator: Duplicator = Duplicator(
            output=self._output, device=self._device
        )
        self.nvidia_gpu = nvidia_gpu
        # Set the rotation angle from the output device
        self.rotation_angle: int = self._output.rotation_angle
        # Initialize Processor with the rotation angle
        self._processor: Processor = Processor(
            output_color=output_color, 
            nvidia_gpu=nvidia_gpu, 
            rotation_angle=self.rotation_angle
        )

        self.width, self.height = self._output.resolution
        self.channel_size = len(output_color) if output_color != "GRAY" else 1

        self._region_set_by_user = region is not None
        self.region: Tuple[int, int, int, int] = region
        if self.region is None:
            self.region = (0, 0, self.width, self.height)
        self._validate_region(self.region)

        self.max_buffer_len = max_buffer_len
        self.is_capturing = False

        self.__thread = None
        self.__lock = Lock()
        self.__stop_capture = Event()

        self.__frame_available = Event()
        self.__frame_buffer: np.ndarray = None
        self.__head = 0
        self.__tail = 0
        self.__full = False

        self.__timer_handle = None

        self.__frame_count = 0
        self.__capture_start_time = 0
        
    def grab(self, region: Tuple[int, int, int, int] = None):
        if region is None:
            region = self.region
        else:
            self._validate_region(region)
        frame = self._grab(region)
        return frame

    def _grab(self, region: Tuple[int, int, int, int]):
        if self._duplicator.update_frame():
            if not self._duplicator.updated:
                return None
            self._device.im_context.CopyResource(
                self._stagesurf.texture, self._duplicator.texture
            )
            self._duplicator.release_frame()
            rect = self._stagesurf.map()
            frame = self._processor.process(rect, self.width, self.height, region)
            self._stagesurf.unmap()
            return frame
        else:
            self._on_output_change()
            return None

    def _on_output_change(self):
        time.sleep(0.1)  # Wait for Display mode change (Access Lost)
        self._duplicator.release()
        self._stagesurf.release()
        self._output.update_desc()
        self.width, self.height = self._output.resolution
        if self.region is None or not self._region_set_by_user:
            self.region = (0, 0, self.width, self.height)
        self._validate_region(self.region)
        if self.is_capturing:
            self._rebuild_frame_buffer(self.region)
        self.rotation_angle = self._output.rotation_angle
        while True:
            try:
                self._stagesurf.rebuild(output=self._output, device=self._device)
                self._duplicator = Duplicator(output=self._output, device=self._device)
            except comtypes.COMError as ce:
                continue
            break

    def start(
        self,
        region: Tuple[int, int, int, int] = None,
        target_fps: int = 60,
        video_mode=False,
        delay: int = 0,
    ):
        #if delay != 0:
            #time.sleep(delay)
            #self._on_output_change()
        if region is None:
            region = self.region
        #self._validate_region(region)
        self.is_capturing = True
        frame_shape = (region[3] - region[1], region[2] - region[0], self.channel_size)
        self.__frame_buffer = np.ndarray(
            (self.max_buffer_len, *frame_shape), dtype=np.uint8
        )
        self.__thread = Thread(
            target=self.__capture,
            name="BetterCam",
            args=(region, target_fps, video_mode),
        )
        self.__thread.daemon = True
        self.__thread.start()

    def stop(self):
        if self.is_capturing:
            self.__frame_available.set()
            self.__stop_capture.set()
            if self.__thread is not None:
                self.__thread.join(timeout=10)
        self.is_capturing = False
        self.__frame_buffer = None
        self.__frame_count = 0
        self.__frame_available.clear()
        self.__stop_capture.clear()

    def get_latest_frame(self):
        self.__frame_available.wait()
        with self.__lock:
            ret = self.__frame_buffer[(self.__head - 1) % self.max_buffer_len]
            self.__frame_available.clear()
        return np.array(ret)

    def __capture(self, region: Tuple[int, int, int, int], target_fps: int = 60, video_mode=False):
        if target_fps > 0:
            period_ms = 1000 // target_fps  # milliseconds for periodic timer
            self.__timer_handle = create_high_resolution_timer()
            set_periodic_timer(self.__timer_handle, period_ms)

        self.__capture_start_time = time.perf_counter()
        try:
            while not self.__stop_capture.is_set():
                if target_fps > 0:
                    wait_for_timer(self.__timer_handle)

                frame = self._grab(region)
                if frame is None and video_mode:
                    frame = np.array(self.__frame_buffer[(self.__head - 1) % self.max_buffer_len], copy=False)

                if frame is not None:
                    with self.__lock:
                        self.__frame_buffer[self.__head] = frame
                        self.__head = (self.__head + 1) % self.max_buffer_len
                        if self.__head == self.__tail:
                            self.__tail = (self.__tail + 1) % self.max_buffer_len
                        self.__frame_available.set()
                        self.__frame_count += 1
        finally:
            if target_fps > 0:
                cancel_timer(self.__timer_handle)
            #print(f"Screen Capture FPS: {int(self.__frame_count / (time.perf_counter() - self.__capture_start_time))}")



    def _rebuild_frame_buffer(self, region: Tuple[int, int, int, int]):
        if region is None:
            region = self.region
        frame_shape = (
            region[3] - region[1],
            region[2] - region[0],
            self.channel_size,
        )
        with self.__lock:
            self.__frame_buffer = np.ndarray(
                (self.max_buffer_len, *frame_shape), dtype=np.uint8
            )
            self.__head = 0
            self.__tail = 0
            self.__full = False

    def _validate_region(self, region: Tuple[int, int, int, int]):
        l, t, r, b = region
        if not (self.width >= r > l >= 0 and self.height >= b > t >= 0):
            raise ValueError(
                f"Invalid Region: Region should be in {self.width}x{self.height}"
            )

    def release(self):
        self.stop()
        self._duplicator.release()
        self._stagesurf.release()

    def __del__(self):
        self.release()

    def __repr__(self) -> str:
        return "<{}:\n\t{},\n\t{},\n\t{},\n\t{}\n>".format(
            self.__class__.__name__,
            self._device,
            self._output,
            self._stagesurf,
            self._duplicator,
        )
