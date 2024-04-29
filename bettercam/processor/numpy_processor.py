import ctypes
import numpy as np
from numpy import rot90, ndarray, newaxis, uint8, zeros
from numpy.ctypeslib import as_array
from .base import Processor


class NumpyProcessor(Processor):
    def __init__(self, color_mode):
        self.cvtcolor = None
        self.color_mode = color_mode
        self.PBYTE = ctypes.POINTER(ctypes.c_ubyte)
        if self.color_mode=='BGRA':
            self.color_mode = None

    def process_cvtcolor(self, image):
        import cv2

        # only one time process
        if self.cvtcolor is None:
            color_mapping = {
                "RGB": cv2.COLOR_BGRA2RGB,
                "RGBA": cv2.COLOR_BGRA2RGBA,
                "BGR": cv2.COLOR_BGRA2BGR,
                "HSV": cv2.COLOR_BGR2HSV,
                "GRAY": cv2.COLOR_BGRA2GRAY
            }
            cv2_code = color_mapping[self.color_mode]
            if cv2_code != cv2.COLOR_BGRA2GRAY:
                self.cvtcolor = lambda image: cv2.cvtColor(image, cv2_code)
            else:
                self.cvtcolor = lambda image: cv2.cvtColor(image, cv2_code)[
                    ..., np.newaxis
                ] 
        return self.cvtcolor(image)

    def shot(self, image_ptr, rect, width, height):
        ctypes.memmove(image_ptr, rect.pBits, height*width*4)

    def process_A0(self, rect, width, height, region):
        pitch = int(rect.Pitch)
        offset = region[1] * pitch
        new_height = region[3] - region[1]
        new_width = region[2] - region[0]
        size = pitch * new_height

        buffer = (ctypes.c_char * size).from_address(ctypes.addressof(rect.pBits.contents) + offset)
        image = np.ndarray((new_height, new_width, 4), dtype=np.uint8, buffer=buffer)

        # Crop if necessary
        image = image[max(0, region[1]):min(new_height, region[3]), max(0, region[0]):min(new_width, region[2])]

        # Convert color if necessary
        if self.color_mode is not None:
            image = self.process_cvtcolor(image)

        return image

    def process_A90(self, rect, width, height, region):
        pitch = int(rect.Pitch)
        offset = region[0] * pitch
        new_height = region[2] - region[0]
        new_width = region[3] - region[1]
        size = pitch * new_width

        buffer = (ctypes.c_char * size).from_address(ctypes.addressof(rect.pBits.contents) + offset)
        image = np.ndarray((new_width, new_height, 4), dtype=np.uint8, buffer=buffer)
        image = np.rot90(image, k=1)  # Rotate counterclockwise

        # Crop if necessary
        image = image[max(0, region[1]):min(new_width, region[3]), max(0, region[0]):min(new_height, region[2])]

        # Convert color if necessary
        if self.color_mode is not None:
            image = self.process_cvtcolor(image)

        return image

    def process_A180(self, rect, width, height, region):
        pitch = int(rect.Pitch)
        offset = (height - region[3]) * pitch
        new_height = region[3] - region[1]
        new_width = region[2] - region[0]
        size = pitch * new_height

        buffer = (ctypes.c_char * size).from_address(ctypes.addressof(rect.pBits.contents) + offset)
        image = np.ndarray((new_height, new_width, 4), dtype=np.uint8, buffer=buffer)
        image = np.rot90(image, k=2)  # Rotate 180 degrees

        # Crop if necessary
        image = image[max(0, region[1]):min(new_height, region[3]), max(0, region[0]):min(new_width, region[2])]

        # Convert color if necessary
        if self.color_mode is not None:
            image = self.process_cvtcolor(image)
        return image

    def process_A270(self, rect, width, height, region):
        pitch = int(rect.Pitch)
        offset = (width - region[3]) * pitch
        new_height = region[2] - region[0]
        new_width = region[3] - region[1]
        size = pitch * new_width

        buffer = (ctypes.c_char * size).from_address(ctypes.addressof(rect.pBits.contents) + offset)
        image = np.ndarray((new_width, new_height, 4), dtype=np.uint8, buffer=buffer)
        image = np.rot90(image, k=-1)  # Rotate clockwise

        # Crop if necessary
        image = image[max(0, region[1]):min(new_width, region[3]), max(0, region[0]):min(new_height, region[2])]

        # Convert color if necessary
        if self.color_mode is not None:
            image = self.process_cvtcolor(image)

        return image







