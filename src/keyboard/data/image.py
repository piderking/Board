import cv2
import numpy as np
import mediapipe as mp
from ..config import get_file
class KeyboardImage(object):
    def __init__(self, file_name:str, resolution: dict or None = None,) -> None:
        self.file_name = file_name

        self.file = mp.Image.create_from_file(get_file("{}".format(file_name), _type="image"))

        # Use provided resolution or else just use the resolution fo the image
        self.resolution = resolution if resolution is type(dict) else {"width":self.file.numpy_view().shape[1], "height":self.file.numpy_view().shape[0]}

    def __call__(self) -> mp.Image:
        return self.file
    @property
    def size(self) -> dict:
        return self.resolution
    def __repr__(self) -> str:
        return "<Image at ({}), Resolution:{}> ".format(id(self), "{}x{}".format(self.size["width"], self.size["height"]))

