import mediapipe as mp
from ..config import get_file, TASK_FILE, NUM_HANDS
from threading import Thread
import numpy as np
import os
from .image import KeyboardImage
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode



class ImageLandMarker():
    def __init__(self) -> None:
        self._data, self.results = [], []
        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=get_file(TASK_FILE + ".task", _type="task")),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=NUM_HANDS,
        )

    @property
    def data(self):
        return self._data

    @property
    def tdata(self) -> dict:
        """Get data thats going to be processed

        Returns:
            dict: First Frame
        """
        f = self._data[0]
        self._data.pop()
        return f

    @data.setter
    def data(self, info: dict):
        """_summary_

        Args:
            info (dict): {id:str, data: str or np.ndarray or list or mp.Image}

        Raises:
            FileNotFoundError: If new image is a file path and not specfied
        """
        new_image = info["image"]
        if type(new_image) == str:
            # File
            if os.path.exists(get_file(new_image, _type="image")):
                self._data.append({"id":info["id"], "image":mp.Image.create_from_file(get_file(new_image, _type="image"))})
            else:
                raise FileNotFoundError("No Image Specified at {}".format(get_file(new_image, _type="image")))
        elif type(new_image) == np.ndarray or type(new_image) == list:
            if type(new_image) == list: # Transform into an NP Array
                new_image = np.array(new_image)

            self._data.append({"id":info["id"], "image":mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(new_image))})

        elif type(new_image) == KeyboardImage:
            self._data.append({"id":info["id"], "image":new_image()})

        else:
            raise TypeError("Image is not correct type")

    def process(self) -> None:
        """
        self.data = [ numpy ]
        """
        with HandLandmarker.create_from_options(self.options) as landmarker:

            while len(self.data) > 0:
                    f: dict[np.ndarray, str] = self.tdata # Save Frame of Data
                    hand_landmarker_result: mp.tasks.vision.HandLandmarkerResult = landmarker.detect(
                        f["image"],
                    )

                    offsets = [(-1)**n for n in range(len(hand_landmarker_result.hand_world_landmarks))]
                    # TODO Add Transformations Here
                    array = np.array([[[x.x, x.y, x.z] for x in z] for c, z in enumerate(hand_landmarker_result.hand_world_landmarks)])
                    handies = [{"display_name":x[0].display_name }for x in hand_landmarker_result.handedness]
                    print(array.shape)
                    print(hand_landmarker_result.handedness[0][0].display_name)
                    self.results.append({
                            "id":f["id"],
                            "graph":[
                            {
                                "side": handy["display_name"],
                                "data": [
                                    x for x in bT.T # (x, y, z)
                                ]
                            } for handy, bT in zip(handies, array)
                            ],
                            "world": [
                                {
                                "side": handy.display_name,
                                "data": bT # Just the actualy Land_marks
                                } for handy, bT in zip(hand_landmarker_result.handedness[0], array)
                            ]
                        })

