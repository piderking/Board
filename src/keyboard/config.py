import os

ABS = os.path.abspath(".")
DATA_PATH = os.path.join(ABS, "data")
IMAGE_PATH = os.path.join(ABS, "data", "images")
VIDEO_PATH = os.path.join(ABS, "data", "videos")
TASK_PATH = os.path.join(ABS, "data", "model")
TASK_FILE = "hand_landmarker"
NUM_HANDS = 2

types = {
    "data": DATA_PATH, "image": IMAGE_PATH, "video":VIDEO_PATH, "task":TASK_PATH
}
def get_file(fileName: str, _type: str = "data"):
    t = types.get(_type)
    if t is None: raise FileNotFoundError(os.path.join( _type ))
    return os.path.join(t, fileName)



