from keyboard.data.image import KeyboardImage
from keyboard.data.landmark import ImageLandMarker
from keyboard.data.graph import Graph3D
from uuid import uuid4

import matplotlib.pyplot as plt

k = KeyboardImage("men.png")
print(k)
l = ImageLandMarker()
l.data = {
    "id":str(uuid4()),
    "image":k

}
l.process()


g = Graph3D(l)

data = g.graph()



