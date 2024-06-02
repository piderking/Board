# Graphing
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from .landmark import ImageLandMarker
from threading import Thread
import mediapipe as mp
from ..measurements import meters_to_inches, frange
mp_hands = mp.solutions.hands

class Keyboard():
    def __init__(self) -> None: #TODO Add multiple kinds of keyboards

        pass

class Graph3D():
    def graph(self,keyboard:bool = False, include_id_in_title: bool = True) -> None:
        self.amt_of_hands = 0
        fig = plt.figure()
        ax = plt.axes(projection ='3d')

        data = [
            [], [], []
        ]
        title: list[str] = []
        scale = 10
        for image in self.image.results:
            print(len(image["graph"]))
            offset = [(-1)**(l+1) for l in range(len(image["graph"]))]
            for p, graph in enumerate(image["graph"]):
                self.amt_of_hands += 1
                title.append(graph["side"])
                ax.text(0, offset[p], 1, str(graph["side"]),)


                for c, _data in enumerate(graph["data"]):
                    _data = [x * scale + (offset[p] if c == 1 else 0) for x in _data]


                    data[c].extend(_data)


                    #ax.text(_x, _y, _z, str(c),)
                             # Create Lines


                    # For coloring

        colors = [c/self.amt_of_hands/21 for _c, _ in enumerate(data[0])]

        for c in range(int(len(data[0])/21)):
            for d in range(21):
                ax.text(data[0][c*21 + d], data[1][c*21 + d], data[2][c*21 + d], str(d),)

            for (d, b) in mp_hands.HAND_CONNECTIONS:
                    ax.plot3D([data[0][c*21 + d], data[0][c*21 + b]], [data[1][c*21 + d], data[1][c*21 + b]], [data[2][c*21 + d], data[2][c*21 + b]], 'gray')

            for c in range(21):
                if c % 4 == 0:
                    pass
                else:
                    colors[c] = 0
            print(np.array(data))

        # Add Scatter Plot Data
        ax.scatter(data[0], data[1], data[2], ) #  c=colors
        ax.set_title(" + ".join(title))

        # Add Temp Keyboard
        xx, yy = np.meshgrid(range(-1, 2), range(-2, 3))
        z =  np.zeros(xx.shape)

        # plot the plane
        ax.plot_surface(xx, yy, z, alpha=0.5)



        plt.xlabel("x axis (scaled by {}x)".format(scale))
        plt.ylabel("y axis (scaled by {}x)".format(scale))
        ax.set(xlim=(-1, 1), ylim=(-1*self.amt_of_hands, 1*self.amt_of_hands), zlim=(-1, 1))
        ax.set(xticks=range(-1, 1,), yticks=range(-1*self.amt_of_hands, 1*self.amt_of_hands,1), zticks=[-1, 0, 1])
        ax.set_aspect("equal", adjustable="box")
        #ax.figure.set_size_inches(meters_to_inches(0.4), meters_to_inches(0.4))
        plt.tight_layout()
        plt.show()

        return data
    def __init__(self, image: ImageLandMarker) -> None:

        self.image = image
        self.data = []
        self.amt_of_hands = 0


