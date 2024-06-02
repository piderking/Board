__version__ = "0.0.1"
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from keyboard.config import get_file
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from mediapipe import solutions
import cv2
from scipy.spatial.transform import Rotation as R
from graph import Graph
from graph.shapes import Point_Cloud

# Graphing
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode



# Create a hand landmarker instance with the image mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=get_file("hand_landmarker.task", _type="task")),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    )
r1 = R.from_euler("XYZ", [0, 180, 0], degrees=True)  # intrinsic

with HandLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
    print(landmarker)

    # Load the input image from an image file.
    mp_image = mp.Image.create_from_file(get_file("men.png", _type="image"))

    # Load the input image from a numpy array.
    #mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    hand_landmarker_result: mp.tasks.vision.HandLandmarkerResult = landmarker.detect(mp_image)
    #print(dir(hand_landmarker_result))
   #  np.array(hand_landmarker_result.hand_world_landmarks)

    two_d_array = [[[x.x, x.y] for x in z] for z in hand_landmarker_result.hand_landmarks]

    two_d_points = []
    array = np.array([[r1.apply([x.x*10, x.y*10, x.z*10]) for x in z] for z in hand_landmarker_result.hand_world_landmarks])

    print(array[0][0])
    points = np.asarray([array[0][0], array[0][5], array[0][17]])
    normal_vector = np.cross(points[2] - points[0], points[1] - points[2])
    normal_vector /= np.linalg.norm(normal_vector)
    normal_vector *= 180

    # Modify for 3D
    #r2 = R.from_euler("XYZ", normal_vector, degrees=True)  # intrinsic
    #array = np.array([[r2.apply(x) for x in z] for z in array])
    #print(array[0][0])

    frame_height, frame_width, channels  = array.shape

    # Fiddle with this number to get the camera image
    # hands to align with the mediapipe points. Unless
    # you know your camera's focal length, then put in
    # here.
    focal_length = frame_width * 0.75
    center = (frame_width/2, frame_height/2)
    camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype = "double"
                                 )
    distortion = np.zeros((4, 1))
    fov_x = np.rad2deg(2 * np.arctan2(focal_length, 2 * focal_length))

    results = hand_landmarker_result
    world_points_total = []
    if results.hand_landmarks:
        for [i, hand_landmarks] in enumerate(results.hand_landmarks):
            world_landmarks = results.hand_landmarks[i]

            model_points = np.float32([[-l.x, -l.y, -l.z] for l in world_landmarks])
            image_points = np.float32([[l.x * frame_width, l.y * frame_height] for l in hand_landmarks])
            success, rvecs, tvecs, = cv2.solvePnP(
                    model_points,
                    image_points,
                    camera_matrix,
                    distortion,
                    flags=cv2.SOLVEPNP_SQPNP
            )

            transformation = np.eye(4)  # needs to 4x4 because you have to use homogeneous coordinates
            transformation[0:3, 3] = tvecs.squeeze()
                # the transformation consists only of the translation, because the rotation is accounted for in the model coordinates. Take a look at this (https://codepen.io/mediapipe/pen/RwGWYJw to see how the model coordinates behave - the hand rotates, but doesn't translate

                # transform model coordinates into homogeneous coordinates
            model_points_hom = np.concatenate((model_points, np.ones((21, 1))), axis=1)

                # apply the transformation
            world_points = model_points_hom.dot(np.linalg.inv(transformation).T)
            world_points_total.append(world_points)

    # Graph

    # Set to new array
    array = [r1.apply([x.x, x.y, x.z]) for x  in world_landmarks]
    array=np.array(array)
    print(array.shape)

    hand = ""
    index = 0
    for handy in hand_landmarker_result.handedness:
        if index == handy[0].index:
            hand +=handy[0].display_name

    # print(np.shape(array[0]))
    x = np.array([x[0] for x in array])
    y = np.array([x[1] for x in array])
    z = [x[2] for x in array]
    colors = [2*(_c/4) for _c, _ in enumerate(array)]
    for c, _ in enumerate(z):
        if c % 4 == 0:
            #print(z[0])
            colors[c] = 0
        else:
            # z[c] = 10
            colors[c] = 0

    # print(np.array(array).shape)


    fig = plt.figure()

    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')


    for c, (_x, _y, _z) in enumerate(zip(x, y, z)):
        label = str(c)
        ax.text(_x, _y, _z, label,)

        if (c-1) % 4 == 0 and c != 0:
            ax.plot3D([x[0], _x], [y[0], _y], [z[0], _z], 'red')
        ax.plot3D([x[4], x[20]], [y[4], y[20]], [z[4], z[20]], 'red')
        ax.plot3D([x[4], x[20]], [y[4], y[20]], [z[20], z[20]] if z[20] < z[4] else [z[4], z[4]] , 'red')




    # Create Lines
    for (d, b) in mp_hands.HAND_CONNECTIONS:
        ax.plot3D([x[d], x[b]], [y[d], y[b]], [z[d], z[b]], 'red' if ((d-1) % 4 == 0 and (b-1) % 4 == 0) or (d == 0 or b == 0) else'gray')
    print([list(px) for px in list(mp_hands.HAND_CONNECTIONS)])


    #
    ax.scatter(x, y, z, c=colors)

    # syntax for plotting
    ax.set_title(hand)
    plt.xlabel("x-label")
    plt.ylabel("y-label")
    #plt.show()
    d = Graph("http://127.0.0.1:5000/")

    special = []
    for td in array:
        special.extend(td)

    #uuid2 = d.graph(np.array(sp
    # ecial), {"x":2,"y":2,"z":2}, .1, {"x":1,"y":1,"z":1}, 3404, True, False)
    obj = Point_Cloud(
        "http://127.0.0.1:5000/",
            np.array(special),
        lines=True,
        size=.1,
        connections=[[4, 20, 10, 9001]]

    )
    uuid2 = d.graph(obj)

    #obj.scale(2,2,2)

    #d.moveObject(uuid2, "point", 1.0, 0, 0)
print(uuid2)
