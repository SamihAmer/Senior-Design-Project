import os
os.environ["KIVY_VIDEO"] = "ffpyplayer"
import kivy
import cv2
import ffmpeg
import threading
import multiprocessing
from functools import partial
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.video import Video
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.behaviors import ButtonBehavior
from kivy.properties import ObjectProperty
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.animation import Animation
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.app import App
from kivy.properties import StringProperty

window = (1920,1080)

Window.clearcolor = (0.1882,0.1647,0.1843,0.88)
Window.size = window

class IntroScreen(Screen):
    pass

class MainScreen(Screen):
    pass

class MotionExam(Screen):
    def face_mesh(self):
        import math
        import cv2  # Library to access webcam and video
        import mediapipe as mp  # library for face mesh
        import numpy as np  # library for mathematical computation, we'll use this for ML
        import json  # to output .json files and work with .json format
        import matplotlib.pyplot as plt  # for plotting eventually
        import csv  # could output as a .csv file
        import pandas as pd
        import bottleneck as bn
        import time
        import glob, os
        from zipfile import ZipFile
        from moviepy.editor import VideoFileClip
        import itertools
        import threading
        import sys

        mp_drawing = mp.solutions.drawing_utils  # using drawing utils mediapipe solution to draw
        mp_face_mesh = mp.solutions.face_mesh  # using face_mesh mediapipe solution to apply face mesh
        mp_drawing_syles = mp.solutions.drawing_styles  # this just lets us get fancy with how we can draw what we want
        # mp_face_mesh.landmarks_refinement_calculator_pb2
        # mp_face_mesh.landmarks_smoothing_calculator_pb2
        # specifications of face mesh drawing ( WE DONT WANT TO DRAW THE WHOLE FACE MESH SO THIS IS POINTLESS)
        drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1,
                                              circle_radius=1)
        self.video = cv2.VideoCapture(0)  # using opencv to access camera with VideoCapture function, upload .mp4 files
        # video.set(3, 200)    # width
        # video.set(4, 200)    # height
        # video.set(10, 100)   # brightness
        arr = []
        leftmvmnt_L = []
        leftmvmnt_R = []
        rightmvmnt_L = []
        rightmvmnt_R = []

        alfred = "test"

        def truncate(f, n):
            return math.floor(f * 10 ** n) / 10 ** n

        def rollavg_bottlneck(a, n):  # moving average: https://www.delftstack.com/howto/python/moving-average-python/
            return bn.move_mean(a, window=n, min_count=None)

        done = False
        def animate():
            for c in itertools.cycle(['.  ', '.. ',  '...', '   ']):
                if done:
                    break
                sys.stdout.flush()
                time.sleep(0.5)
            sys.stdout.write('\rDone!     \n')

        t = threading.Thread(target=animate)
        t.start()



        # clip = VideoFileClip(bob)
        # ending = clip.duration*2
        # ending2 = clip.duration

        start = time.time()
        # NOW WE ARE ENTERING THE FACEMESH
        with mp_face_mesh.FaceMesh(min_detection_confidence=0.9,  # initializing detection confidences of face_mesh
                                   min_tracking_confidence=0.9) as face_mesh:
            # while loop to capture video, draw facemesh landmarks
            while True:
                ret, image = self.video.read()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                width = int(self.video.get(3))
                height = int(self.video.get(4))
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                lower_blue = np.array([90, 50, 50])
                upper_blue = np.array([130, 255, 255])
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                result = cv2.bitwise_and(image, image, mask=mask)
                # cv2.imshow('frame', result)
                # cv2.imshow('mask', mask)
                if results.multi_face_landmarks:  # drawing landmarks
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(image=image,
                                                  landmark_list=face_landmarks,
                                                  connections=mp_face_mesh.FACEMESH_IRISES,  # specifying the iris connections
                                                  landmark_drawing_spec=None,
                                                  connection_drawing_spec=mp_drawing_syles.get_default_face_mesh_iris_connections_style())
                        mp_drawing.draw_landmarks(image=image,
                                                  landmark_list=face_landmarks,
                                                  connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                                                  # specifying the iris connections
                                                  landmark_drawing_spec=None,
                                                  connection_drawing_spec=mp_drawing_syles.get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(image=image,
                                                  landmark_list=face_landmarks,
                                                  connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                                                  # specifying the iris connections
                                                  landmark_drawing_spec=None,
                                                  connection_drawing_spec=mp_drawing_syles.get_default_face_mesh_contours_style())
                        h, w, c = image.shape
                        cx_min = w
                        cy_min = h
                        cx_max = cy_max = 0
                        for id, lm in enumerate(face_landmarks.landmark):
                            x, y = int(lm.x * w), int(lm.y * h)
                            # cv2.putText(image, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 0.4, (0, 255, 0), 1 )
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            if cx < cx_min:
                                cx_min = cx
                            if cy < cy_min:
                                cy_min = cy
                            if cx > cx_max:
                                cx_max = cx
                            if cy > cy_max:
                                cy_max = cy
                        cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (255, 255, 0), 2)

                        # RIGHT PUPIL
                        idx = 468  # RIGHT PUPIL LANDMARK
                        # 133 left right eye
                        # 33 is right right eye
                        loc_x = int(face_landmarks.landmark[idx].x * image.shape[1])
                        loc_y = int(face_landmarks.landmark[idx].y * image.shape[0])
                        r_left_x = face_landmarks.landmark[133].x
                        r_right_x = face_landmarks.landmark[33].x
                        rel_rlx = r_left_x - face_landmarks.landmark[idx].x
                        rel_rrx = r_right_x - face_landmarks.landmark[idx].x
                        if abs(rel_rrx) > abs(rel_rlx):
                            print("looking left")
                        elif abs(rel_rrx) < abs(rel_rlx):
                            print("looking right")
                        elif abs(rel_rlx) == abs(rel_rlx):
                            print("centered")
                        rightmvmnt_L.append(rel_rlx)
                        rightmvmnt_R.append(rel_rrx)
                        # print("LEFT", rel_rlx)
                        # print("RIGHT", rel_rrx)
                        cv2.circle(image, (loc_x, loc_y), 2, (255, 255, 255), 2)

                        # LEFT PUPIL
                        idx2 = 473
                        # left right eye 362
                        # left left eye 263
                        loc_x2 = int(face_landmarks.landmark[idx2].x * image.shape[1])
                        loc_y2 = int(face_landmarks.landmark[idx2].y * image.shape[0])
                        l_right_x = face_landmarks.landmark[362].x
                        l_left_x = face_landmarks.landmark[263].x
                        rel_lrx = l_right_x - face_landmarks.landmark[idx2].x
                        rel_llx = l_left_x - face_landmarks.landmark[idx2].x
                        # if abs(rel_lrx) > abs(rel_llx): #if the distance from the rightmost eye point is greater than the distance from leftmost eye point you are looking left
                        #     print("looking left")
                        # elif abs(rel_lrx) < abs(rel_llx):
                        #     print("looking right")
                        # elif abs(rel_llx) == abs(rel_llx):
                        #     print("centered")
                        leftmvmnt_L.append(rel_llx)
                        leftmvmnt_R.append(rel_lrx)
                        # print("LEFT", rel_llx)
                        # print("RIGHT", rel_lrx)
                        cv2.circle(image, (loc_x2, loc_y2), 2, (255, 255, 255), 2)
                # Display Output Image
                cv2.imshow("Face Mesh", image)
                k = cv2.waitKey(1)
                # if k == ord('q'):
                    # time_current_a = time.time() - start
                    # cframe = video.get(cv2.CAP_PROP_POS_FRAMES) # retrieves the current frame number
                    # tframe = video.get(cv2.CAP_PROP_FRAME_COUNT) # get total frame count
                    # fps = video.get(cv2.CAP_PROP_FPS)  #get the FPS of the videos
                    # print("real-time: ", time_current_a)
                    # print("frame number: ", cframe, "/",tframe)

                if k == ord('q'):
                    # ELAPSED TIME CALCULATION
                    done = True
                    # print("\nframes processed: " + str(cframe) + "/" + str(tframe))
                    end = time.time()
                    elapsed = (end - start)/2
                    print("time of .mp4: " + str(elapsed) + " vs. time to process file: " + str(end-start))
                    print(len(leftmvmnt_L), len(leftmvmnt_R), len(rightmvmnt_L), len(rightmvmnt_R))
                    arrlen = len(leftmvmnt_L)
                    t = np.zeros(arrlen)
                    for i in range(0, len(t)):
                        t[i] = elapsed * (i / (arrlen - 1))
                    print(t)

                    #STORE FRAME AND NON-NORMALIZED DATA FOR ACCURATE PREDICTION
                    unfiltered_right =  np.add(rightmvmnt_L, rightmvmnt_R)
                    dfx = pd.DataFrame({"ELAPSED_SECONDS" : t, "rightmvmnt_L": rightmvmnt_L, "rightmvmnt_R": rightmvmnt_R, "RELATIVE_POSITION" : unfiltered_right})
                    # dfx.to_csv((alfred+"UNFILTERED_RIGHT_EYE.csv"), index=False)
                    unfiltered_left = np.add(leftmvmnt_L, leftmvmnt_R)
                    dfy = pd.DataFrame({"ELAPSED_SECONDS" : t, "leftmvmnt_L": leftmvmnt_L, "leftmvmnt_R": leftmvmnt_R, "RELATIVE_POSITION" : unfiltered_left})
                    # dfy.to_csv((alfred+"UNFILTERED_LEFT_EYE.csv"), index=False)

                    # NORMALIZATION IMPLEMENTATION
                    norm1 = np.linalg.norm(rightmvmnt_L)
                    normalizeR_L = rightmvmnt_L / norm1
                    norm2 = np.linalg.norm(rightmvmnt_R)
                    normalizeR_R = rightmvmnt_R / norm2
                    norm3 = np.linalg.norm(leftmvmnt_L)  # normalization constant
                    normalizeL_L = leftmvmnt_L / norm3  # normalize
                    norm4 = np.linalg.norm(leftmvmnt_R)
                    normalizeL_R = leftmvmnt_R / norm4

                    # RIGHT PUPIL MOTION CALCULATOR
                    # over_all_right = rollavg_bottlneck(normalizeR_L, 5) + rollavg_bottlneck(normalizeR_R, 5)
                    over_all_right = np.add(rollavg_bottlneck(normalizeR_L, 5), rollavg_bottlneck(normalizeR_R, 5))
                    s1 = normalizeR_L + normalizeR_R
                    df1 = pd.DataFrame({"ELAPSED_SECONDS": t,"NORMALIZED": s1, "MOVING_AVG": over_all_right})
                    # name1 = alfred + "right_pupil_data.csv"
                    # df1.to_csv((alfred+ "right_pupil_data.csv"), index=False)
                    fig1 = plt.figure("Overall Right Eye Movement: Relative Position vs. Time Elapsed (s)")
                    plt.plot(t, over_all_right, c= "black", label = 'normalized & moving average')
                    plt.plot(t, s1, c = "red", label = 'normalized')
                    plt.plot(t, unfiltered_right, c = "green", label = 'unfiltered')
                    plt.title("Right Pupil Movements")
                    plt.legend(loc="upper left")
                    plt.savefig(("right_pupil.png"), format="png", dpi=300)

                    # LEFT PUPIL MOTION CALCULATOR
                   # over_all_left = rollavg_bottlneck(normalizeL_L, 5) + rollavg_bottlneck(normalizeL_R, 5)
                    over_all_left = np.add(rollavg_bottlneck(normalizeL_L, 5),rollavg_bottlneck(normalizeL_R, 5))
                    s2 = normalizeL_L + normalizeL_R
                    df = pd.DataFrame({"ELAPSED_SECONDS": t, "NORMALIZED": s2, "MOVING_AVG": over_all_left})
                    # df.to_csv((alfred + "left_pupil_data.csv"), index=False)
                    fig2 = plt.figure("Overall Left Eye Movement: RelativePosition vs. Time Elapsed (s)")
                    plt.plot(t, over_all_left, c="black", label = 'normalized & moving average')
                    plt.plot(t, s2, c = "red", label = 'normalized')
                    plt.plot(t, unfiltered_left, c = "green", label = 'unfiltered')
                    plt.title("Left Pupil Movements")
                    plt.legend(loc="upper left")
                    plt.savefig(("left_pupil.png"), format="png", dpi=300)
                    # plt.show()

                    # Fourier Transform of Unfiltered Data
                    # over_all = over_all_right + over_all_left  # normalized + moving average
                    # bakup = normalizeR_L + normalizeR_R + normalizeL_L + normalizeL_R  # normalized
                    # bakup2 = rightmvmnt_L + rightmvmnt_R + leftmvmnt_L + leftmvmnt_R  # untampered
                    # fft = np.fft.fft(bakup2)
                    # fftfreq = np.fft.fftfreq(len(bakup2))  # try t
                    # plt.plot(fftfreq, fft)
                    # plt.savefig((alfred + "FFT.png"), format="png")
                    # plt.show()
                    break
        # cv2.imshow('Hidden', frame)
        # cv2.waitKey(1)
        self.video.release()
        cv2.destroyAllWindows()
    t2 = threading.Thread(target=face_mesh, args=(1,))
    t2.start()

    # b = multiprocessing.Process(target=face_mesh)
    # b.start()

class Results(Screen):
    pass

class ScreenManagement(ScreenManager):
    pass

class CameraPreview(Image):
    def __init__(self, **kwargs):
        super(CameraPreview, self).__init__(**kwargs)
        #Connect to 0th camera
        self.capture = cv2.VideoCapture(0)
        #Set drawing interval
        Clock.schedule_interval(self.update, 1.0 / 30)
    #Drawing method to execute at intervals
    def update(self, dt):
        #Load frame
        ret, self.frame = self.capture.read()
        #Convert to Kivy Texture
        buf = cv2.flip(self.frame, 0).tobytes()
        texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        #Change the texture of the instance
        self.texture = texture

class BtnTextInput(BoxLayout):
    pass

class uiApp(App):
    def build(self):
        return

# def open_parent():
#     uiApp().run()
# def open_child():
#     app.Iris_Face_Mesh()
#
# if __name__ == '__main__':
#     a = multiprocessing.Process(target=open_parent)
#     a.start()
#     b = multiprocessing.Process(target=open_child)
#     b.start()


uiApp().run()
