
import cv2   #Library to access webcam and video
import mediapipe as mp   # library for face mesh
import numpy as np  # library for mathematical computation, we'll use this for ML
import json  # to output .json files and work with .json format
import matplotlib.pyplot as plt   # for plotting eventually
import csv  # could output as a .csv file
import pandas # pandas is a data structuring library. Makes dataframes to use data


mp_drawing = mp.solutions.drawing_utils       #using drawing utils mediapipe solution to draw
mp_face_mesh = mp.solutions.face_mesh         #using face_mesh mediapipe solution to apply face mesh


mp_face_mesh.landmarks_refinement_calculator_pb2


drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255),thickness= 1,
                                      circle_radius = 1)                     #specifications of face mesh drawing

video = cv2.VideoCapture(0)      # using opencv to access camera with VideoCapture function, upload .mp4 files


with mp_face_mesh.FaceMesh(min_detection_confidence = 0.8,       #initializing detection confidences of face_mesh
                           min_tracking_confidence= 0.8) as face_mesh:


    # while loop to capture video, draw facemesh landmarks
    while True:
        ret, image = video.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(image)
        #print(results)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



        if results.multi_face_landmarks:                                 #drawing landmarks
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image = image ,
                                            landmark_list = 0,
                                            connections = mp_face_mesh.FACEMESH_IRISES,  #specifying the iris connections
                                            landmark_drawing_spec = drawing_spec,
                                            connection_drawing_spec = drawing_spec)
                #
                # h, w, c = image.shape
                # cx_min = w
                # cy_min = h
                # cx_max = cy_max = 0
                # for id, lm in enumerate(face_landmarks.landmark):
                #
                #     x,y = int(lm.x*w), int(lm.y*h)
                #     cv2.putText(image, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1 )
                #     cx, cy = int(lm.x * w), int(lm.y * h)
                #     if cx < cx_min:
                #         cx_min = cx
                #     if cy < cy_min:
                #         cy_min = cy
                #     if cx > cx_max:
                #         cx_max = cx
                #     if cy > cy_max:
                #         cy_max = cy
                # cv2.rectangle(image, (cx_min, cy_min), (cx_max, cy_max), (468, 473, 0), 2)

                #RIGHT IRIS
                landmarks_list = [468]
                for idx in landmarks_list:
                    loc_x = int(face_landmarks.landmark[idx].x * image.shape[1])
                    loc_y = int(face_landmarks.landmark[idx].y * image.shape[0])
                    print("Right location", loc_x, loc_y)
                    data = {'xcoord': loc_x, 'ycoord:': loc_y}
                   # graphdata = [loc_x, loc_y]     for potential graph usage
                    with open('RIGHT_IRIS.json', 'a') as f: json.dump(data, f, indent=2)
                  #  with open('graphdata1.json', 'a') as f: json.dump(graphdata, f, indent=2)    potential graph application
                    cv2.circle(image, (loc_x, loc_y), 2, (255, 255, 255), 2)
                                                                                                                                                                               
                



                #LEFT IRIS
                landmarks_list2 = [473]
                for idx2 in landmarks_list2:
                    loc_x2 = int(face_landmarks.landmark[idx2].x * image.shape[1])
                    loc_y2 = int(face_landmarks.landmark[idx2].y * image.shape[0])
                    print("Left location",loc_x2, loc_y2)
                    data2 = {'xcoord': loc_x2, 'ycoord:': loc_y2}
                    with open('LEFT_IRIS.json', 'a') as f: json.dump(data2,f,indent=2)
                    cv2.circle(image, (loc_x2, loc_y2), 2, (255, 255, 255), 2)



        cv2.imshow("Face Mesh", image)                    #display output image
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()



    