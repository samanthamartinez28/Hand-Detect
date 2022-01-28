import mediapipe as mp
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
state=False
counter = 0
stage = 'Not Littering'
stagetwo= 'Littering'

mp_drawing = mp.solutions.drawing_utils;
mp_holistic = mp.solutions.holistic;




def calculate_angle( a,b,c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    #mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections



def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=0, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

    # Draw right-hand landmarks
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(240, 0, 0), thickness=2, circle_radius=2))

    # Draw left-hand landmarks
    #mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              #mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                              #mp_drawing.DrawingSpec(color=(240, 0, 0), thickness=2, circle_radius=2))

    # Draw body detections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        #print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)


        # Extract landmarks
        try:
            landmarks = results.right_hand_landmarks.landmark
            #print(landmarks)

            index = [landmarks[mp_holistic.HandLandmark.INDEX_FINGER_DIP.value].x,
                     landmarks[mp_holistic.HandLandmark.INDEX_FINGER_DIP.value].y]
            wrist = [landmarks[mp_holistic.HandLandmark.INDEX_FINGER_PIP.value].x,
                     landmarks[mp_holistic.HandLandmark.INDEX_FINGER_PIP.value].y]
            thumb = [landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].x,
                     landmarks[mp_holistic.HandLandmark.INDEX_FINGER_MCP.value].y]
            angle = calculate_angle(index, wrist, thumb)

            #Visualize angle
            cv2.putText(image, str(angle),
                       tuple(np.multiply(wrist, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            # Curl counter logic
            #if angle <0:
                # stage = "littering"






        except:
            pass

            #Render curl counter
            #Setup status box
            #cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

            # Rep data
            #cv2.putText(image, 'REPS', (15, 12),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            #cv2.putText(image, str(counter),
                        #(10, 60),
                        #cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Stage data
            if(state == False):
                cv2.putText(image, 'STAGE', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage,
                        (60, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        else:
            if (angle>160):
                cv2.putText(image, stagetwo,
                            (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)


            # Show to screen


        cv2.imshow('Model Detections', image)



        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


    #def extract_keypoints(results):
        #pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        #lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        #rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        #return np.concatenate([rh])