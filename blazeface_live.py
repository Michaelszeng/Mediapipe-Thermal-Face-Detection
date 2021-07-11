"""
Runs inference on a live video feed from the Lepton 3.5. You may have to change CAMERA_ID depending on your setup.
"""

import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

CAMERA_ID = 1


#Capture a video feed
cap = cv2.VideoCapture(CAMERA_ID)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():

        #Read a new frame from the videofeed
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2GRAY)
        # Convert back to RGB so we have a black/white image with 3 color channels
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        #Show the annotated image
        cv2.imshow('MediaPipe Face Detection', image)

        key = cv2.waitKey(20)
        if key == 27:   #exit on ESC
            break

#Release video capture when the program is done
cap.release()
