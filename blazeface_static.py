"""
Runs inference on images. Define the image files in the list IMAGE_FILES
"""

import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


IMAGE_FILES = ["image.jpg"]


MODEL_SELECTION = 0   #1 for long-range (<5m) detection, 0 for short range (<2m) detection
MIN_CONFIDENCE = 0.5
with mp_face_detection.FaceDetection(model_selection=MODEL_SELECTION, min_detection_confidence=MIN_CONFIDENCE) as face_detection:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)

        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        #If no faces are detected, don't do anything
        if not results.detections:
            continue

        # Draw face detections of each face.
        annotated_image = image.copy()
        for detection in results.detections:
            print('Nose tip:')
            print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
            print('Mouth center:')
            print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.MOUTH_CENTER))
            print('Right eye:')
            print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE))
            print('Left eye:')
            print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE))
            print('Right ear tragion:')
            print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION))
            print('Left ear tragion:')
            print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION))
            mp_drawing.draw_detection(annotated_image, detection)
        cv2.imshow("annotated_image", annotated_image)
        cv2.waitKey(0)
        # cv2.imwrite('annotated_image' + str(idx) + '.png', annotated_image)
