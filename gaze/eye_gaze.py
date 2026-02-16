import cv2
import mediapipe as mp
import numpy as np
import time

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

last_blink_time = 0
BLINK_COOLDOWN = 1.5

current_gaze = "CENTER"


def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


def get_gaze():
    global current_gaze, last_blink_time

    ret, frame = cap.read()
    if not ret:
        current_gaze = "CENTER"
        return current_gaze

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    current_gaze = "CENTER"

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        def pt(i):
            return np.array([int(landmarks[i].x * w), int(landmarks[i].y * h)])

        left_eye = [pt(i) for i in [33, 160, 158, 133, 153, 144]]
        right_eye = [pt(i) for i in [362, 385, 387, 263, 373, 380]]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

        now = time.time()
        if ear < 0.20 and now - last_blink_time > BLINK_COOLDOWN:
            current_gaze = "BLINK"
            last_blink_time = now
            return current_gaze

        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
        eye_center_x = (left_center[0] + right_center[0]) / 2

        if eye_center_x < w * 0.45:
            current_gaze = "LEFT"
        elif eye_center_x > w * 0.55:
            current_gaze = "RIGHT"

    return current_gaze








