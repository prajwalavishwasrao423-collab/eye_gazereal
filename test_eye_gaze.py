import cv2
import mediapipe as mp
import numpy as np
import time

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

BLINK_THRESHOLD = 0.20
last_blink_time = 0
BLINK_COOLDOWN = 1.5


def eye_ratio(landmarks, eye):
    left = np.array(landmarks[eye[0]])
    right = np.array(landmarks[eye[1]])
    return np.linalg.norm(left - right)


while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera frame not read")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    direction = None

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        landmarks = [(int(p.x * w), int(p.y * h)) for p in lm]

        left_eye_width = eye_ratio(landmarks, LEFT_EYE)
        right_eye_width = eye_ratio(landmarks, RIGHT_EYE)

        if left_eye_width < BLINK_THRESHOLD * w:
            if time.time() - last_blink_time > BLINK_COOLDOWN:
                direction = "BLINK"
                last_blink_time = time.time()

        left_iris_x = np.mean([landmarks[i][0] for i in LEFT_IRIS])
        left_eye_center = np.mean(
            [landmarks[LEFT_EYE[0]][0], landmarks[LEFT_EYE[1]][0]]
        )

        offset = left_iris_x - left_eye_center

        if offset > 5:
            direction = "RIGHT"
        elif offset < -5:
            direction = "LEFT"

        if direction:
            print("➡️", direction)

    cv2.imshow("Eye Gaze Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

