from flask import Flask, render_template, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import requests

app = Flask(__name__)

# ================= CAMERA SETUP =================
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Camera not opened")
else:
    print("✅ Camera opened successfully")

# ================= MEDIAPIPE =================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
# ================= STATE =================
OPTIONS = ["water", "light", "fan", "emergency"]
current_index = 0
last_blink_time = 0
blink_cooldown = 1.2
gaze_state = "CENTER"
last_selected = None

ESP32_IP = "http://192.168.1.100"

# ================= UTILS =================
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ================= CAMERA THREAD =================
def camera_loop():
    global current_index, last_blink_time, gaze_state, last_selected

    while True:
        success, frame = cap.read()

        if not success:
            print("⚠ Failed to read frame")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = face_mesh.process(rgb)

        # ================= NO FACE =================
        if not result.multi_face_landmarks:
            gaze_state = "NO FACE"
            print("⚠ NO FACE DETECTED")

            # Show camera anyway
            cv2.imshow("Camera Debug", frame)
            cv2.waitKey(1)

            time.sleep(0.05)
            continue

        landmarks = result.multi_face_landmarks[0].landmark

        # ================= BLINK DETECTION =================
        left_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in [33,160,158,133,153,144]])
        right_eye = np.array([[landmarks[i].x, landmarks[i].y] for i in [362,385,387,263,373,380]])

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

        if ear < 0.20:
            if time.time() - last_blink_time > blink_cooldown:
                last_blink_time = time.time()

                selected = OPTIONS[current_index]
                print("✅ BLINK SELECT:", selected)

                last_selected = selected

                try:
                    response = requests.get(f"{ESP32_IP}/{selected}", timeout=1)
                    print("📡 ESP32 Response:", response.status_code)
                except Exception as e:
                    print("❌ ESP32 not reachable:", e)

        # ================= GAZE DETECTION =================
        left_iris_x = landmarks[468].x
        right_iris_x = landmarks[473].x
        eye_center = (left_iris_x + right_iris_x) / 2

        if eye_center < 0.45:
            if gaze_state != "LEFT":
                gaze_state = "LEFT"
                current_index = max(0, current_index - 1)
                print("⬅ LEFT")

        elif eye_center > 0.55:
            if gaze_state != "RIGHT":
                gaze_state = "RIGHT"
                current_index = min(len(OPTIONS) - 1, current_index + 1)
                print("➡ RIGHT")

        else:
            gaze_state = "CENTER"

        # ================= SHOW CAMERA =================
        cv2.imshow("Camera Debug", frame)
        cv2.waitKey(1)

        time.sleep(0.05)

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/status")
def status():
    global last_selected

    response = {
        "selected": OPTIONS[current_index],
        "index": current_index,
        "gaze": gaze_state,
        "blink": last_selected
    }

    # Send blink only once
    last_selected = None

    return jsonify(response)

# ================= START THREAD =================
threading.Thread(target=camera_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)


