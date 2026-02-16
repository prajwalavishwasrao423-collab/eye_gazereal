import cv2

# Force Windows DirectShow backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("✅ Camera opened")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Frame not read")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
