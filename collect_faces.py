import cv2
import os
import time
time.sleep(2)

name = input("Enter student name: ")

save_dir = f"dataset/{name}"
os.makedirs(save_dir, exist_ok=True)

print("Loading face detector model...")

net = cv2.dnn.readNetFromCaffe(
    r"C:\Users\m9426\Desktop\project\Smart Attendance\Advance Level\models\deploy.prototxt",
    r"C:\Users\m9426\Desktop\project\Smart Attendance\Advance Level\models\res10_300x300_ssd_iter_140000.caffemodel"
)

print("Opening camera...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

count = 0

while True:
    print("Frame running...")
    ret, frame = cap.read()

    if not ret:
        print("❌ Frame not received")
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype(int)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            cv2.imwrite(f"{save_dir}/{count}.jpg", face)
            count += 1

            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)

    cv2.putText(frame, f"Captured: {count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Dataset Collector", frame)

    if cv2.waitKey(1) & 0xFF == 27 or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()