import cv2
import numpy as np
from database import mark_attendance, attendance_exists
from datetime import datetime

# ===== load trained model =====
model = cv2.face.LBPHFaceRecognizer_create()
model.read("lbph_model.xml")

label_map = np.load("label_map.npy", allow_pickle=True).item()

# ===== load face detector =====
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

marked_names = set()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300,300)),
        1.0,
        (300,300),
        (104,177,123)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):

        confidence = detections[0,0,i,2]

        if confidence > 0.6:

            box = detections[0,0,i,3:7] * [w,h,w,h]
            x1,y1,x2,y2 = box.astype(int)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200,200))

            label, dist = model.predict(gray)

            confidence_score = round(100 * (1 - dist / 300))

            name = "Unknown"

            if dist < 70:
                name = label_map[label]

            # ===== date and time =====
            now = datetime.now()

            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            # ===== mark attendance =====
            if name != "Unknown":

                if name not in marked_names:

                    if not attendance_exists(name, date):

                        mark_attendance(name, date, time, confidence_score)

                        print(f"{name} attendance marked")

                    else:
                        print(f"{name} already marked today")

                    marked_names.add(name)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,
                        f"{name} {confidence_score}%",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,(0,0,255),2)

    cv2.imshow("LBPH Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

