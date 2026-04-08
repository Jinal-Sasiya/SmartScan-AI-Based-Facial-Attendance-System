import cv2
import numpy as np

# ===== load trained model =====
model = cv2.face.LBPHFaceRecognizer_create()
model.read("lbph_model.xml")

label_map = np.load("label_map.npy", allow_pickle=True).item()

# ===== load face detector =====
net = cv2.dnn.readNetFromCaffe(
    r"C:\Users\m9426\Desktop\project\Smart Attendance\Advance Level\models\deploy.prototxt",
    r"C:\Users\m9426\Desktop\project\Smart Attendance\Advance Level\models\res10_300x300_ssd_iter_140000.caffemodel"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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

            name = "Unknown"

            if dist < 70:
                name = label_map[label]

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,
                        f"{name} {round(dist,1)}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,(0,0,255),2)

    cv2.imshow("LBPH Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

