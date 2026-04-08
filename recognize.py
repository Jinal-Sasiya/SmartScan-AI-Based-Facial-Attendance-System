import cv2
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

with open("face_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = data['embeddings']
known_names = data['names']

net = cv2.dnn.readNetFromCaffe(
    r"C:\Users\m9426\Desktop\project\Smart Attendance\Advance Level\models\deploy.prototxt",
    r"C:\Users\m9426\Desktop\project\Smart Attendance\Advance Level\models\res10_300x300_ssd_iter_140000.caffemodel"
)

def get_embedding(face):
    face = cv2.resize(face, (100,100))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face / 255.0
    return face.reshape(1,-1)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret,frame = cap.read()
    if not ret:
        break
    h,w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)),
                                 1.0,(300,300),(104.0,177.0,123.0))

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

            emb = get_embedding(face)

            sims = cosine_similarity(emb, known_embeddings)[0]
            best_idx = np.argmax(sims)

            name = "Unknown"

            print("Similarity:", sims[best_idx])

            if sims[best_idx] > 0.85:
                name = known_names[best_idx]

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, name, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,0,255), 2)
            
    cv2.imshow("Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()