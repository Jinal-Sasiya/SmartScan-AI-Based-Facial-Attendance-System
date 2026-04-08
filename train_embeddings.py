import cv2
import os
import numpy as np
from sklearn.preprocessing import normalize
import pickle 

dataset_path = "dataset"

embeddings = []
names = []

def get_embedding(face):

    try:
        face = cv2.resize(face, (100,100))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.flatten().astype("float32")
        face = face / 255.0
        return face

    except:
        return None

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        emb = get_embedding(img)

        if emb is not None and emb.shape[0] == 10000:
            embeddings.append(emb)
            names.append(person)

embeddings = np.array(embeddings)
embeddings = normalize(embeddings)

data = {"embeddings": embeddings, "names": names}

with open("face_embeddings.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ Embeddings training completed")
print("Total samples:", len(names))