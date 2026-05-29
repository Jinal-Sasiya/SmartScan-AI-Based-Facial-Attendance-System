import cv2
import os
import numpy as np

# =========================
# DATASET PATH
# =========================

dataset_path = "dataset"

faces = []
labels = []

label_map = {}

current_label = 0

# =========================
# READ DATASET
# =========================

for person in os.listdir(dataset_path):

    person_path = os.path.join(
        dataset_path,
        person
    )

    # skip non-folder files

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for img_name in os.listdir(person_path):

        img_path = os.path.join(
            person_path,
            img_name
        )

        img = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        # skip broken images

        if img is None:
            continue

        img = cv2.resize(
            img,
            (200,200)
        )

        faces.append(img)

        labels.append(current_label)

    current_label += 1

# =========================
# CONVERT TO NUMPY
# =========================

faces = np.array(faces)

labels = np.array(labels)

# =========================
# TRAIN MODEL
# =========================

model = cv2.face.LBPHFaceRecognizer_create()

model.train(
    faces,
    labels
)

# =========================
# SAVE MODEL
# =========================

model.write("lbph_model.xml")

np.save(
    "label_map.npy",
    label_map
)

print("\n✅ LBPH Model Training Complete")
print("✅ Model saved as lbph_model.xml")
print("✅ Label map saved as label_map.npy")
print(f"✅ Total students trained: {len(label_map)}")