import cv2
import numpy as np
import time
from collections import defaultdict

# ===== load LBPH model =====
model = cv2.face.LBPHFaceRecognizer_create()
model.read("lbph_model.xml")

label_map = np.load("label_map.npy", allow_pickle=True).item()

# ===== face detector =====
net = cv2.dnn.readNetFromCaffe(
    r"C:\Users\m9426\Desktop\project\Smart Attendance\Advance Level\models\deploy.prototxt",
    r"C:\Users\m9426\Desktop\project\Smart Attendance\Advance Level\models\res10_300x300_ssd_iter_140000.caffemodel"
)

print("Starting camera...")

cap = None
for i in range(3):
    temp = cv2.VideoCapture(i)
    if temp.isOpened():
        cap = temp
        print(f"Camera opened at index {i}")
        break

if cap is None:
    print("❌ No camera found")
    exit()

time.sleep(2)

# ===== tracking memory =====
tracker_next_id = 0
tracked_centers = {}
lost_frames = {}
MAX_LOST = 60

# ===== identity memory =====
identity_registry = {}
active_identity = {}
logical_next_id = 0

# ===== attendance engine =====
attendance = {}
exit_time = {}
last_seen = {}
FRAME_THRESHOLD = 60

# ===== voting buffer =====
name_history = defaultdict(list)

def get_center(x1,y1,x2,y2):
    return ((x1+x2)//2, (y1+y2)//2)

while True:

    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    detections_list = []

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame,(300,300)),
        1.0,(300,300),(104,177,123)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):

        confidence = detections[0,0,i,2]

        if confidence > 0.6:
            box = detections[0,0,i,3:7] * [w,h,w,h]
            x1,y1,x2,y2 = box.astype(int)
            detections_list.append((x1,y1,x2,y2))

    updated_centers = {}
    updated_lost = {}

    detected_names_this_frame = set()

    for (x1,y1,x2,y2) in detections_list:

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        if (y2-y1) < 120 or (x2-x1) < 120:
            continue

        center = get_center(x1,y1,x2,y2)

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(200,200))

        label, dist = model.predict(gray)

        name = "Unknown"
        if dist < 55:
            name = label_map[label]

        # tracker assignment
        assigned_tid = None
        min_dist = 9999

        for tid, prev_center in tracked_centers.items():
            d = np.linalg.norm(np.array(center)-np.array(prev_center))
            if d < 120 and d < min_dist:
                min_dist = d
                assigned_tid = tid

        if assigned_tid is None:
            assigned_tid = tracker_next_id
            tracker_next_id += 1

        updated_centers[assigned_tid] = center
        updated_lost[assigned_tid] = 0

        # voting
        name_history[assigned_tid].append(name)
        if len(name_history[assigned_tid]) > 8:
            name_history[assigned_tid].pop(0)

        final_name = max(set(name_history[assigned_tid]),
                         key=name_history[assigned_tid].count)

        # logical identity
        if final_name != "Unknown":

            detected_names_this_frame.add(final_name)

            if final_name not in identity_registry:
                identity_registry[final_name] = logical_next_id
                logical_next_id += 1

            active_identity[final_name] = identity_registry[final_name]
            logical_id = active_identity[final_name]

            # ===== ENTRY MARK =====
            if final_name not in attendance:
                attendance[final_name] = time.strftime("%H:%M:%S")
                print(f"✅ ENTRY marked → {final_name}")

            last_seen[final_name] = 0

        else:
            logical_id = assigned_tid

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,
                    f"ID {logical_id} {final_name}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,0,255),2)

    # ===== EXIT detection =====
    for student in list(attendance.keys()):

        if student not in detected_names_this_frame:
            last_seen[student] = last_seen.get(student,0) + 1

            if last_seen[student] > FRAME_THRESHOLD:
                if student not in exit_time:
                    exit_time[student] = time.strftime("%H:%M:%S")
                    print(f"❌ EXIT marked → {student}")

    # tracker lost tolerance
    for tid in tracked_centers.keys():
        if tid not in updated_centers:
            lf = lost_frames.get(tid,0) + 1
            if lf < MAX_LOST:
                updated_centers[tid] = tracked_centers[tid]
                updated_lost[tid] = lf

    tracked_centers = updated_centers
    lost_frames = updated_lost

    # ===== UI present counter =====
    present_now = len(attendance) - len(exit_time)

    cv2.putText(frame,
                f"Present: {present_now}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(255,0,0),2)

    cv2.imshow("Smart Attendance System", frame)

    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()