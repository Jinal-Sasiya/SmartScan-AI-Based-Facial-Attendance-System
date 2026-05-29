import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from database import mark_attendance, attendance_exists
from datetime import datetime
import os
from flask import (
    Flask,
    render_template,
    send_file,
    Response,
    request,
    redirect,
    session,
    url_for
)


# =========================
# LOAD LBPH MODEL
# =========================

model = cv2.face.LBPHFaceRecognizer_create()

model.read("lbph_model.xml")

label_map = np.load(
    "label_map.npy",
    allow_pickle=True
).item()

# =========================
# FACE DETECTOR
# =========================

face_detector = cv2.CascadeClassifier(

    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)



app = Flask(__name__)

app.secret_key = "smart_attendance_secret_key"

DB_NAME = 'attendance.db'

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "1234"

# ===== load LBPH model =====

model = cv2.face.LBPHFaceRecognizer_create()
model.read("lbph_model.xml")

label_map = np.load("label_map.npy", allow_pickle=True).item()

# ===== load DNN face detector =====

net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

camera = cv2.VideoCapture(0)

marked_names = set()

last_marked = {}

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(

            gray,

            scaleFactor=1.2,

            minNeighbors=5,

            minSize=(80,80)
        )

        for (x,y,w,h) in faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face, (200,200))

            label, confidence = model.predict(face)

            name = "Unknown"

            accuracy = round(
                100 - confidence,
                1
            )

            if confidence < 70:

                name = label_map[label]

                # avoid multiple attendance entries

                now = datetime.now()

                current_time = now.strftime("%H:%M:%S")

                current_date = now.strftime("%Y-%m-%d")

                if name not in last_marked:

                    if not attendance_exists(
                        name,
                        current_date
                    ):

                        mark_attendance(

                            name,
                            current_date,
                            current_time,
                            accuracy
                        )

                    last_marked[name] = True

            # rectangle

            cv2.rectangle(

                frame,

                (x,y),

                (x+w,y+h),

                (0,255,0),

                2
            )

            # text

            cv2.putText(

                frame,

                f"{name} {accuracy}%",

                (x,y-10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0,255,255),

                2
            )

        ret, buffer = cv2.imencode(

            '.jpg',
            frame
        )

        frame = buffer.tobytes()

        yield (

            b'--frame\r\n'

            b'Content-Type: image/jpeg\r\n\r\n'

            + frame +

            b'\r\n'
        )





@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if (
            username == ADMIN_USERNAME and
            password == ADMIN_PASSWORD
        ):

            session["user"] = username

            return redirect(url_for("index"))

        else:

            return render_template(
                "login.html",
                error="Invalid Credentials"
            )

    return render_template("login.html")

@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_NAME)
    query = "SELECT * FROM attendance"
    df = pd.read_sql_query(query, conn)
    conn.close()
    records = df.to_dict(orient='records')
    return render_template('index.html', records=records)

@app.route("/export")
def export_csv():

    conn = sqlite3.connect(DB_NAME)

    query = "SELECT * FROM attendance"

    df = pd.read_sql_query(query, conn)

    conn.close()

    csv_file = "attendance.csv"

    df.to_csv(csv_file, index=False)

    return send_file(csv_file, as_attachment=True)

@app.route('/analytics')
def analytics():

    conn = sqlite3.connect("attendance.db")

    cursor = conn.cursor()

    # total attendance per student

    cursor.execute("""

        SELECT name, COUNT(*)

        FROM attendance

        GROUP BY name

    """)

    data = cursor.fetchall()

    conn.close()

    labels = [row[0] for row in data]

    values = [row[1] for row in data]

    return render_template(

        "analytics.html",

        labels=labels,

        values=values
    )


def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
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

                now = datetime.now()

                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                if name != "Unknown":

                    if name not in marked_names:

                        if not attendance_exists(name, date):

                            mark_attendance(name, date, time, confidence_score)

                            print(f"{name} attendance marked")

                        marked_names.add(name)

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(
                    frame,
                    f"{name} {confidence_score}%",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame + b'\r\n')
        


@app.route("/camera")
def camera_page():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("camera.html")

@app.route('/video_feed')
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':

        name = request.form['name']

        path = f"dataset/{name}"

        os.makedirs(path, exist_ok=True)

        camera = cv2.VideoCapture(0)

        detector = cv2.CascadeClassifier(

            cv2.data.haarcascades +
            'haarcascade_frontalface_default.xml'
        )

        count = 0

        while True:

            success, frame = camera.read()

            if not success:
                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = detector.detectMultiScale(

                gray,

                scaleFactor=1.2,

                minNeighbors=5
            )

            for (x,y,w,h) in faces:

                count += 1

                face = gray[y:y+h, x:x+w]

                face = cv2.resize(
                    face,
                    (200,200)
                )

                cv2.imwrite(

                    f"{path}/{count}.jpg",

                    face
                )

                cv2.rectangle(

                    frame,

                    (x,y),

                    (x+w,y+h),

                    (0,255,0),

                    2
                )

                cv2.putText(

                    frame,

                    f"Capturing {count}/50",

                    (x,y-10),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (0,255,255),

                    2
                )

            cv2.imshow(
                "Face Capture",
                frame
            )

            if cv2.waitKey(1) == 27 or count >= 50:
                break

        camera.release()

        cv2.destroyAllWindows()

        return redirect('/')

    return render_template('register.html')

@app.route("/capture/<name>")
def capture_student(name):

    save_path = f"dataset/{name}"

    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)

    count = 0

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

                file_path = f"{save_path}/{count}.jpg"

                cv2.imwrite(file_path, gray)

                count += 1

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(
                    frame,
                    f"Capturing {count}/50",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

        cv2.imshow("Capturing Faces", frame)

        if count >= 50:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()

    train_model()

    return f"{name} registered successfully!"

def train_model():

    faces = []

    labels = []

    label_map = {}

    current_id = 0

    dataset_path = "dataset"

    for person_name in os.listdir(dataset_path):

        person_folder = os.path.join(dataset_path, person_name)

        if not os.path.isdir(person_folder):
            continue

        label_map[current_id] = person_name

        for image_name in os.listdir(person_folder):

            image_path = os.path.join(person_folder, image_name)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            faces.append(img)

            labels.append(current_id)

        current_id += 1

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.train(faces, np.array(labels))

    recognizer.save("lbph_model.xml")

    np.save("label_map.npy", label_map)

    print("Model trained successfully")

@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect(url_for("login"))

@app.route('/train_model')
def train_model():

    subprocess.run(

        ['python', 'train_lbph.py']
    )

    return redirect('/')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)