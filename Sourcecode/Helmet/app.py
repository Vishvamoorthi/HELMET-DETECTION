from flask import *
import cv2
from ultralytics import YOLO
import numpy as np
import base64
import os
import time
import threading
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- YOLO ----------------
model = YOLO('besnextt.pt')
names = model.names
import tensorflow as tf
import numpy as np
import cv2

clf_model = tf.keras.models.load_model("helmet_classifier.h5")
# ---------------- GEMINI ----------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyDd2jmrBEE0yXaDphLwFkdFoe-tGSycR9o"
gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.4
)

# ---------------- GLOBAL STATE ----------------
video_path = None
cap = None

detected_plates = []
processed_ids = set()
last_sent_time = {}

video_finished = False
frame_count = 0
SEND_INTERVAL = 5


# ---------------- HELPERS ----------------
def encode_image_to_base64(image):
    _, img_buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(img_buffer).decode("utf-8")


def log_number_plate(track_id, result_text):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("numberplate_log.txt", "a") as f:
        f.write(f"[{now}] Track ID {track_id} - {result_text}\n")

import smtplib
from email.mime.text import MIMEText

def send_fine_email(track_id, plate_text=None, violation=""):
    sender = "vishwamoorthy80@gmail.com"
    password = "gdhhzbodbhmkufsl"
    receiver = "shree231406@gmail.com"

    subject = "Traffic Violation Detected 🚨"

    body = f"""
    Violation: {violation}
    Track ID: {track_id}
    Number Plate: {plate_text if plate_text else "Not detected"}
    """

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        print(f"📧 Email sent for Track {track_id}")
    except Exception as e:
        print("Email error:", e)
def analyze_image(base64_image, track_id):
    try:
        message = HumanMessage(content=[
            {"type": "text", "text": "Extract only number plate"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ])

        response = gemini_model.invoke([message])
        result = response.content.strip()

        print(f"[Track {track_id}] → {result}")

        detected_plates.append(result)
        log_number_plate(track_id, result)
        processed_ids.add(track_id)
        # ✅ SEND EMAIL HERE
        send_fine_email(track_id, plate_text=result, violation="Number Plate Detected")

    except Exception as e:
        print("Gemini error:", e)


# ---------------- STREAM ----------------
def generate_frames():
    global cap, frame_count, video_finished

    while True:
        if cap is None:
            continue

        success, frame = cap.read()

        if not success:
            video_finished = True
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))
        results = model.track(frame, persist=True)

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for track_id, box, class_id in zip(ids, boxes, class_ids):
                x1, y1, x2, y2 = box
                label = names[class_id]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if 'no-helmet' in label:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                elif 'numberplate' in label:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    now = time.time()
                    last = last_sent_time.get(track_id, 0)

                    if track_id not in processed_ids and (now - last > SEND_INTERVAL):
                        last_sent_time[track_id] = now

                        crop = frame[y1:y2, x1:x2]
                        crop = cv2.resize(crop, (800, 100))

                        base64_img = encode_image_to_base64(crop)

                        threading.Thread(
                            target=analyze_image,
                            args=(base64_img, track_id)
                        ).start()

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ---------------- ROUTES ----------------

@app.route('/', methods=['GET', 'POST'])
def upload():
    global video_path, cap, detected_plates, processed_ids, video_finished

    if request.method == 'POST':
        file = request.files['video']

        if file:
            # RESET STATE
            detected_plates = []
            processed_ids = set()
            video_finished = False

            video_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(video_path)

            cap = cv2.VideoCapture(video_path)

            return redirect(url_for('stream'))

    return render_template("upload.html")


@app.route('/stream')
def stream():
    return """
    <html>
    <head>
        <script>
            setInterval(function(){
                fetch('/status')
                .then(res => res.json())
                .then(data => {
                    if(data.finished){
                        window.location.href = "/result";
                    }
                });
            }, 2000);
        </script>
    </head>
    <body style="text-align:center;">
        <h2>🚀 Detection Running...</h2>
        <img src="/video" width="900">
    </body>
    </html>
    """


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    return jsonify({"finished": video_finished})


@app.route('/result')
def result():
    unique_plates = list(set(detected_plates))
    return render_template("result.html", plates=unique_plates)

@app.route('/image', methods=['GET', 'POST'])
def image_upload():
    if request.method == 'POST':
        file = request.files['image']

        if file:
            path = os.path.join("uploads", file.filename)
            file.save(path)

            img = cv2.imread(path)

            # 🔥 Run YOLO
            results = model(img)

            # 🔥 Draw results
            annotated = results[0].plot()

            output_path = os.path.join("uploads", "output_" + file.filename)
            cv2.imwrite(output_path, annotated)

            return f"""
            <h3>Detection Result</h3>
            <img src="/{output_path}" width="600">
            <br><br>
            <a href="/image">⬅ Upload Another Image</a>
            """

    return render_template("image_upload.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory("uploads", filename)


def generate_classification_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # preprocess
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # predict
        pred = clf_model.predict(img, verbose=0)[0][0]

        if pred > 0.5:
            label = "NO HELMET"
            color = (0, 0, 255)
        else:
            label = "HELMET"
            color = (0, 255, 0)

        # draw UI
        cv2.putText(frame, label, (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # encode
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        

@app.route('/video_classify')
def video_classify():
    return Response(generate_classification_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/live-classify')
def live_classify():
    return render_template("live_classify.html")
# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)