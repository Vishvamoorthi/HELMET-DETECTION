import cv2
from ultralytics import YOLO
import numpy as np
import base64
import os
import time
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import threading
import cvzone
from datetime import datetime

# Load YOLOv8 model
model = YOLO('besnextt.pt')
names = model.names

# Initialize Gemini model
GOOGLE_API_KEY = "AIzaSyDQOpdQ-1-D9B35moZOCG87xyqzbfQArjU"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

# Open video
cap = cv2.VideoCapture("helmet.mp4")

# Encode image to base64
def encode_image_to_base64(image):
    _, img_buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(img_buffer).decode("utf-8")

# Save result to file
def log_number_plate(track_id, result_text):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("numberplate_log.txt", "a") as f:
        f.write(f"[{now}] Track ID {track_id} - Number Plate: {result_text}\n")

# Gemini analysis thread
def analyze_image_with_gemini_base64(base64_image, track_id):
    try:
        message = HumanMessage(
            content=[
                {"type": "text", "text": f"Analyze this image and extract only:\n\n|Number Plate|\n|-------------|"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )
        response = gemini_model.invoke([message])
        result_text = response.content.strip()
        print(f"[Track ID {track_id}] Gemini Response:\n{result_text}\n")

        # Log result
        log_number_plate(track_id, result_text)

        # Mark as processed
        processed_ids.add(track_id)

    except Exception as e:
        print(f"[Track ID {track_id}] Error with Gemini:", e)


# Trackers
last_sent_time = {}
processed_ids = set()
SEND_INTERVAL = 5  # seconds
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
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

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1 + 3, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#            cvzone.putTextRect(frame, f'{track_id}', (x2, y2), 1, 1)

            # Highlight based on class
            if 'no-helmet' in label:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            elif 'numberplate' in label:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                current_time = time.time()
                last_time = last_sent_time.get(track_id, 0)

                if (track_id not in processed_ids) and (current_time - last_time >= SEND_INTERVAL):
                    last_sent_time[track_id] = current_time
                    crop = frame[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (800, 100))
                    base64_img = encode_image_to_base64(crop)
                    threading.Thread(target=analyze_image_with_gemini_base64,
                                     args=(base64_img, track_id)).start()

    import cv2

    cv2.namedWindow("FRAME", cv2.WINDOW_NORMAL)
    cv2.imshow("FRAME", frame)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
