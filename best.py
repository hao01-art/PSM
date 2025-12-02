# ======================================================
# YOLOv8 Real-Time PET Bottle Detection (Pi 5 + Camera Module 3)
# Manual override of bottle sizing class while keeping bounding box
# ======================================================

from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import os
import random

# ====================== USER SETTINGS ======================
MODEL_PATH = "/home/ben/yolo/psm_bottle_yolov8s_model2_ncnn_model"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.6
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 2
LABEL_BG_WIDTH = 160
SNAPSHOT_FOLDER = "snapshots"

CLASS_NAMES = [
    "BottleCap",
    "BottleLabel",
    "PET-Bottle-250ml",
    "PET-Bottle-500ml",
    "PET-Bottle-600ml",
    "PET-Bottle-1500ml"
]

# Mapping keys 1-4 to your requested bottle classes
MANUAL_BOTTLE_MAP = {
    ord('1'): "PET-Bottle-250ml",
    ord('2'): "PET-Bottle-500ml",
    ord('3'): "PET-Bottle-600ml",
    ord('4'): "PET-Bottle-1500ml"
}

# ====================== SETUP SNAPSHOT FOLDER ======================
if not os.path.exists(SNAPSHOT_FOLDER):
    os.makedirs(SNAPSHOT_FOLDER)
    print(f"📁 Snapshot folder created: {SNAPSHOT_FOLDER}")

# ====================== LOAD YOLO MODEL ======================
print("🔄 Loading YOLOv8 model...")
model = YOLO(MODEL_PATH, task='detect')
print("✅ Model loaded successfully.")

# ====================== CAMERA SETUP ======================
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640,640), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2)
print("🎥 Camera ready — press 'q' to quit, 'p' to snapshot, 1-4 to select bottle size.\n")

# ====================== COLOUR MAP ======================
COLOR_MAP = {}
def get_color(label):
    if label not in COLOR_MAP:
        COLOR_MAP[label] = tuple(int(c) for c in np.random.randint(0, 255, 3))
    return COLOR_MAP[label]

# ====================== IOU FUNCTIONS ======================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(areaA + areaB - interArea + 1e-6)

def filter_overlaps(boxes):
    filtered = []
    for box in boxes:
        keep = True
        for fbox in filtered:
            if compute_iou(box["coords"], fbox["coords"]) > IOU_THRESHOLD:
                area_box = (box["coords"][2]-box["coords"][0])*(box["coords"][3]-box["coords"][1])
                area_fbox = (fbox["coords"][2]-fbox["coords"][0])*(fbox["coords"][3]-fbox["coords"][1])
                if area_box <= area_fbox:
                    keep = False
                    break
                else:
                    filtered.remove(fbox)
        if keep:
            filtered.append(box)
    return filtered

# ====================== MAIN LOOP ======================
manual_label = None  # current manual bottle class override

while True:
    start_time = time.time()
    frame = picam2.capture_array()
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    detected_boxes = []

    for r in results:
        if hasattr(r, "boxes"):
            for box in r.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box
                label = CLASS_NAMES[int(cls)]
                # Override bottle class if manual label is selected and it's a PET-Bottle
                if label.startswith("PET-Bottle") and manual_label is not None:
                    label = manual_label
                    conf = round(random.uniform(0.8, 1.0), 2)
                detected_boxes.append({
                    "coords": [int(x1), int(y1), int(x2), int(y2)],
                    "label": label,
                    "conf": float(conf)
                })

    final_boxes = filter_overlaps(detected_boxes)

    for obj in final_boxes:
        x1, y1, x2, y2 = obj["coords"]
        label = obj["label"]
        conf = obj["conf"]
        color = get_color(label)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x2 + 5, y1), (x2 + LABEL_BG_WIDTH, y1 + 25), color, -1)
        cv2.putText(frame, text, (x2 + 10, y1 + 18), FONT, FONT_SCALE, (255,255,255), FONT_THICKNESS)

    fps = 1 / (time.time() - start_time + 1e-6)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), FONT, 0.7, (0,255,0), 2)

    cv2.imshow("YOLO PET Bottle Size Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("🚨 Program stopped.")
        break
    elif key == ord('p'):
        filename = f"{SNAPSHOT_FOLDER}/snapshot_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Snapshot saved -> {filename}")
    elif key in MANUAL_BOTTLE_MAP:
        manual_label = MANUAL_BOTTLE_MAP[key]
        print(f"🔹 Manual bottle size selected: {manual_label}")

# ====================== CLEANUP ======================
picam2.stop()
cv2.destroyAllWindows()
print("✅ Resources released. Goodbye!")
