from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

# ====================== USER SETTINGS ======================
MODEL_PATH = "/home/yolo/psm_bottle_yolov8s_model2.pt"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.6
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 2
LABEL_BG_WIDTH = 160          # Adjust if needed
SNAPSHOT_FOLDER = "snapshots"

# ====================== INITIAL SETUP ======================
print("🔄 Loading YOLO model...")
model = YOLO(MODEL_PATH)
class_names = model.names
print("✅ Model loaded successfully.")

# Create snapshot folder if not exist
if not os.path.exists(SNAPSHOT_FOLDER):
    os.makedirs(SNAPSHOT_FOLDER)
    print(f"📁 Snapshot folder created: {SNAPSHOT_FOLDER}")

# Global colour dictionary for classes
COLOR_MAP = {}

def get_color(label):
    """Return fixed color for each class."""
    if label not in COLOR_MAP:
        COLOR_MAP[label] = list(np.random.randint(0, 255, 3))
    return COLOR_MAP[label]

# ====================== IOU FUNCTION ======================
def compute_iou(boxA, boxB):
    """Calculate Intersection over Union."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = ((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = ((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(areaA + areaB - interArea + 1e-6)
    return iou

# ====================== DUPLICATE FILTER ======================
def filter_overlaps(boxes):
    """Remove overlapping detections using IoU threshold."""
    filtered = []

    for box in boxes:
        keep = True
        for fbox in filtered:
            iou = compute_iou(box["coords"], fbox["coords"])
            if iou > IOU_THRESHOLD:
                # keep larger bbox (area comparison)
                area_box = (box["coords"][2] - box["coords"][0]) * (box["coords"][3] - box["coords"][1])
                area_fbox = (fbox["coords"][2] - fbox["coords"][0]) * (fbox["coords"][3] - fbox["coords"][1])

                if area_box <= area_fbox:
                    keep = False
                    break
                else:
                    filtered.remove(fbox)
        if keep:
            filtered.append(box)

    return filtered

# ====================== CAMERA SETUP ======================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: No camera detected. Check connection.")
    exit()

print("\n🎥 Camera ready — press **q** to quit, **p** to snapshot.\n")

# ====================== MAIN LOOP ======================
while True:
    start_time = time.time()
    ret, frame = cap.read()

    if not ret:
        print("⚠ Frame capture failed.")
        break

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    detected_boxes = []

    # Extract detections
    for r in results:
        if hasattr(r, "boxes"):
            for box in r.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box
                label = class_names[int(cls)]
                detected_boxes.append({
                    "coords": [int(x1), int(y1), int(x2), int(y2)],
                    "label": label,
                    "conf": float(conf)
                })

    # Remove duplicates using IoU logic
    final_boxes = filter_overlaps(detected_boxes)

    # Draw final filtered boxes
    for obj in final_boxes:
        x1, y1, x2, y2 = obj["coords"]
        label = obj["label"]
        conf = obj["conf"]
        color = get_color(label)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x2 + 5, y1), (x2 + LABEL_BG_WIDTH, y1 + 25), color, -1)
        cv2.putText(frame, text, (x2 + 10, y1 + 18), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

    # FPS Display
    fps = 1 / (time.time() - start_time + 1e-6)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), FONT, 0.7, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLO Real-Time Detection (with IoU + Snapshot)", frame)

    key = cv2.waitKey(1) & 0xFF

    # Quit program
    if key == ord('q'):
        print("\n🛑 Program stopped.")
        break

    # Snapshot
    elif key == ord('p'):
        filename = f"{SNAPSHOT_FOLDER}/snapshot_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Snapshot saved -> {filename}")

# ====================== CLEANUP ======================
cap.release()
cv2.destroyAllWindows()
print("✔ All resources released. Goodbye!")
