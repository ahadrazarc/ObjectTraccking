import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
VIDEO_PATH = "YTDown_YouTube_4K-Road-traffic-video-for-object-detecti_Media_MNn9qKG2UFI_003_480p.mp4"
OUTPUT_PATH = "output_counted.mp4"
MODEL_PATH = "yolo26m.pt"  # Ensure this is a valid weights file, like yolov8n.pt

# Map the COCO IDs so YOLO knows what they are
CLASS_ID_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
# We track all of them so YOLO actually sees them
TRACK_CLASSES = list(CLASS_ID_MAP.keys())

# Your target is ONLY car. Everything else (motorcycle, bus, truck) becomes "Other"
TARGET_CLASSES = {"car"}

# --- NEW COLOR SCHEME (OpenCV uses BGR, not RGB) ---
CLASS_COLORS = {
    "car": (190, 190, 40),  # Cyan/Teal
    "Other": (140, 140, 140),  # Gray
}

BOX_THICK = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --- Classes ---
class VehicleCounter:
    def __init__(self, zone_polygon):
        self.zone_polygon = np.array(zone_polygon, dtype=np.int32)
        self.counts = {c: 0 for c in TARGET_CLASSES}
        self.counts["Other"] = 0
        self.counted_ids = set()

        # Dictionary to remember where each vehicle was in the previous frame
        self.track_history = {}
        # Dictionary to track which polygon lines the vehicle has crossed
        self.track_progress = {}

        # Get the Y-coordinates for the top and bottom lines of the polygon
        self.y_top = min(p[1] for p in self.zone_polygon)
        self.y_bottom = max(p[1] for p in self.zone_polygon)

        # Get the X-coordinates for horizontal boundaries
        self.x_min = min(p[0] for p in self.zone_polygon)
        self.x_max = max(p[0] for p in self.zone_polygon)

    def _bucket(self, class_name):
        return class_name if class_name in TARGET_CLASSES else "Other"

    def update(self, detections):
        for track_id, class_name, x1, y1, x2, y2 in detections:
            # Calculate current center of the vehicle
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # If this is the first time we see this vehicle, save its position and init progress
            if track_id not in self.track_history:
                self.track_history[track_id] = (cx, cy)
                self.track_progress[track_id] = set()
                continue

            # Get where the vehicle was in the previous frame
            prev_cx, prev_cy = self.track_history[track_id]

            if track_id not in self.counted_ids:
                within_width = self.x_min <= cx <= self.x_max

                # 1. Did it cross the top line of the polygon?
                crossed_top = (prev_cy <= self.y_top < cy) or (prev_cy >= self.y_top > cy)
                if crossed_top and within_width:
                    self.track_progress[track_id].add('top')

                # 2. Did it cross the bottom line of the polygon?
                crossed_bottom = (prev_cy <= self.y_bottom < cy) or (prev_cy >= self.y_bottom > cy)
                if crossed_bottom and within_width:
                    self.track_progress[track_id].add('bottom')

                # VALIDATION PASS: If it has successfully crossed BOTH lines
                if 'top' in self.track_progress[track_id] and 'bottom' in self.track_progress[track_id]:
                    bucket = self._bucket(class_name)
                    self.counts[bucket] += 1
                    self.counted_ids.add(track_id)

            # Update history for the next frame
            self.track_history[track_id] = (cx, cy)

    def draw_zone(self, frame):
        # Create a transparent "glass" region
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.zone_polygon], (255, 255, 255))

        # 15% opacity for the fill to make it highly transparent
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Draw a thin, clean outer border
        cv2.polylines(frame, [self.zone_polygon], True, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def draw_detections(self, frame, detections):
        for track_id, class_name, x1, y1, x2, y2 in detections:
            bucket = self._bucket(class_name)
            color = CLASS_COLORS[bucket]

            # Simple labeling: If it's not a car, just label it "Other"
            label = class_name.capitalize() if bucket != "Other" else "Other"

            # Draw sleek bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, BOX_THICK)

            # Draw smaller, tighter label background
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.4, 1)
            cv2.rectangle(frame, (int(x1), int(y1) - th - 6),
                          (int(x1) + tw + 6, int(y1)), color, -1)

            # Draw label text
            cv2.putText(frame, label, (int(x1) + 3, int(y1) - 3),
                        FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw a tiny dot at the exact center so you can see what is being tracked
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        return frame

    def draw_count_box(self, frame):
        h, w = frame.shape[:2]

        # Box dimensions
        row_h, pad, width = 30, 15, 200
        order = ["car", "Other"]
        height = pad * 2 + row_h * len(order)

        # Calculate Top-Center position dynamically based on frame width
        x0 = int((w - width) / 2)
        y0 = 40  # 40 pixels from the top

        # Draw dark translucent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)  # 75% dark overlay

        # Draw text and color indicators
        for i, cls in enumerate(order):
            cy = y0 + pad + i * row_h + int(row_h / 2)
            color = CLASS_COLORS[cls]

            # Draw small color square
            cv2.rectangle(frame, (x0 + pad, cy - 6), (x0 + pad + 12, cy + 6), color, -1)

            # Draw Class Name
            cv2.putText(frame, cls.capitalize(), (x0 + pad + 25, cy + 5),
                        FONT, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

            # Draw Count (Right-Aligned)
            count_str = str(self.counts[cls])
            (cw, ch), _ = cv2.getTextSize(count_str, FONT, 0.6, 2)
            cv2.putText(frame, count_str, (x0 + width - pad - cw, cy + 5),
                        FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def render(self, frame, detections):
        self.update(detections)
        frame = self.draw_zone(frame)
        frame = self.draw_detections(frame, detections)
        frame = self.draw_count_box(frame)
        return frame


# --- Main Execution ---
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"Could not open {VIDEO_PATH}"

fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Expanded Width: Starts at 5% on the left and reaches 95% on the right
zone_points = [(int(w * 0.05), int(h * 0.55)), (int(w * 0.95), int(h * 0.55)),
               (int(w * 0.95), int(h * 0.75)), (int(w * 0.05), int(h * 0.75))]

counter = VehicleCounter(zone_points)

print("Starting video processing... Press 'q' in the video window to quit early.")

# Make the OpenCV window responsive/resizable
cv2.namedWindow("Live Vehicle Tracking", cv2.WINDOW_NORMAL)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model.track(frame, classes=TRACK_CLASSES, persist=True, verbose=False)[0]

    detections = []
    if results.boxes is not None and results.boxes.id is not None:
        ids = results.boxes.id.cpu().numpy().astype(int)
        clss = results.boxes.cls.cpu().numpy().astype(int)
        xyxy = results.boxes.xyxy.cpu().numpy()
        for tid, cid, box in zip(ids, clss, xyxy):
            class_name = CLASS_ID_MAP.get(cid, "Other")
            x1, y1, x2, y2 = box
            detections.append((int(tid), class_name, x1, y1, x2, y2))

    # Draw overlays
    frame = counter.render(frame, detections)

    # Save frame to output video
    writer.write(frame)

    # Show live feed on desktop
    cv2.imshow("Live Vehicle Tracking", frame)

    # Check for 'q' key press to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Live preview interrupted by user.")
        break

# --- Cleanup ---
cap.release()
writer.release()
cv2.destroyAllWindows()

print("Final counts:", counter.counts)
print("Saved:", OUTPUT_PATH)