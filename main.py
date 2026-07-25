"""
Build a labeled 2x2 grid collage from 4 tracker output videos using only
OpenCV (cv2) - no ffmpeg install, no PATH setup needed.

Run:
    python make_collage_cv2.py
"""

import cv2
import numpy as np

# --- EDIT THESE -----------------------------------------------------------
VIDEOS = [
    "contents/bytetrack.mp4",
    "contents/botsort.mp4",
    "contents/deepocsort.mp4",
    "contents/tracktrack.mp4",
]
LABELS = ["ByteTrack", "BoT-SORT", "Deep OC-SORT", "TrackTrack"]
OUTPUT_PATH = "tracker_comparison_collage.mp4"
TILE_WIDTH = 640    # each quadrant's width; final video = TILE_WIDTH*2 x TILE_HEIGHT*2
TILE_HEIGHT = 360
# ---------------------------------------------------------------------------

assert len(VIDEOS) == 4 and len(LABELS) == 4, "Need exactly 4 videos and 4 labels"

caps = [cv2.VideoCapture(v) for v in VIDEOS]
for v, cap in zip(VIDEOS, caps):
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {v}")

fps = caps[0].get(cv2.CAP_PROP_FPS) or 25  # fallback if a video reports 0 fps
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (TILE_WIDTH * 2, TILE_HEIGHT * 2))

frame_count = 0
while True:
    frames = []
    all_ok = True
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            all_ok = False
            break
        frames.append(cv2.resize(frame, (TILE_WIDTH, TILE_HEIGHT)))

    if not all_ok:
        break

    for frame, label in zip(frames, LABELS):
        cv2.rectangle(frame, (0, 0), (10 + len(label) * 14, 34), (0, 0, 0), -1)
        cv2.putText(frame, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

    top = np.hstack([frames[0], frames[1]])
    bottom = np.hstack([frames[2], frames[3]])
    grid = np.vstack([top, bottom])
    out.write(grid)
    frame_count += 1

    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames...")

for cap in caps:
    cap.release()
out.release()

print(f"\nDone. {frame_count} frames written -> {OUTPUT_PATH}")