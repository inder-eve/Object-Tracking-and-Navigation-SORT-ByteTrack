from ultralytics import YOLO
import cv2
import os
from bytetrack import ByteTrackTracker   # make sure bytetrack.py is in same folder

# ===================== CONFIG ===================== #

# 1) Your trained YOLO model
MODEL_PATH = "tank_best.pt"   # change to "best.pt" if your file has that name

# 2) All test videos you want to run
VIDEO_LIST = [
    "Video15.mp4",
    "Video16.mp4",
    "Video17.mp4",
    "Video18.mp4",
    "Video19.mp4",
    "Video20.mp4",
    "Video21.mp4",
    
]

# 3) Detection + tracking settings
LOW_CONF = 0.10      # ignore detections below this confidence
HIGH_CONF = 0.50     # ByteTrack treats detections above this as "high confidence"
IMG_SIZE = 1280      # image size for YOLO inference (better small-object detection, slower)
SAVE_OUTPUT = True   # save tracked video to file

# ================================================== #


def process_video(video_path, model):
    """
    Process a single video: run YOLO + ByteTrack and save a tracked output video.
    """

    print(f"\n[INFO] Processing video: {video_path}")

    # ---- initialize ByteTrack tracker for this video ----
    tracker = ByteTrackTracker(
        high_thresh=HIGH_CONF,
        low_thresh=LOW_CONF,
        iou_thresh=0.3,
        max_age=30,
        min_hits=3
    )

    # ---- open video ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] FPS: {fps}, Size: {width}x{height}")

    # ---- set up output writer ----
    if SAVE_OUTPUT:
        base_name = os.path.splitext(os.path.basename(video_path))[0]  # e.g. "test1"
        output_path = f"tracked_{base_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps if fps > 0 else 25,
            (width, height)
        )
        print(f"[INFO] Saving tracked video to: {output_path}")
    else:
        out = None

    # ---- main frame loop ----
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video.")
            break

        # 1) Run YOLO detection on current frame
        results = model(frame, imgsz=1280, verbose=False)

        detections = []  # list of [x1, y1, x2, y2, conf]

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < LOW_CONF:
                    continue

                cls_id = int(box.cls[0])

                # NOTE:
                # If your model has multiple classes later, you can filter here:
                # if model.names[cls_id] != "tank":
                #     continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Add detection with confidence for ByteTrack
                detections.append([x1, y1, x2, y2, conf])

        # 2) Update ByteTrack tracker with current frame detections
        tracked = tracker.update(detections)

        # 3) Draw tracked boxes + IDs
        for (bbox, track_id) in tracked:
            x1, y1, x2, y2 = bbox.flatten().tolist()

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            label = f"ID {track_id}"
            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # 4) Optional overlay text
        cv2.putText(
            frame,
            "YOLO + ByteTrack",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # 5) Write frame to output video file
        if SAVE_OUTPUT and out is not None:
            out.write(frame)

        # 6) Show frame live
        cv2.imshow("Tank Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Interrupted by user.")
            break

    # ---- cleanup ----
    cap.release()
    cv2.destroyAllWindows()
    if SAVE_OUTPUT and out is not None:
        out.release()
        print("[INFO] Tracked video saved.")


def main():
    # Load YOLO model once
    print(f"[INFO] Loading YOLO model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model classes:", model.names)

    # Loop over each video in the list
    for video_path in VIDEO_LIST:
        if not os.path.exists(video_path):
            print(f"[WARN] File not found, skipping: {video_path}")
            continue
        process_video(video_path, model)

    print("\n[INFO] All videos processed.")


if __name__ == "__main__":
    main()


