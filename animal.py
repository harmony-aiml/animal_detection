#!/usr/bin/env python3
"""Webcam object detection using a pretrained YOLOv8 model."""

import cv2
import time
from ultralytics import YOLO


def main():
    """Run detection from the default webcam."""
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = "output_annotated.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("✅ Webcam activated. Starting detection...")

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{class_name} {conf:.2f}"

            print(
                f"Detected: {class_name} | Confidence: {conf:.2f} | Box: ({x1}, {y1}), ({x2}, {y2})"
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        curr_time = time.time()
        fps_disp = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(
            frame,
            f"FPS: {fps_disp:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        out.write(frame)

        cv2.imshow("YOLOv8 Detection - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"🛑 Detection stopped. Output saved to `{output_path}`")


if __name__ == "__main__":
    main()
