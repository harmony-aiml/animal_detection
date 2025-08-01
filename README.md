# Animal Detection

This project provides a simple script to run real-time object detection using a webcam and the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) model. Detected objects are drawn on the video frames and saved to an output video file.

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the script:

```bash
python animal.py
```

Press `q` in the video window to stop the detection. The annotated video will be saved as `output_annotated.mp4` in the current directory.
