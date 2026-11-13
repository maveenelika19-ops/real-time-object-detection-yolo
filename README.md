Real-Time Object Detection and Tracking Using YOLO11

Project Overview

An AI-powered computer vision application developed using Python, OpenCV, Ultralytics YOLO11, ByteTrack and Streamlit.

The project detects objects in images and tracks objects across video frames through an interactive web dashboard.

Features

- Image upload and object detection
- Object classification with bounding boxes
- Confidence threshold adjustment
- Object category filtering
- Object counting
- Detection details table with coordinates and CSV export
- Annotated detection screenshot downloads
- PDF detection reports with summary, annotated image, and result table
- Image zoom and pan controls for uploaded photos
- Light and dark dashboard themes
- YOLO11 model choice: Nano for speed, Small for balance, Medium for accuracy
- Vehicle detection for bicycles, cars, motorcycles, buses and trucks
- Face detection and optional facial-expression estimates
- Live 3D face mesh and head-pose direction estimation
- Landmark-based facial symmetry scan for front-facing photos
- Live webcam detection, zoom and pan controls, and temporary ByteTrack IDs for people and vehicles
- Live alerts, sound alerts, detection zones, line-crossing counts, FPS status, and optional face blur
- Live people and vehicle count chart in webcam detection
- Live event log with time, object type, tracking ID, and confidence
- CSV export for live event logs
- Alert snapshot gallery with PNG downloads
- Optional email alerts with annotated snapshot attachments
- Detection history with CSV download
- Detection History filtering by object type
- Video upload and object tracking
- Video tracking summary with unique tracking IDs and category totals
- Annotated screenshot download from processed tracking videos
- Tracking IDs using ByteTrack
- Downloadable processed videos
- Interactive Streamlit dashboard

Technologies Used

- Python
- OpenCV
- YOLO11
- PyTorch
- ByteTrack
- Streamlit
- NumPy
- Pillow
- ImageIO-FFmpeg

Installation

Clone the repository and open the project directory.

Create a virtual environment:

python -m venv .venv

Activate it on Windows Command Prompt:

.venv\Scripts\activate

Install dependencies:

python -m pip install -r requirements.txt

The first use of emotion detection downloads the DeepFace emotion model. Face
locations still work if DeepFace is not installed.

To enable email alerts, copy `.env.example` to `.env` and enter your SMTP
host, port, sender account, app password, and recipient email. The `.env` file
is excluded from version control.

Live webcam detection requires camera permission in the browser. It detects
people and vehicle types but does not identify a person's identity from a
reference photo.

The first use of 3D face mesh downloads the MediaPipe Face Landmarker model.

Run the Dashboard

python -m streamlit run app.py

Open the local URL displayed in the terminal.

How It Works

1. The user uploads an image or video.
2. YOLO11 performs object detection.
3. Users can adjust the detection confidence threshold.
4. ByteTrack associates detections across consecutive video frames.
5. The dashboard displays annotated results.
6. Processed videos can be downloaded.

Future Improvements

- Live webcam integration into Streamlit
- Detection analytics and reporting
- Database integration

Project Purpose

This project demonstrates practical skills in computer vision, deep learning inference, object detection, multi-object tracking and Python application development.
