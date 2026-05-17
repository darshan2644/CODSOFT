# Task 5 - Face Detection

Detects faces and eyes in images or through a live webcam feed using OpenCV's built-in **Haar Cascade classifiers**. Runs in two modes: analyze an image file, or start a real-time webcam detection session. No model training needed — the classifiers come pre-installed with OpenCV.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `opencv-python` | Core library for face/eye detection using Haar Cascades and webcam access |
| `numpy` | Handles image data as arrays for processing |
| `Pillow` | Opens and converts image files before passing to OpenCV |

---

## How to Run

**Step 1 — Install dependencies:**

```bash
pip install opencv-python numpy Pillow
```

**Step 2 — Run the script:**

```bash
python face_detection.py
```

---

## Sample Output

```
============================================================
        Face Detection using OpenCV Haar Cascades
============================================================

Choose mode:
  1. Detect faces in an image file
  2. Live webcam face detection
  3. Exit

Enter choice (1/2/3): 1

Enter image path: group_photo.jpg

Analyzing image...

------------------------------------------------------------
Results for: group_photo.jpg
------------------------------------------------------------
  Resolution   : 1280 x 720 pixels
  Faces found  : 3
  Eyes found   : 6
  Time taken   : 0.042 seconds
------------------------------------------------------------

Face 1: located at x=112, y=98, width=134, height=134
Face 2: located at x=390, y=85, width=128, height=128
Face 3: located at x=670, y=102, width=141, height=141

Saved result image as: group_photo_detected.jpg

------------------------------------------------------------

Choose mode:
  1. Detect faces in an image file
  2. Live webcam face detection
  3. Exit

Enter choice (1/2/3): 2

Starting webcam... Press 'q' to quit.
Webcam feed opened. Detecting faces in real-time.
[Live window opens showing webcam with rectangles around detected faces]
Webcam closed.

------------------------------------------------------------

Choose mode:
  1. Detect faces in an image file
  2. Live webcam face detection
  3. Exit

Enter choice (1/2/3): 3

Exiting. Goodbye!
============================================================
```
