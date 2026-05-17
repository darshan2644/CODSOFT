# ============================================================
#   Face Detection & Recognition - OpenCV Haar Cascades
#   CodSoft AI Internship | Task 5
#   Author: Your Name
#
#   Run this once before starting:
#   pip install opencv-python pillow numpy
# ============================================================


import cv2
import numpy as np
import os
import sys
from datetime import datetime


# ─── LOAD CASCADES ──────────────────────────────────────────
# opencv-python ships these XML files built-in, no download needed

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

if face_cascade.empty():
    print("couldn't load face cascade. something's wrong with your opencv install.")
    sys.exit(1)

if eye_cascade.empty():
    print("couldn't load eye cascade. something's wrong with your opencv install.")
    sys.exit(1)


# ─── CORE DETECTION ─────────────────────────────────────────

def run_detection(gray, frame):
    """
    Takes a grayscale frame for detection and a color frame for drawing.
    Returns detected faces and total eye count.
    """
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    total_eyes = 0

    for i, (x, y, w, h) in enumerate(faces):

        # green box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)

        # label just above the box
        cv2.putText(
            frame, f"face {i + 1}",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2
        )

        # now look for eyes inside this face region only
        face_gray  = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(15, 15)
        )

        for (ex, ey, ew, eh) in eyes:
            # blue box around each eye
            cv2.rectangle(
                face_color,
                (ex, ey), (ex + ew, ey + eh),
                (255, 100, 0), 2
            )

        total_eyes += len(eyes)

    return faces, total_eyes


# ─── MODE 1: IMAGE FILE ─────────────────────────────────────

def detect_from_image(filepath):
    filepath = filepath.strip().strip('"').strip("'")

    if not os.path.exists(filepath):
        print(f"\n  can't find that file: {filepath}")
        print("  double-check the path and try again.")
        return

    print(f"\n  loading → {filepath}")
    image = cv2.imread(filepath)

    if image is None:
        print("  opencv couldn't read this file.")
        print("  make sure it's a proper image (jpg, png, etc.)")
        return

    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # run detection and time it
    t_start = datetime.now()
    faces, total_eyes = run_detection(gray, image)
    t_end = (datetime.now() - t_start).total_seconds()

    # results
    print("\n  " + "-" * 38)
    print(f"  faces found    : {len(faces)}")
    print(f"  eyes found     : {total_eyes}")
    print(f"  resolution     : {w_img} x {h_img} px")
    print(f"  time taken     : {t_end:.3f}s")
    print("  " + "-" * 38)

    if len(faces) == 0:
        print("\n  no faces detected.")
        print("  tips: use a well-lit frontal photo, avoid heavy blur.")

    # save the annotated result
    out_dir  = os.path.dirname(os.path.abspath(filepath))
    out_path = os.path.join(out_dir, "detected_output.jpg")
    cv2.imwrite(out_path, image)
    print(f"\n  result saved → {out_path}")

    # show it
    print("  press any key to close the window.")
    cv2.imshow("face detection result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─── MODE 2: LIVE WEBCAM ────────────────────────────────────

def detect_live():
    print("\n  starting webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("  couldn't open webcam.")
        print("  make sure nothing else is using the camera and try again.")
        return

    print("  webcam running. press Q to stop.\n")

    frame_count = 0
    fps         = 0.0
    fps_timer   = datetime.now()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("  lost the webcam feed — stopping.")
            break

        # fps counter
        frame_count += 1
        elapsed = (datetime.now() - fps_timer).total_seconds()
        if elapsed >= 1.0:
            fps         = frame_count / elapsed
            frame_count = 0
            fps_timer   = datetime.now()

        # detection
        gray             = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, eye_count = run_detection(gray, frame)
        face_count       = len(faces)

        # status overlay — top left
        status_color = (0, 220, 0) if face_count > 0 else (0, 120, 255)
        cv2.putText(
            frame,
            f"faces: {face_count}  eyes: {eye_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2
        )

        # fps — top right
        cv2.putText(
            frame,
            f"fps: {fps:.1f}",
            (frame.shape[1] - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2
        )

        # quit hint — bottom left
        cv2.putText(
            frame,
            "press Q to quit",
            (10, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1
        )

        cv2.imshow("live face detection", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n  webcam closed. done!")


# ─── BANNER & MENU ──────────────────────────────────────────

def print_banner():
    print()
    print("=" * 46)
    print("   Face Detection AI — CodSoft Task 5")
    print("   OpenCV Haar Cascade  |  CPU friendly")
    print("=" * 46)
    print("  detects faces and eyes in images or webcam")
    print("=" * 46)

def print_menu():
    print()
    print("  1. detect faces in an image file")
    print("  2. live detection via webcam")
    print("  3. exit")
    print()


# ─── MAIN ───────────────────────────────────────────────────

def main():
    print_banner()

    while True:
        print_menu()
        choice = input("  pick an option (1-3): ").strip()

        if choice == "1":
            path = input("  image file path: ").strip()
            if path:
                detect_from_image(path)
            else:
                print("  no path entered, try again.")

        elif choice == "2":
            detect_live()

        elif choice == "3":
            print("\n  bye! 👋\n")
            break

        else:
            print("  that's not a valid option, pick 1, 2, or 3.")


if __name__ == "__main__":
    main()