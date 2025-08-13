# webcam_yawn_opencv_fixed.py
import os
import time
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
import requests
import cv2
from PIL import Image
import tensorflow as tf

# config
MODEL_FILE = "yawn_96.h5"
FACE_CASCADE = "haarcascade_frontalface_default.xml"
MOUTH_CASCADE = "haarcascade_mcs_mouth.xml"
CONSECUTIVE_FRAMES_REQUIRED = 6
YAWN_PROB_THRESHOLD = 0.5
ALERT_COUNT = 5
ALERT_WINDOW_SECONDS = 120
FRAME_SCALE_WIDTH = 640  # resize camera frame width for speed

# robust downloader for cascade files
def download_if_missing(fname, urls):
    """
    Ensure local file 'fname' exists. `urls` can be a single URL or a list of URLs.
    Tries each url in turn; on success writes the content to fname.
    """
    p = Path(fname)
    if p.exists():
        return
    if isinstance(urls, str):
        urls = [urls]

    for url in urls:
        try:
            print("Trying to download cascade from:", url)
            r = requests.get(url, timeout=12)
            # accept any 200 OK with non-empty content (XML can have various content-types)
            if r.status_code == 200 and r.content:
                p.write_bytes(r.content)
                print(f"Downloaded '{fname}' from {url}")
                return
            else:
                print("Download attempt returned status", r.status_code, "for", url)
        except Exception as e:
            print("Error downloading from", url, ":", e)
        time.sleep(0.4)

    raise RuntimeError(
        f"Could not download {fname} from known URLs. "
        "Please download it manually and place it next to your script."
    )

# reliable lists of fallback URLs for cascades (raw GitHub links)
FACE_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml",
]

MOUTH_URLS = [
    "https://raw.githubusercontent.com/atduskgreg/opencv-processing/master/lib/cascade-files/haarcascade_mcs_mouth.xml",
    "https://raw.githubusercontent.com/austinjoyal/haar-cascade-files/master/haarcascades/haarcascade_mcs_mouth.xml",
]

# ensure cascades exist locally (will download if missing)
download_if_missing(FACE_CASCADE, FACE_URLS)
download_if_missing(MOUTH_CASCADE, MOUTH_URLS)

# load cascades and validate
face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
if face_cascade.empty():
    raise FileNotFoundError(f"Failed to load face cascade from '{FACE_CASCADE}'. Place the file next to the script.")

mouth_cascade = cv2.CascadeClassifier(MOUTH_CASCADE)
if mouth_cascade.empty():
    print(f"WARNING: Failed to load mouth cascade from '{MOUTH_CASCADE}'. Mouth detection will fallback to heuristic region.")
    # keep going; code will handle mouth_cascade being effectively unavailable

# model presence
if not Path(MODEL_FILE).exists():
    raise FileNotFoundError(f"{MODEL_FILE} not found. Put your model file in the same folder")

print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_FILE, compile=False)
print("Model loaded. input shape:", model.input_shape)

# helper to prepare mouth ROI for model
def preprocess_roi_for_model(roi_bgr, model):
    pil = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
    # default target
    H = W = 96
    C = 1
    input_shape = getattr(model, "input_shape", None)
    if input_shape:
        # input_shape can be (None, h, w, c) or (None, h, w)
        if len(input_shape) == 4:
            _, h, w, c = input_shape
            H = int(h) if h else H
            W = int(w) if w else W
            C = int(c) if c else C
        elif len(input_shape) == 3:
            _, h, w = input_shape
            H = int(h) if h else H
            W = int(w) if w else W
            C = 1
    pil = pil.resize((W, H)).convert("RGB")
    arr = np.array(pil).astype("float32") / 255.0
    if C == 1:
        # convert to grayscale with luminance weights
        arr = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
        arr = np.expand_dims(arr, -1)
    arr = np.expand_dims(arr, 0)  # batch dim
    return arr

# thread safe state
state_lock = threading.Lock()
consecutive_yawn_frames = 0
yawn_timestamps = []
yawn_total = 0

def prune_old_timestamps():
    cutoff = time.time() - ALERT_WINDOW_SECONDS
    with state_lock:
        while yawn_timestamps and yawn_timestamps[0] < cutoff:
            yawn_timestamps.pop(0)

def count_recent():
    prune_old_timestamps()
    with state_lock:
        return len(yawn_timestamps)

def main():
    global consecutive_yawn_frames, yawn_total
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera (index 0). If you have multiple cameras, try changing the index.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # resize for speed
        h, w = frame.shape[:2]
        if w != FRAME_SCALE_WIDTH:
            ratio = FRAME_SCALE_WIDTH / float(w)
            frame = cv2.resize(frame, (FRAME_SCALE_WIDTH, int(h * ratio)))
            h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        status_text = "No face"

        if len(faces) > 0:
            # pick largest face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, fw, fh = faces[0]
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

            # search for mouth in the lower half of face
            lower_y = int(y + fh * 0.5)
            face_lower = frame[lower_y:y + fh, x:x + fw]
            if face_lower.size == 0:
                status_text = "No mouth roi"
            else:
                gray_lower = cv2.cvtColor(face_lower, cv2.COLOR_BGR2GRAY)
                mouths = []
                # only try mouth cascade detection if it loaded successfully
                if not mouth_cascade.empty():
                    mouths = mouth_cascade.detectMultiScale(gray_lower, scaleFactor=1.5, minNeighbors=11, minSize=(30, 30))
                mouth_box = None
                if len(mouths) > 0:
                    mouths = sorted(mouths, key=lambda m: m[2] * m[3], reverse=True)
                    mx, my, mw, mh = mouths[0]
                    mx_abs = x + mx
                    my_abs = lower_y + my
                    mouth_box = (mx_abs, my_abs, mw, mh)
                    cv2.rectangle(frame, (mx_abs, my_abs), (mx_abs + mw, my_abs + mh), (255, 255, 0), 2)
                else:
                    # fallback: approximate mouth region as lower third of face
                    mx_abs = x + int(fw * 0.15)
                    my_abs = y + int(fh * 0.65)
                    mw = int(fw * 0.7)
                    mh = int(fh * 0.25)
                    mouth_box = (mx_abs, my_abs, mw, mh)
                    cv2.rectangle(frame, (mx_abs, my_abs), (mx_abs + mw, my_abs + mh), (100, 100, 100), 1)

                # extract mouth roi and classify
                mx, my, mw, mh = mouth_box
                # clamp coordinates
                mx = max(0, mx); my = max(0, my)
                mx2 = min(frame.shape[1], mx + mw); my2 = min(frame.shape[0], my + mh)
                roi = frame[my:my2, mx:mx2]
                if roi.size != 0:
                    inp = preprocess_roi_for_model(roi, model)
                    try:
                        pred = model.predict(inp, verbose=0)
                        if pred is None:
                            raise RuntimeError("model returned None")
                        # handle several output shapes:
                        # - binary sigmoid: shape (1,1) -> use pred[0,0]
                        # - two-class softmax: shape (1,2) -> use prob of class 1
                        # - sometimes pred can be a vector shape (2,) if batch dim missing
                        pred = np.asarray(pred)
                        if pred.ndim == 1:
                            prob = float(pred.ravel()[-1])  # use last element (best-effort)
                        elif pred.ndim == 2 and pred.shape[1] == 1:
                            prob = float(pred[0, 0])
                        elif pred.ndim == 2 and pred.shape[1] >= 2:
                            prob = float(pred[0, 1])
                        else:
                            prob = float(np.ravel(pred)[0])
                    except Exception as e:
                        prob = 0.0
                        print("Prediction error:", e)

                    if prob > YAWN_PROB_THRESHOLD:
                        with state_lock:
                            consecutive_yawn_frames += 1
                        status_text = f"Yawning {prob:.2f}"
                    else:
                        with state_lock:
                            if consecutive_yawn_frames >= CONSECUTIVE_FRAMES_REQUIRED:
                                yawn_timestamps.append(time.time())
                                yawn_total += 1
                            consecutive_yawn_frames = 0
                        status_text = f"Monitoring {prob:.2f}"
                else:
                    status_text = "Empty roi"
        else:
            status_text = "No face"
            with state_lock:
                consecutive_yawn_frames = 0

        recent = count_recent()
        alert_on = recent >= ALERT_COUNT

        # overlay
        cv2.putText(frame, f"Status: {status_text}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Yawns last {ALERT_WINDOW_SECONDS//60}m: {recent}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Yawns total: {yawn_total}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if alert_on:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(frame, f"ALERT: {recent} yawns in last {ALERT_WINDOW_SECONDS//60} minutes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Yawn webcam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            with state_lock:
                consecutive_yawn_frames = 0
                yawn_timestamps.clear()
                yawn_total = 0
                print("Reset counts")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
