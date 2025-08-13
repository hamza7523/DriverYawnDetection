# app.py
# Yawn detection webcam app tuned to yawn_96.h5 (expects RGB 96x96 input)
# - robust model loader to handle HDF5 InputLayer deserialization mismatches
# - automatic input-shape detection from loaded model or the HDF5 file
# - preprocesses camera ROI to exactly the model input shape/channels

import os
import time
import threading
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import requests
import tensorflow as tf

MODEL_FILE = "yawn_96.h5"
FACE_CASCADE = "haarcascade_frontalface_default.xml"
MOUTH_CASCADE = "haarcascade_mcs_mouth.xml"

CONSECUTIVE_FRAMES_REQUIRED = 6
YAWN_PROB_THRESHOLD = 0.5
ALERT_COUNT = 5
ALERT_WINDOW_SECONDS = 120
FRAME_SCALE_WIDTH = 640  # speed tradeoff

# ---------------------------
# Download helpers for cascades
# ---------------------------
def download_if_missing(fname, urls):
    p = Path(fname)
    if p.exists():
        return
    if isinstance(urls, str):
        urls = [urls]
    for url in urls:
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 200 and r.content:
                p.write_bytes(r.content)
                print(f"Downloaded {fname} from {url}")
                return
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"Could not download {fname}; please add it next to the script.")

FACE_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml",
]
MOUTH_URLS = [
    "https://raw.githubusercontent.com/atduskgreg/opencv-processing/master/lib/cascade-files/haarcascade_mcs_mouth.xml",
    "https://raw.githubusercontent.com/austinjoyal/haar-cascade-files/master/haarcascades/haarcascade_mcs_mouth.xml",
]

download_if_missing(FACE_CASCADE, FACE_URLS)
download_if_missing(MOUTH_CASCADE, MOUTH_URLS)

face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
if face_cascade.empty():
    raise FileNotFoundError(f"Failed to load face cascade from {FACE_CASCADE}")

mouth_cascade = cv2.CascadeClassifier(MOUTH_CASCADE)
if mouth_cascade.empty():
    print("WARNING: mouth cascade failed to load — mouth detection will fallback to heuristic region")

# ---------------------------
# Robust model loader
# ---------------------------
# Robust model loader WITHOUT tensorflow.keras.saving.hdf5_format
def load_keras_model_robust(path):
    """
    Try tf.keras.models.load_model first; on TypeError (InputLayer deserialization),
    attempt to reconstruct model from HDF5 'model_config' attribute and load weights.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file '{path}' not found.")

    # 1) Try the normal loader first (preferred)
    try:
        print("Attempting standard tf.keras.models.load_model() ...")
        model = tf.keras.models.load_model(path, compile=False)
        print("Loaded model using standard loader.")
        return model
    except TypeError as te:
        # Common mismatch: "Unrecognized keyword arguments: ['batch_shape']"
        print("Standard loader failed with TypeError:", te)
        print("Attempting to reconstruct model from HDF5 'model_config' and load weights...")
    except Exception as other_exc:
        # If it's not a TypeError (e.g., file corruption), re-raise with context
        print("Standard loader failed with unexpected exception:", other_exc)
        raise

    # 2) HDF5 fallback: try to read 'model_config' from file and reconstruct model
    try:
        import h5py, json
        with h5py.File(path, "r") as f:
            if "model_config" not in f.attrs:
                raise RuntimeError("HDF5 model has no 'model_config' attribute; cannot reconstruct model.")
            raw = f.attrs["model_config"]
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            cfg = json.loads(raw)

        # cfg expected shape: {'class_name': 'Model', 'config': {...}}
        model_config = cfg.get("config", cfg)
        # Reconstruct model from public API
        model = tf.keras.models.model_from_config(model_config)
        # Load weights (Keras accepts an HDF5 filepath here)
        model.load_weights(path)
        print("Reconstructed model from config and loaded weights successfully.")
        return model

    except Exception as e:
        # Helpful error message with recommended next steps
        print("Reconstruction fallback failed:", e)
        raise RuntimeError(
            "Failed to load model with both tf.keras.models.load_model and the HDF5-reconstruction fallback.\n\n"
            "Likely causes:\n"
            " - The .h5 was saved with newer/older Keras serialization that changes layer config keys.\n"
            " - The model uses custom layers (you must pass custom_objects when rebuilding).\n\n"
            "Options to fix:\n"
            "  A) Re-save model in SavedModel format on the training machine (recommended):\n"
            "       model = tf.keras.models.load_model('yawn_96.h5')\n"
            "       model.save('yawn_saved_model')\n"
            "     Then deploy the SavedModel directory and load via tf.keras.models.load_model('yawn_saved_model').\n\n"
            "  B) Re-save the .h5 with the same TF/Keras version you will use in production.\n\n"
            "  C) If the model uses custom layers, provide a custom_objects dict when reconstructing. Example:\n"
            "       model = tf.keras.models.model_from_config(model_config, custom_objects={'MyLayer': MyLayer})\n\n"
            f"Original reconstruction error: {e}\n"
        ) from e

# If the H5 fails to fully load, this function reads input shape from the HDF5 attributes
def infer_input_shape_from_h5(path):
    try:
        import h5py, json
        with h5py.File(path, "r") as f:
            if "model_config" in f.attrs:
                raw = f.attrs["model_config"]
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8")
                cfg = json.loads(raw)
                # walk to find InputLayer config entries
                inputs = []
                def walk(node):
                    if isinstance(node, dict):
                        if node.get("class_name") == "InputLayer":
                            conf = node.get("config", {})
                            inputs.append(conf)
                        for v in node.values():
                            walk(v)
                    elif isinstance(node, list):
                        for x in node:
                            walk(x)
                walk(cfg.get("config", cfg))
                if inputs:
                    # return the first input batch_shape if present
                    for inp in inputs:
                        bs = inp.get("batch_shape") or inp.get("shape")
                        if bs:
                            # batch_shape might be [None, h, w, c]
                            if isinstance(bs, list) and len(bs) >= 3:
                                # return (h,w,c)
                                return tuple(int(x) for x in bs[-3:])
                    # fallback unknown
    except Exception:
        pass
    return None

# ---------------------------
# Load model and determine expected input shape (H,W,C)
# ---------------------------
print("Loading model (robust)...")
model = load_keras_model_robust(MODEL_FILE)

# Try to read input shape directly from model object
input_shape = None
try:
    ish = getattr(model, "input_shape", None) or getattr(model, "inputs", None)
    if isinstance(ish, tuple):
        # typical: (None, H, W, C)
        if len(ish) == 4:
            input_shape = (int(ish[1]), int(ish[2]), int(ish[3]))
    elif isinstance(ish, list):
        # list of inputs
        first = ish[0]
        if hasattr(first, "shape"):
            s = tuple([int(x) if x is not None else None for x in first.shape.as_list()])
            if len(s) == 4:
                input_shape = (s[1], s[2], s[3])
except Exception:
    input_shape = None

if input_shape is None:
    # fallback: try to read from HDF5 attributes
    print("Falling back to reading input shape from HDF5 file metadata...")
    input_shape = infer_input_shape_from_h5(MODEL_FILE)

if input_shape is None:
    # last resort default
    print("Could not determine input shape automatically; falling back to (96,96,3).")
    input_shape = (96, 96, 3)

H_in, W_in, C_in = input_shape
print(f"Model expects HxW x C = {H_in} x {W_in} x {C_in}")

# ---------------------------
# Preprocess function matched to model input shape
# ---------------------------
def preprocess_roi_for_model(roi_bgr):
    """
    roi_bgr: BGR image from OpenCV (H_roi x W_roi x 3)
    output: np array shaped (1, H_in, W_in, C_in) dtype float32 normalized [0,1]
    """
    # convert to RGB (PIL works in RGB)
    pil = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
    pil = pil.resize((W_in, H_in)).convert("RGB")
    arr = np.array(pil).astype("float32") / 255.0  # shape (H_in, W_in, 3)
    if C_in == 1:
        # convert to luminance
        arr = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
        arr = np.expand_dims(arr, -1)
    elif C_in == 3:
        # keep RGB channels
        pass
    else:
        # unlikely, but attempt to adapt
        if arr.shape[-1] != C_in:
            # truncate or tile channels as best-effort
            if C_in < 3:
                arr = arr[..., :C_in]
            else:
                arr = np.tile(arr, (1,1,C_in//3 + 1))[:,:,:C_in]
    arr = np.expand_dims(arr, 0)
    return arr

# ---------------------------
# Thread-safe counters and helpers
# ---------------------------
state_lock = threading.Lock()
consecutive_yawn_frames = 0
yawn_timestamps = []
yawn_total = 0
yawn_active = False

def prune_old_timestamps():
    cutoff = time.time() - ALERT_WINDOW_SECONDS
    with state_lock:
        while yawn_timestamps and yawn_timestamps[0] < cutoff:
            yawn_timestamps.pop(0)

def count_recent():
    prune_old_timestamps()
    with state_lock:
        return len(yawn_timestamps)

# ---------------------------
# Main webcam loop
# ---------------------------
def main():
    global consecutive_yawn_frames, yawn_total, yawn_active
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera. If running on server, use an IP camera or video file.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # scale for speed
            h, w = frame.shape[:2]
            if w != FRAME_SCALE_WIDTH:
                ratio = FRAME_SCALE_WIDTH / float(w)
                frame = cv2.resize(frame, (FRAME_SCALE_WIDTH, int(h * ratio)))
                h, w = frame.shape[:2]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
            status_text = "No face"

            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x,y,fw,fh = faces[0]
                cv2.rectangle(frame, (x,y), (x+fw, y+fh), (0,255,0), 2)

                lower_y = int(y + fh*0.5)
                face_lower = frame[lower_y:y+fh, x:x+fw]
                if face_lower.size == 0:
                    status_text = "No mouth roi"
                    with state_lock:
                        consecutive_yawn_frames = 0
                        yawn_active = False
                else:
                    gray_lower = cv2.cvtColor(face_lower, cv2.COLOR_BGR2GRAY)
                    mouths = []
                    if not mouth_cascade.empty():
                        mouths = mouth_cascade.detectMultiScale(gray_lower, scaleFactor=1.5, minNeighbors=11, minSize=(30,30))
                    if len(mouths) > 0:
                        mouths = sorted(mouths, key=lambda m: m[2]*m[3], reverse=True)
                        mx,my,mw,mh = mouths[0]
                        mx_abs = x + mx
                        my_abs = lower_y + my
                        mouth_box = (mx_abs, my_abs, mw, mh)
                        cv2.rectangle(frame, (mx_abs, my_abs), (mx_abs+mw, my_abs+mh), (255,255,0), 2)
                    else:
                        mx_abs = x + int(fw*0.15)
                        my_abs = y + int(fh*0.65)
                        mw = int(fw*0.7)
                        mh = int(fh*0.25)
                        mouth_box = (mx_abs, my_abs, mw, mh)
                        cv2.rectangle(frame, (mx_abs, my_abs), (mx_abs+mw, my_abs+mh), (100,100,100), 1)

                    mx, my, mw, mh = mouth_box
                    mx = max(0, mx); my = max(0, my)
                    mx2 = min(frame.shape[1], mx+mw); my2 = min(frame.shape[0], my+mh)
                    roi = frame[my:my2, mx:mx2]
                    if roi.size != 0:
                        inp = preprocess_roi_for_model(roi)
                        try:
                            pred = model.predict(inp, verbose=0)
                            pred = np.asarray(pred)
                            # interpret prediction robustly
                            if pred.ndim == 1:
                                prob = float(pred.ravel()[-1])
                            elif pred.ndim == 2 and pred.shape[1] == 1:
                                prob = float(pred[0,0])
                            elif pred.ndim == 2 and pred.shape[1] >= 2:
                                prob = float(pred[0,1])
                            else:
                                prob = float(np.ravel(pred)[0])
                        except Exception as e:
                            prob = 0.0
                            print("Prediction error:", e)

                        if prob > YAWN_PROB_THRESHOLD:
                            with state_lock:
                                consecutive_yawn_frames += 1
                                if (not yawn_active) and (consecutive_yawn_frames >= CONSECUTIVE_FRAMES_REQUIRED):
                                    yawn_active = True
                                    yawn_timestamps.append(time.time())
                                    yawn_total += 1
                                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Yawn counted! total={yawn_total}")
                            status_text = f"Yawning {prob:.2f}"
                        else:
                            with state_lock:
                                consecutive_yawn_frames = 0
                                yawn_active = False
                            status_text = f"Monitoring {prob:.2f}"
                    else:
                        status_text = "Empty roi"
                        with state_lock:
                            consecutive_yawn_frames = 0
                            yawn_active = False
            else:
                status_text = "No face"
                with state_lock:
                    consecutive_yawn_frames = 0
                    yawn_active = False

            recent = count_recent()
            alert_on = recent >= ALERT_COUNT

            cv2.putText(frame, f"Status: {status_text}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"Yawns last {ALERT_WINDOW_SECONDS//60}m: {recent}", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            with state_lock:
                total_display = yawn_total
            cv2.putText(frame, f"Yawns total: {total_display}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            if alert_on:
                cv2.rectangle(frame, (0,0), (frame.shape[1], 50), (0,0,255), -1)
                cv2.putText(frame, f"ALERT: {recent} yawns in last {ALERT_WINDOW_SECONDS//60} minutes", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Yawn webcam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                with state_lock:
                    consecutive_yawn_frames = 0
                    yawn_timestamps.clear()
                    yawn_total = 0
                    yawn_active = False
                print("Reset counts")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Exited. Final counts:")
        with state_lock:
            print(f"  total yawns: {yawn_total}")
            print(f"  recent yawns (last {ALERT_WINDOW_SECONDS}s): {len(yawn_timestamps)}")

if __name__ == "__main__":
    main()
