# server.py
import os
import time
import json
import base64
import threading
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
import cv2
import requests
import h5py

from flask import Flask, render_template, request, jsonify

# --- Config (keep in sync with your app.py) ---
MODEL_FILE = "yawn_96.h5"
FACE_CASCADE = "haarcascade_frontalface_default.xml"
MOUTH_CASCADE = "haarcascade_mcs_mouth.xml"

CONSECUTIVE_FRAMES_REQUIRED = 6
YAWN_PROB_THRESHOLD = 0.5
ALERT_COUNT = 5
ALERT_WINDOW_SECONDS = 120
FRAME_SCALE_WIDTH = 640

# Cascade download sources (same as your script)
FACE_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml",
]
MOUTH_URLS = [
    "https://raw.githubusercontent.com/atduskgreg/opencv-processing/master/lib/cascade-files/haarcascade_mcs_mouth.xml",
    "https://raw.githubusercontent.com/austinjoyal/haar-cascade-files/master/haarcascades/haarcascade_mcs_mouth.xml",
]

# ---------------------------
# Utility: download if missing
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

download_if_missing(FACE_CASCADE, FACE_URLS)
download_if_missing(MOUTH_CASCADE, MOUTH_URLS)

face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
if face_cascade.empty():
    raise FileNotFoundError(f"Failed to load face cascade from {FACE_CASCADE}")

mouth_cascade = cv2.CascadeClassifier(MOUTH_CASCADE)
if mouth_cascade.empty():
    print("WARNING: mouth cascade failed to load — mouth detection will fallback to heuristic region")

# ---------------------------
# Robust model loader (reconstructed from your app)
# ---------------------------
import tensorflow as tf

def infer_input_shape_from_h5(path):
    try:
        with h5py.File(path, "r") as f:
            if "model_config" in f.attrs:
                raw = f.attrs["model_config"]
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8")
                cfg = json.loads(raw)
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
                    for inp in inputs:
                        bs = inp.get("batch_shape") or inp.get("shape")
                        if bs and isinstance(bs, list) and len(bs) >= 3:
                            return tuple(int(x) for x in bs[-3:])
    except Exception:
        pass
    return None

def load_keras_model_robust(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file '{path}' not found.")
    try:
        print("Attempting standard tf.keras.models.load_model() ...")
        m = tf.keras.models.load_model(path, compile=False)
        print("Loaded model using standard loader.")
        return m
    except TypeError as te:
        print("Standard loader failed with TypeError:", te)
    except Exception as other_exc:
        print("Standard loader failed with unexpected exception:", other_exc)
        raise

    # HDF5 fallback
    try:
        with h5py.File(path, "r") as f:
            if "model_config" not in f.attrs:
                raise RuntimeError("HDF5 model has no 'model_config' attribute; cannot reconstruct model.")
            raw = f.attrs["model_config"]
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            cfg = json.loads(raw)
        model_config = cfg.get("config", cfg)
        model = tf.keras.models.model_from_config(model_config)
        model.load_weights(path)
        print("Reconstructed model from config and loaded weights successfully.")
        return model
    except Exception as e:
        raise RuntimeError("Failed to load model (tried both load_model and HDF5 reconstruction). "
                           "See original error: " + str(e)) from e

# Load model and determine input shape
print("Loading model (robust)...")
model = load_keras_model_robust(MODEL_FILE)

input_shape = None
try:
    ish = getattr(model, "input_shape", None) or getattr(model, "inputs", None)
    if isinstance(ish, tuple):
        if len(ish) == 4:
            input_shape = (int(ish[1]), int(ish[2]), int(ish[3]))
    elif isinstance(ish, list):
        first = ish[0]
        if hasattr(first, "shape"):
            s = tuple([int(x) if x is not None else None for x in first.shape.as_list()])
            if len(s) == 4:
                input_shape = (s[1], s[2], s[3])
except Exception:
    input_shape = None

if input_shape is None:
    print("Falling back to reading input shape from HDF5 file metadata...")
    input_shape = infer_input_shape_from_h5(MODEL_FILE)

if input_shape is None:
    print("Could not determine input shape automatically; falling back to (96,96,3).")
    input_shape = (96, 96, 3)

H_in, W_in, C_in = input_shape
print(f"Model expects HxW x C = {H_in} x {W_in} x {C_in}")

# ---------------------------
# Preprocess function
# ---------------------------
def preprocess_roi_for_model(roi_bgr):
    pil = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
    pil = pil.resize((W_in, H_in)).convert("RGB")
    arr = np.array(pil).astype("float32") / 255.0
    if C_in == 1:
        arr = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
        arr = np.expand_dims(arr, -1)
    elif C_in == 3:
        pass
    else:
        if arr.shape[-1] != C_in:
            if C_in < 3:
                arr = arr[..., :C_in]
            else:
                arr = np.tile(arr, (1,1,C_in//3 + 1))[:,:,:C_in]
    arr = np.expand_dims(arr, 0)
    return arr

# ---------------------------
# Server state & helpers
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
# Image utilities
# ---------------------------
def dataurl_from_bgr_image(bgr_img):
    # encode to PNG and return data URL
    ok, buf = cv2.imencode('.png', bgr_img)
    if not ok:
        raise RuntimeError("Failed to encode image")
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return f"data:image/png;base64,{b64}"

def bgr_from_dataurl(data_url):
    header, b64 = data_url.split(',', 1)
    img_bytes = base64.b64decode(b64)
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return bgr

# ---------------------------
# Frame processing (core: face->mouth->predict->annotate)
# ---------------------------
def process_frame_and_annotate(frame_bgr):
    global consecutive_yawn_frames, yawn_total, yawn_active

    h, w = frame_bgr.shape[:2]
    if w != FRAME_SCALE_WIDTH:
        ratio = FRAME_SCALE_WIDTH / float(w)
        frame_bgr = cv2.resize(frame_bgr, (FRAME_SCALE_WIDTH, int(h * ratio)))
        h, w = frame_bgr.shape[:2]

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    status_text = "No face"

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x,y,fw,fh = faces[0]
        cv2.rectangle(frame_bgr, (x,y), (x+fw, y+fh), (0,255,0), 2)

        lower_y = int(y + fh*0.5)
        face_lower = frame_bgr[lower_y:y+fh, x:x+fw]
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
                cv2.rectangle(frame_bgr, (mx_abs, my_abs), (mx_abs+mw, my_abs+mh), (255,255,0), 2)
            else:
                mx_abs = x + int(fw*0.15)
                my_abs = y + int(fh*0.65)
                mw = int(fw*0.7)
                mh = int(fh*0.25)
                mouth_box = (mx_abs, my_abs, mw, mh)
                cv2.rectangle(frame_bgr, (mx_abs, my_abs), (mx_abs+mw, my_abs+mh), (100,100,100), 1)

            mx, my, mw, mh = mouth_box
            mx = max(0, mx); my = max(0, my)
            mx2 = min(frame_bgr.shape[1], mx+mw); my2 = min(frame_bgr.shape[0], my+mh)
            roi = frame_bgr[my:my2, mx:mx2]
            if roi.size != 0:
                inp = preprocess_roi_for_model(roi)
                try:
                    pred = model.predict(inp, verbose=0)
                    pred = np.asarray(pred)
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

    # overlay info
    cv2.putText(frame_bgr, f"Status: {status_text}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame_bgr, f"Yawns last {ALERT_WINDOW_SECONDS//60}m: {recent}", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    with state_lock:
        total_display = yawn_total
    cv2.putText(frame_bgr, f"Yawns total: {total_display}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    if alert_on:
        cv2.rectangle(frame_bgr, (0,0), (frame_bgr.shape[1], 50), (0,0,255), -1)
        cv2.putText(frame_bgr, f"ALERT: {recent} yawns in last {ALERT_WINDOW_SECONDS//60} minutes", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    return frame_bgr, status_text, recent, alert_on

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        j = request.get_json(force=True)
        if 'image' not in j:
            return jsonify({'error': 'no image provided'}), 400
        frame = bgr_from_dataurl(j['image'])
        annotated, status_text, recent, alert_on = process_frame_and_annotate(frame)
        data_url = dataurl_from_bgr_image(annotated)
        return jsonify({
            'status': status_text,
            'image': data_url,
            'yawn_count_recent': recent,
            'alert': bool(alert_on)
        })
    except Exception as e:
        print("Predict endpoint error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    global consecutive_yawn_frames, yawn_timestamps, yawn_total, yawn_active
    with state_lock:
        consecutive_yawn_frames = 0
        yawn_timestamps = []
        yawn_total = 0
        yawn_active = False
    return jsonify({'status':'reset'})

if __name__ == '__main__':
    # Run on local network (0.0.0.0) so other devices can reach it if needed.
    app.run(host='0.0.0.0', port=5000, debug=False)
