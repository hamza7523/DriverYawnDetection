# server.py
# runtime server: uses ONNX Runtime for yawn model and ultralytics YOLO for face detection

import io
import base64
import time
import threading
from datetime import datetime
from pathlib import Path
from collections import deque
import json

from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

# filenames expected in project root
ONNX_MODEL = "yawn_model.onnx"
META_FILE = "model_metaPytorch.json"
YOLO_FACE = "yolov8n_face.pt"   # put your yolov8 face weights here

# runtime params
FRAME_SCALE_WIDTH = 640
CONSECUTIVE_REQUIRED = 6
YAWN_PROB_THRESHOLD = 0.5
ALERT_COUNT = 5
ALERT_WINDOW_SECONDS = 120

# state
state_lock = threading.Lock()
consecutive_yawn_frames = 0
yawn_timestamps = deque()
yawn_total = 0
yawn_active = False

def prune_old_timestamps():
    cutoff = time.time() - ALERT_WINDOW_SECONDS
    with state_lock:
        while yawn_timestamps and yawn_timestamps[0] < cutoff:
            yawn_timestamps.popleft()

def count_recent():
    prune_old_timestamps()
    with state_lock:
        return len(yawn_timestamps)

# load meta
meta_path = Path(META_FILE)
if not meta_path.exists():
    raise FileNotFoundError(f"Missing {META_FILE}. Run conversion script first.")
meta = json.loads(meta_path.read_text())
IN_H = int(meta.get("in_h", 96))
IN_W = int(meta.get("in_w", 96))
IN_C = int(meta.get("in_c", 3))
ONNX_INPUT_NAME = meta.get("onnx_input_name")
INPUT_LAYOUT = meta.get("onnx_input_layout", "NHWC")
print("Model meta:", meta)

# load onnx runtime session
onnx_path = Path(ONNX_MODEL)
if not onnx_path.exists():
    raise FileNotFoundError(f"Missing ONNX model: {ONNX_MODEL}. Run conversion script first.")
print("Loading ONNX Runtime session...")
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

# load YOLO face detector
yolo_path = Path(YOLO_FACE)
if not yolo_path.exists():
    raise FileNotFoundError(f"Missing YOLO face weights: {YOLO_FACE}")
print("Loading YOLO face detector...")
face_model = YOLO(str(yolo_path))

# helper image conversions
def dataurl_from_pil(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b}"

def pil_from_dataurl(data_url):
    header, b64 = data_url.split(",", 1)
    b = base64.b64decode(b64)
    return Image.open(io.BytesIO(b)).convert("RGB")

# preprocess for ONNX according to input layout
def preprocess_roi(pil_roi):
    roi = pil_roi.resize((IN_W, IN_H)).convert("RGB")
    arr = np.asarray(roi).astype("float32") / 255.0  # H W C (NHWC)
    if INPUT_LAYOUT == "NHWC":
        inp = np.expand_dims(arr, axis=0)  # 1 H W C
    else:
        # NCHW
        inp = np.transpose(arr, (2, 0, 1))[None, ...]  # 1 C H W
    return inp

def run_onnx_inference(inp_array):
    # sess.run expects feeds as {input_name: arr}
    feed = {ONNX_INPUT_NAME: inp_array}
    out = sess.run(None, feed)
    return out

# core processing
def process_frame_and_annotate(pil_img):
    global consecutive_yawn_frames, yawn_total, yawn_active

    # scale to FRAME_SCALE_WIDTH width
    w0, h0 = pil_img.size
    if w0 != FRAME_SCALE_WIDTH:
        ratio = FRAME_SCALE_WIDTH / float(w0)
        pil_img = pil_img.resize((FRAME_SCALE_WIDTH, int(h0 * ratio)))

    draw = ImageDraw.Draw(pil_img)
    status_text = "No face"

    # detect faces
    results = face_model(pil_img, imgsz=640, half=False)
    face_box = None
    if len(results) > 0:
        res = results[0]
        boxes_obj = getattr(res.boxes, "xyxy", None)
        confs_obj = getattr(res.boxes, "conf", None)
        if boxes_obj is not None and len(boxes_obj) > 0:
            try:
                boxes_list = boxes_obj.tolist()
            except Exception:
                boxes_list = [[float(x) for x in boxes_obj[i]] for i in range(len(boxes_obj))]
            best = None
            best_area = 0.0
            for i, b in enumerate(boxes_list):
                x1, y1, x2, y2 = b
                area = (x2 - x1) * (y2 - y1)
                conf_val = 0.0
                try:
                    conf_val = float(confs_obj[i])
                except Exception:
                    try:
                        conf_val = float(confs_obj[i].item())
                    except Exception:
                        conf_val = 0.0
                if area > best_area:
                    best_area = area
                    best = (x1, y1, x2, y2, conf_val)
            if best is not None:
                face_box = best

    if face_box is None:
        with state_lock:
            consecutive_yawn_frames = 0
            yawn_active = False
        status_text = "No face"
    else:
        x1, y1, x2, y2, conf = face_box
        draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=3)
        fh = y2 - y1
        lower_top = y1 + int(fh * 0.55)
        mouth_box = (x1, lower_top, x2, y2)
        draw.rectangle(mouth_box, outline="yellow", width=2)

        mx1, my1, mx2, my2 = [int(v) for v in mouth_box]
        mx1 = max(0, mx1); my1 = max(0, my1)
        mx2 = min(pil_img.width, mx2); my2 = min(pil_img.height, my2)

        if mx2 <= mx1 or my2 <= my1:
            with state_lock:
                consecutive_yawn_frames = 0
                yawn_active = False
            status_text = "Empty mouth roi"
        else:
            roi = pil_img.crop((mx1, my1, mx2, my2)).convert("RGB")
            inp = preprocess_roi(roi)  # numpy array shaped according to INPUT_LAYOUT
            try:
                out = run_onnx_inference(inp)
                # out is a list of outputs; we choose first output
                out0 = out[0]
                out0 = np.asarray(out0)
                # handle shapes: (1,), (1,1), (1,2), etc
                prob = 0.0
                if out0.ndim == 1 and out0.size == 1:
                    prob = float(out0.ravel()[-1])
                elif out0.ndim == 2 and out0.shape[1] == 1:
                    prob = float(out0[0,0])
                elif out0.ndim == 2 and out0.shape[1] >= 2:
                    prob = float(out0[0,1])
                else:
                    prob = float(out0.ravel()[-1])
            except Exception as e:
                print("ONNX inference error:", e)
                prob = 0.0

            if prob > YAWN_PROB_THRESHOLD:
                with state_lock:
                    consecutive_yawn_frames += 1
                    if (not yawn_active) and (consecutive_yawn_frames >= CONSECUTIVE_REQUIRED):
                        yawn_active = True
                        yawn_timestamps.append(time.time())
                        yawn_total += 1
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Yawn counted total={yawn_total}")
                status_text = f"Yawning {prob:.2f}"
            else:
                with state_lock:
                    consecutive_yawn_frames = 0
                    yawn_active = False
                status_text = f"Monitoring {prob:.2f}"

    recent = count_recent()
    alert_on = recent >= ALERT_COUNT

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((10, 10), f"Status: {status_text}", fill="white", font=font)
    draw.text((10, 30), f"Yawns last {ALERT_WINDOW_SECONDS//60}m: {recent}", fill="white", font=font)
    draw.text((10, 50), f"Yawns total: {yawn_total}", fill="white", font=font)
    if alert_on:
        draw.rectangle([(0, pil_img.height - 40), (pil_img.width, pil_img.height)], fill="red")
        draw.text((10, pil_img.height - 30), f"ALERT: {recent} yawns recent", fill="white", font=font)

    return pil_img, status_text, recent, alert_on

# Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        j = request.get_json(force=True)
        if "image" not in j:
            return jsonify({"error": "no image provided"}), 400
        pil = pil_from_dataurl(j["image"])
        annotated, status_text, recent, alert_on = process_frame_and_annotate(pil)
        data_url = dataurl_from_pil(annotated)
        return jsonify({
            "status": status_text,
            "image": data_url,
            "yawn_count_recent": recent,
            "alert": bool(alert_on)
        })
    except Exception as e:
        print("Predict error", e)
        return jsonify({"error": str(e)}), 500

@app.route("/reset", methods=["POST"])
def reset():
    global consecutive_yawn_frames, yawn_timestamps, yawn_total, yawn_active
    with state_lock:
        consecutive_yawn_frames = 0
        yawn_timestamps.clear()
        yawn_total = 0
        yawn_active = False
    return jsonify({"status": "reset"})
