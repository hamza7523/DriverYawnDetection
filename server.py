# server.py (updated, more robust error handling + YOLO parsing)
import io, base64, time, threading, traceback
from datetime import datetime
from pathlib import Path
from collections import deque
import json

from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

# ---------- CONFIG ----------
DEBUG = True  # set False in production on Render
ONNX_MODEL = "yawn_model.onnx"           # your ONNX model (or yawn_96.onnx)
META_FILE = "model_metaPytorch.json"
YOLO_FACE = "yolov8n-face.pt"
FRAME_SCALE_WIDTH = 640
CONSECUTIVE_REQUIRED = 6
YAWN_PROB_THRESHOLD = 0.5
ALERT_COUNT = 5
ALERT_WINDOW_SECONDS = 120
# ----------------------------

app = Flask(__name__, static_folder="static", template_folder="templates")

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

# helper: load meta
meta_path = Path(META_FILE)
if not meta_path.exists():
    if DEBUG:
        print(f"Warning: {META_FILE} not found. Proceeding with defaults (96x96x3)")
    meta = {}
else:
    meta = json.loads(meta_path.read_text())

IN_H = int(meta.get("in_h", 96))
IN_W = int(meta.get("in_w", 96))
IN_C = int(meta.get("in_c", 3))
ONNX_INPUT_NAME = meta.get("onnx_input_name", None)
INPUT_LAYOUT = meta.get("onnx_input_layout", "NHWC")

# load ONNX runtime
onnx_path = Path(ONNX_MODEL)
if not onnx_path.exists():
    if DEBUG:
        print(f"Warning: ONNX model {ONNX_MODEL} not found. ONNX inference disabled.")
    sess = None
else:
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        # detect input name if not provided
        if ONNX_INPUT_NAME is None:
            ONNX_INPUT_NAME = sess.get_inputs()[0].name
        if DEBUG:
            print("ONNX session created. input name:", ONNX_INPUT_NAME)
    except Exception as e:
        print("Failed to create ONNX session:", e)
        sess = None

# load YOLO face model
yolo_path = Path(YOLO_FACE)
if not yolo_path.exists():
    raise FileNotFoundError(f"Missing YOLO weights: {YOLO_FACE}. Put the .pt file in repo root.")

try:
    face_model = YOLO(str(yolo_path))
    if DEBUG: print("Loaded YOLO model.")
except Exception as e:
    print("Failed to load YOLO model:", e)
    raise

# PIL <> base64 helpers
def dataurl_from_pil(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def pil_from_dataurl(data_url):
    header, b64 = data_url.split(",", 1) if "," in data_url else ("", data_url)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

# preprocess for ONNX
def preprocess_roi(pil_roi):
    roi = pil_roi.resize((IN_W, IN_H)).convert("RGB")
    arr = np.asarray(roi).astype("float32") / 255.0
    if INPUT_LAYOUT.upper() == "NHWC":
        inp = arr[None, ...]  # 1,H,W,C
    else:
        inp = np.transpose(arr, (2,0,1))[None, ...]  # 1,C,H,W
    return inp

def run_onnx_inference(inp_array):
    if sess is None:
        raise RuntimeError("ONNX session not initialized")
    feed = {ONNX_INPUT_NAME: inp_array} if ONNX_INPUT_NAME else {sess.get_inputs()[0].name: inp_array}
    out = sess.run(None, feed)
    return out

# Robust extraction from ultralytics Result
def extract_boxes_and_confs(result):
    boxes = []
    confs = []
    try:
        boxes_obj = getattr(result.boxes, "xyxy", None)
        confs_obj = getattr(result.boxes, "conf", None)
        if boxes_obj is not None:
            # try several access patterns
            try:
                arr = boxes_obj.cpu().numpy()
            except Exception:
                try:
                    arr = np.array(boxes_obj)
                except Exception:
                    arr = None
            if arr is not None:
                boxes = arr.tolist()
        if confs_obj is not None:
            try:
                c_arr = confs_obj.cpu().numpy()
            except Exception:
                try:
                    c_arr = np.array(confs_obj)
                except Exception:
                    c_arr = None
            if c_arr is not None:
                confs = c_arr.tolist()
    except Exception as e:
        if DEBUG: print("extract_boxes_and_confs error:", e)
    return boxes, confs

# core processing
def process_frame_and_annotate(pil_img):
    global consecutive_yawn_frames, yawn_total, yawn_active

    try:
        w0, h0 = pil_img.size
        if w0 != FRAME_SCALE_WIDTH:
            ratio = FRAME_SCALE_WIDTH / float(w0)
            pil_img = pil_img.resize((FRAME_SCALE_WIDTH, int(h0 * ratio)))

        draw = ImageDraw.Draw(pil_img)
        status_text = "No face"

        # run YOLO
        try:
            np_frame = np.array(pil_img)  # H,W,3 (RGB)
            results = face_model(np_frame, imgsz=640, half=False)  # list of Results
        except Exception as e:
            if DEBUG:
                print("YOLO inference failed:", e)
                traceback.print_exc()
            results = []

        face_box = None
        if results:
            res = results[0]
            boxes, confs = extract_boxes_and_confs(res)
            if boxes:
                # pick box with max area * conf
                best = None; best_score = -1.0
                for i, b in enumerate(boxes):
                    try:
                        x1, y1, x2, y2 = [float(v) for v in b]
                    except Exception:
                        continue
                    area = (x2 - x1) * (y2 - y1)
                    conf_val = confs[i] if (i < len(confs)) else 0.0
                    score = area * (conf_val if conf_val is not None else 1.0)
                    if score > best_score:
                        best_score = score
                        best = (x1, y1, x2, y2, float(conf_val) if conf_val is not None else 0.0)
                face_box = best

        # no face
        if face_box is None:
            with state_lock:
                consecutive_yawn_frames = 0
                yawn_active = False
            status_text = "No face"
        else:
            x1, y1, x2, y2, conf = face_box
            # clamp to ints and inside image
            x1_i = max(0, int(round(x1))); y1_i = max(0, int(round(y1)))
            x2_i = min(pil_img.width - 1, int(round(x2))); y2_i = min(pil_img.height - 1, int(round(y2)))
            draw.rectangle([(x1_i, y1_i), (x2_i, y2_i)], outline="green", width=3)

            fh = y2_i - y1_i
            lower_top = y1_i + int(fh * 0.55)
            mouth_box = (x1_i, lower_top, x2_i, y2_i)
            draw.rectangle(mouth_box, outline="yellow", width=2)

            mx1, my1, mx2, my2 = mouth_box
            if mx2 <= mx1 or my2 <= my1:
                with state_lock:
                    consecutive_yawn_frames = 0
                    yawn_active = False
                status_text = "Empty mouth roi"
            else:
                roi = pil_img.crop((mx1, my1, mx2, my2)).convert("RGB")
                # ONNX inference
                try:
                    inp = preprocess_roi(roi)
                    out = run_onnx_inference(inp)
                    out0 = np.asarray(out[0])
                    # robust prob extraction
                    prob = 0.0
                    if out0.ndim == 1 and out0.size >= 1:
                        prob = float(out0.ravel()[-1])
                    elif out0.ndim == 2 and out0.shape[1] == 1:
                        prob = float(out0[0,0])
                    elif out0.ndim == 2 and out0.shape[1] >= 2:
                        prob = float(out0[0,1])
                    else:
                        prob = float(out0.ravel()[-1])
                except Exception as e:
                    if DEBUG:
                        print("ONNX inference error:", e)
                        traceback.print_exc()
                    prob = 0.0

                if prob > YAWN_PROB_THRESHOLD:
                    with state_lock:
                        consecutive_yawn_frames += 1
                        if (not yawn_active) and (consecutive_yawn_frames >= CONSECUTIVE_REQUIRED):
                            yawn_active = True
                            yawn_timestamps.append(time.time())
                            yawn_total += 1
                            print(f"[{datetime.now()}] Yawn counted total={yawn_total}")
                    status_text = f"Yawning {prob:.2f}"
                else:
                    with state_lock:
                        consecutive_yawn_frames = 0
                        yawn_active = False
                    status_text = f"Monitoring {prob:.2f}"

        recent = count_recent()
        alert_on = recent >= ALERT_COUNT

        # overlays (text)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((10, 6), f"Status: {status_text}", fill="white", font=font)
        draw.text((10, 26), f"Yawns last {ALERT_WINDOW_SECONDS//60}m: {recent}", fill="white", font=font)
        draw.text((10, 46), f"Yawns total: {yawn_total}", fill="white", font=font)
        if alert_on:
            draw.rectangle([(0, pil_img.height - 40), (pil_img.width, pil_img.height)], fill="red")
            draw.text((10, pil_img.height - 30), f"ALERT: {recent} yawns recent", fill="white", font=font)

        return pil_img, status_text, recent, alert_on

    except Exception as e:
        # unexpected error while processing frame
        if DEBUG:
            print("Unexpected error in process_frame_and_annotate:")
            traceback.print_exc()
        raise

@app.route("/")
def index():
    return render_template("templates\index.html")

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
        # write full traceback to server log
        tb = traceback.format_exc()
        print("Predict error:", e)
        print(tb)
        # return structured error detail when debugging
        if DEBUG:
            return jsonify({"error": str(e), "traceback": tb}), 500
        else:
            return jsonify({"error": "internal server error"}), 500

@app.route("/reset", methods=["POST"])
def reset():
    global consecutive_yawn_frames, yawn_timestamps, yawn_total, yawn_active
    with state_lock:
        consecutive_yawn_frames = 0
        yawn_timestamps.clear()
        yawn_total = 0
        yawn_active = False
    return jsonify({"status": "reset"})

@app.route("/health")
def health():
    ok = True
    messages = []
    # quick checks
    messages.append(f"YOLO loaded: {'yes' if 'face_model' in globals() else 'no'}")
    messages.append(f"ONNX session: {'yes' if sess is not None else 'no'}")
    return jsonify({"ok": ok, "messages": messages})

if __name__ == "__main__":
    # local testing: enable debug and reload; in production use gunicorn
    app.run(host="0.0.0.0", port=5000, debug=DEBUG)
