"""
model_utils.py — VinceNet TFLite Inference Engine
==================================================
Model   : vincenet_stroke_f32.tflite  (34 MB)  ← primary
          vincenet_stroke_int8.tflite (8.6 MB) ← quantised alternative
Input   : (1, 128, 128, 3) float32
Preproc : pixel / 127.5 - 1.0  →  [-1, 1]   (Xception-style)
Output  : (1, 1) float32 — sigmoid probability
Threshold: 0.987239  ← from model_meta.json (NOT 0.5)
ROC-AUC : 0.99999
"""

import io, logging
import numpy as np
from PIL import Image

logger    = logging.getLogger(__name__)
IMG_SIZE  = 128
THRESHOLD = 0.987239420413971   # ← trained threshold from model_meta.json


def load_model(model_path: str):
    """Load TFLite model once at startup. Returns interpreter or None."""
    import os, tensorflow as tf
    if not os.path.exists(model_path):
        logger.warning(f"Model not found: {model_path}  →  DEMO mode")
        return None
    logger.info(f"Loading VinceNet: {model_path}  ({os.path.getsize(model_path)//1_000_000} MB)")
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    i = interp.get_input_details()[0]
    o = interp.get_output_details()[0]
    logger.info(f"  Input  : {i['shape']}  {i['dtype'].__name__}")
    logger.info(f"  Output : {o['shape']}  {o['dtype'].__name__}")
    logger.info(f"  Threshold: {THRESHOLD}")
    return interp


def preprocess(image_bytes: bytes) -> np.ndarray:
    """bytes  →  (1, 128, 128, 3) float32  in [-1, 1]"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = arr / 127.5 - 1.0          # Xception preprocess_input
    return np.expand_dims(arr, 0)    # (1, 128, 128, 3)


def infer_one(interp, image_bytes: bytes) -> dict:
    """
    Run inference on one image.
    Returns dict with label, confidence, raw_prob.
    """
    if interp is None:
        return _demo(image_bytes)

    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    tensor = preprocess(image_bytes)

    # Handle int8 quantised model
    if inp_d["dtype"] == np.int8:
        s, z   = inp_d["quantization"]
        tensor = (tensor / s + z).astype(np.int8)

    interp.set_tensor(inp_d["index"], tensor)
    interp.invoke()
    raw = interp.get_tensor(out_d["index"])

    # Dequantise int8 output
    if out_d["dtype"] == np.int8:
        s, z = out_d["quantization"]
        raw  = (raw.astype(np.float32) - z) * s

    prob  = float(raw.ravel()[0])
    label = "Stroke" if prob >= THRESHOLD else "Normal"
    conf  = prob if label == "Stroke" else (1.0 - prob)

    return {
        "label":    label,
        "conf":     round(conf, 6),
        "raw_prob": round(prob, 6),
    }


def infer_batch(interp, images: list[bytes]) -> list[dict]:
    return [infer_one(interp, b) for b in images]


def aggregate(results: list[dict]) -> dict:
    """Majority vote → average confidence of winning class."""
    if len(results) == 1:
        return {"label": results[0]["label"], "conf": results[0]["conf"]}

    strokes = [r["conf"] for r in results if r["label"] == "Stroke"]
    normals = [r["conf"] for r in results if r["label"] == "Normal"]

    if len(strokes) >= len(normals):
        return {"label": "Stroke",
                "conf":  round(sum(strokes) / len(strokes), 6) if strokes else 0.5}
    return {"label": "Normal",
            "conf":  round(sum(normals) / len(normals), 6) if normals else 0.5}


def _demo(image_bytes: bytes) -> dict:
    """Deterministic demo output when no model is loaded."""
    cs    = sum(image_bytes[:512]) % 100
    label = "Stroke" if cs > 45 else "Normal"
    prob  = round(0.993 + (cs - 45) / 1000, 4) if label == "Stroke" else round(0.003, 4)
    conf  = prob if label == "Stroke" else (1.0 - prob)
    return {"label": label, "conf": round(conf, 4), "raw_prob": round(prob, 4)}
