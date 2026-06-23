"""
xai_utils.py — Occlusion-Sensitivity XAI for VinceNet
=======================================================
Works with frozen TFLite (no gradients needed).
Grid: 8×8 patches, JET colormap, 45 % heatmap blend.
Returns base64-encoded PNG or None on error.
"""

import io, base64, logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
GRID  = 8      # 8×8 occlusion grid = 64 patches
ALPHA = 0.45   # heatmap opacity in blend


# ── JET colormap (pure numpy) ─────────────────────────────────────────
def _jet(v: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _blend(orig: np.ndarray, hm_rgb: np.ndarray) -> np.ndarray:
    return np.clip(
        (1 - ALPHA) * orig.astype(np.float32) + ALPHA * hm_rgb.astype(np.float32),
        0, 255
    ).astype(np.uint8)


def _to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode()


# ── Public API ────────────────────────────────────────────────────────
def make_heatmap(interp, image_bytes: bytes) -> str | None:
    """
    Generate XAI heatmap. Returns base64 PNG or None.
    interp=None → synthetic Gaussian demo blob.
    """
    try:
        orig       = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ow, oh     = orig.size
        orig_arr   = np.array(orig, dtype=np.float32)

        if interp is None:
            hm = _demo_blob(ow, oh)
        else:
            hm = _occlusion_map(interp, image_bytes, ow, oh)

        hm_rgb  = _jet(hm)
        blended = _blend(orig_arr, hm_rgb)
        return _to_b64(blended)

    except Exception as exc:
        logger.warning(f"XAI failed: {exc}")
        return None


# ── Occlusion sensitivity ─────────────────────────────────────────────
def _occlusion_map(interp, image_bytes: bytes, ow: int, oh: int) -> np.ndarray:
    from model_utils import preprocess
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    # Baseline probability
    tensor = preprocess(image_bytes)
    _invoke(interp, inp_d, out_d, tensor)
    baseline = _read_out(interp, out_d)

    # Occlusion grid
    ph  = 128 // GRID
    pw  = 128 // GRID
    imp = np.zeros((GRID, GRID), dtype=np.float32)

    for r in range(GRID):
        for c in range(GRID):
            occ = tensor.copy()
            patch = occ[0, r*ph:(r+1)*ph, c*pw:(c+1)*pw, :]
            occ[0, r*ph:(r+1)*ph, c*pw:(c+1)*pw, :] = \
                patch.mean(axis=(0, 1), keepdims=True)
            _invoke(interp, inp_d, out_d, occ)
            imp[r, c] = max(0.0, baseline - _read_out(interp, out_d))

    if imp.max() > 1e-8:
        imp /= imp.max()

    # Upsample to original image size
    hm = np.array(
        Image.fromarray((imp * 255).astype(np.uint8)).resize((ow, oh), Image.BILINEAR),
        dtype=np.float32
    ) / 255.0
    return hm


def _invoke(interp, inp_d, out_d, tensor):
    t = tensor
    if inp_d["dtype"] == np.int8:
        s, z = inp_d["quantization"]
        t    = (t / s + z).astype(np.int8)
    interp.set_tensor(inp_d["index"], t)
    interp.invoke()


def _read_out(interp, out_d) -> float:
    raw = interp.get_tensor(out_d["index"])
    if out_d["dtype"] == np.int8:
        s, z = out_d["quantization"]
        raw  = (raw.astype(np.float32) - z) * s
    return float(raw.ravel()[0])


# ── Demo blob (no model) ──────────────────────────────────────────────
def _demo_blob(ow: int, oh: int) -> np.ndarray:
    cx, cy = int(ow * 0.62), int(oh * 0.38)
    Y, X   = np.ogrid[:oh, :ow]
    sigma  = min(ow, oh) * 0.18
    hm     = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    return hm.astype(np.float32)
