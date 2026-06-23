"""
StrokeAI v4 — VinceNet Backend
================================
Port  : 9000
Open  : http://localhost:9000

One server serves BOTH the web UI and the prediction API.
No React, no Node.js, no CORS issues.

Model  : VinceNet (vincenet_stroke_f32.tflite)
Input  : (1, 128, 128, 3) float32 — Xception preprocessing
Output : (1, 1) float32 — binary sigmoid
Threshold: 0.987239  (from model_meta.json)
ROC-AUC  : 0.99999
"""

import os, io, base64, json, logging, asyncio
from contextlib import asynccontextmanager
from pathlib    import Path
from typing     import List

from fastapi                  import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors  import CORSMiddleware
from fastapi.responses        import HTMLResponse, JSONResponse
from PIL                      import Image
from dotenv                   import load_dotenv

from model_utils import load_model, infer_batch, aggregate, THRESHOLD
from xai_utils   import make_heatmap

# ── Config ────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE       = Path(__file__).parent
FRONTEND   = BASE.parent / "frontend" / "index.html"
META_FILE  = BASE / "model_meta.json"
MODEL_FILE = os.getenv("MODEL_FILE", "vincenet_stroke_f32.tflite")
ALLOWED    = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILES  = 10
MAX_BYTES  = 15 * 1024 * 1024

INTERP = None   # loaded once at startup


# ── App lifespan ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global INTERP
    INTERP = load_model(str(BASE / MODEL_FILE))
    logger.info(
        f"✅ VinceNet ready  threshold={THRESHOLD}"
        if INTERP else "⚠️  DEMO mode — no model file"
    )
    yield


app = FastAPI(title="StrokeAI v4 — VinceNet",
              version="4.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"], allow_credentials=True)


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def ui():
    """Serve the single-page frontend."""
    if FRONTEND.exists():
        return HTMLResponse(FRONTEND.read_text(encoding="utf-8"))
    return HTMLResponse(f"""
    <html><body style="background:#060d18;color:#e2e8f0;
    font-family:sans-serif;padding:40px">
    <h2>⚠️ frontend/index.html not found</h2>
    <p>Expected: <code>{FRONTEND}</code></p>
    <p>Folder structure must be:<br>
    <pre>vincenet-app/
  backend/   ← server lives here
  frontend/
    index.html</pre></p>
    <p>API is live: <a href="/health" style="color:#22d3ee">/health</a></p>
    </body></html>""", status_code=500)


@app.get("/health")
async def health():
    meta = json.loads(META_FILE.read_text()) if META_FILE.exists() else {}
    return {
        "status":       "ok",
        "model":        MODEL_FILE,
        "model_loaded": INTERP is not None,
        "demo_mode":    INTERP is None,
        "threshold":    THRESHOLD,
        "roc_auc":      meta.get("roc_auc",  0.99999),
        "pr_auc":       meta.get("pr_auc",   0.999985),
        "input_shape":  [1, 128, 128, 3],
        "port":         9000,
        "version":      "4.0.0",
    }


@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    # ── Validation ────────────────────────────────────────────────────
    if not files:
        raise HTTPException(400, "No files uploaded.")
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"Max {MAX_FILES} files per request.")

    payloads: list[tuple[str, bytes]] = []
    for f in files:
        if f.content_type not in ALLOWED:
            raise HTTPException(400,
                f"'{f.filename}': unsupported type '{f.content_type}'. "
                f"Use JPEG, PNG, WEBP or BMP.")
        data = await f.read()
        if not data:
            raise HTTPException(400, f"'{f.filename}' is empty.")
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"'{f.filename}' exceeds 15 MB.")
        try:
            Image.open(io.BytesIO(data)).verify()
        except Exception:
            raise HTTPException(400, f"'{f.filename}' is not a valid image.")
        payloads.append((f.filename or f"scan_{len(payloads)+1}", data))

    names  = [p[0] for p in payloads]
    images = [p[1] for p in payloads]

    # ── Batch inference ───────────────────────────────────────────────
    try:
        results = await asyncio.to_thread(infer_batch, INTERP, images)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(500, f"Inference error: {exc}")

    # ── XAI (stroke cases only) ───────────────────────────────────────
    async def xai(r: dict, img: bytes) -> str | None:
        if r["label"] != "Stroke":
            return None
        return await asyncio.to_thread(make_heatmap, INTERP, img)

    heatmaps = await asyncio.gather(
        *[xai(r, img) for r, img in zip(results, images)]
    )

    # ── Aggregate ─────────────────────────────────────────────────────
    agg = aggregate(results)

    # ── Explanation ───────────────────────────────────────────────────
    explanation = await asyncio.to_thread(
        _explain, agg["label"], agg["conf"], len(images), images[0]
    )

    # ── Response ──────────────────────────────────────────────────────
    individual = [
        {
            "filename":   names[i],
            "prediction": r["label"],
            "confidence": r["conf"],
            "raw_prob":   r["raw_prob"],
            "heatmap":    heatmaps[i],
        }
        for i, r in enumerate(results)
    ]

    return JSONResponse({
        "prediction":         agg["label"],
        "confidence":         agg["conf"],
        "individual_results": individual,
        "explanation":        explanation,
        "xai_images":         [h for h in heatmaps if h],
        "model_info": {
            "name":      "VinceNet",
            "file":      MODEL_FILE,
            "threshold": THRESHOLD,
            "roc_auc":   0.99999,
            "pr_auc":    0.999985,
        },
    })


# ── Explanation helpers ───────────────────────────────────────────────

def _explain(label: str, conf: float, n: int, img_bytes: bytes) -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key == "your-key-here":
        return _fallback(label, conf, n)
    try:
        import anthropic
        b64    = base64.standard_b64encode(img_bytes).decode()
        client = anthropic.Anthropic(api_key=key)
        msg    = client.messages.create(
            model="claude-opus-4-5", max_tokens=400,
            messages=[{"role": "user", "content": [
                {"type": "image",
                 "source": {"type": "base64",
                            "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": (
                    f"VinceNet (ROC-AUC 0.99999, threshold 0.987) analysed "
                    f"{n} brain CT scan(s): {label} ({conf:.1%} confidence).\n"
                    "Provide: 1) Plain-language explanation  "
                    "2) Clinical interpretation with likely anatomy  "
                    "3) Recommended next steps  "
                    "4) Disclaimer. Be concise and professional."
                )},
            ]}]
        )
        return msg.content[0].text
    except Exception as exc:
        logger.warning(f"Claude API: {exc}")
        return _fallback(label, conf, n)


def _fallback(label: str, conf: float, n: int) -> str:
    pct = f"{conf:.1%}"
    if label == "Stroke":
        return (
            f"VinceNet detected stroke indicators in {n} scan(s) "
            f"with {pct} confidence (threshold: {THRESHOLD:.3f}, ROC-AUC: 0.99999).\n\n"
            "Clinical Interpretation: The XAI heatmap highlights regions of greatest "
            "model attention — potentially hypodense tissue, signal asymmetry, or cortical "
            "changes consistent with ischemic injury, possibly in the MCA territory.\n\n"
            "Next Steps: Seek immediate emergency neurological evaluation. "
            "A radiologist should review the original scans alongside clinical symptoms "
            "such as sudden weakness, speech difficulty, or facial asymmetry.\n\n"
            "⚕️ Disclaimer: VinceNet is a research AI tool only. "
            "It does not constitute a clinical diagnosis."
        )
    return (
        f"VinceNet found no stroke indicators in {n} scan(s) "
        f"({pct} confidence, threshold: {THRESHOLD:.3f}).\n\n"
        "Clinical Interpretation: Scans appear within normal limits for the features "
        "evaluated. No significant hypodense regions or asymmetry detected.\n\n"
        "Next Steps: If neurological symptoms persist, consult a neurologist. "
        "A negative AI result does not exclude all pathology.\n\n"
        "⚕️ Disclaimer: VinceNet is a research AI tool only. "
        "Always confirm with a qualified healthcare professional."
    )
