"""
app.py - backend for Air Drawing Studio (the browser does camera, hand tracking and voice).
  GET  /            the page
  POST /api/solve   read a drawn arithmetic expression with the CNN
  POST /api/say     spoken arithmetic ("25 times 4")
  POST /api/pdf     upload a PDF -> index its text            (study.py)
  POST /api/ask     answer a question / explain a phrase      (study.py)
  POST /api/summary summarise a page or the whole PDF         (study.py)
  POST /api/quiz    make a quiz from the PDF                  (study.py)
  POST /api/sample  save one drawn symbol to dataset/ (developer tool)
Run:  py -3.11 app.py   ->  http://localhost:5000
"""
print("Starting Air Drawing System... loading TensorFlow (can take 20-40 seconds, please wait)", flush=True)
import base64
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory

import study
from main import (LABELS, MODEL_PATH, fmt_value, merge_boxes, prep_symbol,
                  pretty_expr, safe_eval, spoken_to_expression, words_to_numbers)

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024        # PDFs up to 25 MB
model = tf.keras.models.load_model(os.path.join(BASE, MODEL_PATH))
model(np.zeros((1, 28, 28, 1), np.float32), training=False)
print("Model loaded.", flush=True)
print("AI-written explanations:", study.llm_status(), flush=True)

LABEL_DIRS = {**{str(i): str(i) for i in range(10)},
              "+": "plus", "-": "minus", "*": "times", "/": "divide", "=": "equals"}


# ------------------------------------------------------------------ vision --
def mask_from_data_url(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    m = img[:, :, 3] if img.ndim == 3 and img.shape[2] == 4 else (img.max(axis=2) if img.ndim == 3 else img)
    return (m > 100).astype(np.uint8) * 255


def find_boxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [b for b in (cv2.boundingRect(c) for c in contours) if b[2] >= 4 and b[3] >= 4]
    if not boxes:
        return []
    boxes = merge_boxes(boxes)
    biggest = max(max(b[2], b[3]) for b in boxes)
    return sorted((b for b in boxes if max(b[2], b[3]) >= 0.12 * biggest), key=lambda b: b[0])


def recognize(mask):
    boxes = find_boxes(mask)
    if not boxes:
        return []
    batch = np.stack([prep_symbol(mask[y:y + h, x:x + w]) for x, y, w, h in boxes])
    probs = model((batch.astype("float32") / 255.0).reshape(-1, 28, 28, 1), training=False).numpy()
    return [{"label": LABELS[int(np.argmax(p))], "conf": float(np.max(p)), "box": list(map(int, b))}
            for b, p in zip(boxes, probs)]


def evaluate(expr):
    try:
        return fmt_value(safe_eval(expr)), None
    except ZeroDivisionError:
        return "cannot divide by zero", None
    except Exception:
        return None, "Could not understand that expression"


# ------------------------------------------------------------------ routes --
@app.get("/")
def index():
    for folder in (os.path.join(BASE, "static"), BASE):
        if os.path.exists(os.path.join(folder, "index.html")):
            return send_from_directory(folder, "index.html")
    return f"index.html not found. Put it in a folder named 'static' next to app.py (looked in {BASE}).", 404


@app.get("/health")
def health():
    return "ok"


@app.post("/api/solve")
def solve():
    data = request.get_json(force=True, silent=True) or {}
    mask = mask_from_data_url(data.get("image", ""))
    if mask is None:
        return jsonify(error="Bad image"), 400
    symbols = recognize(mask)
    if not symbols:
        return jsonify(error="Nothing to solve - draw an expression first")
    if data.get("auto") and symbols[-1]["label"] != "=":
        return jsonify(skip=True)
    expr = "".join(s["label"] for s in symbols).split("=")[0]
    if not expr:
        return jsonify(error="Write something before the '='")
    value, err = evaluate(expr)
    if err:
        return jsonify(error=f"{err}: {pretty_expr(expr)}", symbols=symbols)
    return jsonify(expr=pretty_expr(expr), value=value, symbols=symbols,
                   unsure=min(s["conf"] for s in symbols) < 0.6)


@app.post("/api/say")
def say():
    text = ((request.get_json(force=True, silent=True) or {}).get("text") or "").lower()
    expr = spoken_to_expression(words_to_numbers(text))
    if not expr:
        return jsonify(math=False)
    value, err = evaluate(expr)
    if err:
        return jsonify(math=True, error="Sorry, I could not calculate that")
    return jsonify(math=True, expr=pretty_expr(expr), value=value)


@app.get("/api/status")
def status():
    return jsonify(mode=study.ai_mode(), detail=study.llm_status())


NO_DOC = {"ok": False, "error": "Open the PDF again (the AI index is missing)"}


def _doc():
    data = request.get_json(force=True, silent=True) or {}
    return data, study.DOCS.get(data.get("doc_id"))


@app.post("/api/pdf")
def pdf_upload():
    f = request.files.get("file")
    if not f:
        return jsonify(ok=False, error="No file received")
    try:
        doc_id, doc = study.build(f.read())
    except ValueError as e:
        return jsonify(ok=False, error=str(e))
    except Exception:
        return jsonify(ok=False, error="Could not read this PDF")
    return jsonify(ok=True, doc_id=doc_id, pages=doc["pages"], sentences=len(doc["sents"]))


@app.post("/api/ask")
def ask_pdf():
    data, doc = _doc()
    if not doc:
        return jsonify(NO_DOC)
    if data.get("focus"):                                   # "explain this" (text under the finger / highlight)
        res = study.explain_focus(doc, data["focus"], data.get("page"))
    else:
        res = study.answer(doc, data.get("question") or "")
    return jsonify(ok=True, found=bool(res), **(res or {}))


@app.post("/api/summary")
def summary_pdf():
    data, doc = _doc()
    if not doc:
        return jsonify(NO_DOC)
    if data.get("page"):                                    # one slide/page: explain it, do not just read it back
        r = study.slide_summary(doc, int(data["page"]))
        if r:
            return jsonify(ok=True, text=r["answer"], speak=r["speak"], via=r.get("via"), sentences=[])
    sents = study.summary(doc, data.get("page"), int(data.get("n") or 4))
    if sents and not data.get("page"):                      # whole document: let the AI write it in its own words
        ai = study.llm_summary(sents)
        if ai:
            return jsonify(ok=True, text=ai, speak=study._first_sentence(ai), via="llm", sentences=sents)
    return jsonify(ok=True, sentences=sents) if sents else jsonify(ok=False, error="No text on this page (a slide that is only a picture cannot be read)")


@app.post("/api/quiz")
def quiz_pdf():
    data, doc = _doc()
    if not doc:
        return jsonify(NO_DOC)
    qs = study.quiz(doc, int(data.get("n") or 5))
    return jsonify(ok=True, questions=qs) if qs else jsonify(ok=False, error="This PDF is too short for a quiz")


@app.post("/api/sample")
def sample():
    data = request.get_json(force=True, silent=True) or {}
    label, mask = data.get("label"), mask_from_data_url(data.get("image", ""))
    if label not in LABEL_DIRS or mask is None:
        return jsonify(ok=False, error="Bad label or image"), 400
    boxes = find_boxes(mask)
    if len(boxes) != 1:
        return jsonify(ok=False, error=f"Draw exactly ONE symbol (found {len(boxes)})")
    x, y, w, h = boxes[0]
    folder = os.path.join(BASE, "dataset", LABEL_DIRS[label])
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, f"{int(time.time() * 1000)}.png"), prep_symbol(mask[y:y + h, x:x + w]))
    total = sum(len(os.listdir(os.path.join(BASE, "dataset", d))) for d in os.listdir(os.path.join(BASE, "dataset")))
    return jsonify(ok=True, count=len(os.listdir(folder)), total=total)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n>>> READY. Open Chrome and go to  http://localhost:{port}   (keep this window open) <<<\n", flush=True)
    app.run(host="0.0.0.0", port=port)