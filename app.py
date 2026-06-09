"""
Herbal Plant Recognition - GUI Application
==========================================
Browser-based interface built with Gradio.

Features
--------
* Camera capture or image upload
* YOLOv8 instance-segmentation inference (one result per herb species)
* Unknown-herb detection → expert consultation form
* Feedback loop: correct a wrong prediction and fine-tune the model

Usage
-----
    python app.py
    python app.py --port 7860 --share
"""
import argparse
import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# ── background retraining state ───────────────────────────────────────────────
_retrain_thread: threading.Thread | None = None
_retrain_lock   = threading.Lock()   # prevents two concurrent training runs


def _background_retrain(device: str = "cpu") -> None:
    """Run fine-tuning in a daemon thread so the UI stays responsive."""
    from retrain import retrain_with_feedback
    try:
        success = retrain_with_feedback(epochs=40, device=device)
        if success:
            reload_model()
            print("[INFO] Background fine-tuning complete. Model reloaded.")
        else:
            print("[INFO] Background fine-tuning: nothing new to train on.")
    except Exception as exc:
        print(f"[ERROR] Background fine-tuning failed: {exc}")

BASE_DIR = Path(__file__).parent

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH          = BASE_DIR / "runs" / "train" / "weights" / "best.pt"
YAML_PATH           = BASE_DIR / "dataset" / "data.yaml"
EXPERT_QUERIES_FILE = BASE_DIR / "expert_queries.json"
EXPERT_IMAGES_DIR   = BASE_DIR / "expert_images"

CONFIDENCE_THRESHOLD = 0.35
UNKNOWN_CLASS_NAME   = "New"


# ── class names ───────────────────────────────────────────────────────────────

def load_class_names() -> list[str]:
    if YAML_PATH.exists():
        with open(YAML_PATH) as f:
            return yaml.safe_load(f).get("names", [])
    return [
        "Bitter Leaf - Mululuza", "Boerhavia diffusa - Olweza",
        "Ceratonia siliqua - Omutulika", "Chenopodium album - Omwetango",
        "False Daisy - Mutaayiza", "Himalayan balsam - Muzukizi",
        "Hoslundia opposita - Kamunye", "Justicia pectoralis - Muzuukizi",
        "Leucaena leucocephala - Lusina", "Momordica foetida - Ebombo",
        "New", "Perilla frutescens", "Plectranthus prostratus - Mubiri",
        "Plectrarithus cyaneus - Kibwankulata", "Sugar cane - Kikajo", "peppermint",
    ]


CLASS_NAMES: list[str] = load_class_names()

# Choices for the "correct herb" dropdown — all real herbs + Unknown
CORRECT_LABEL_CHOICES = [n for n in CLASS_NAMES if n != UNKNOWN_CLASS_NAME] + ["Unknown / New species"]


# ── model ─────────────────────────────────────────────────────────────────────

_model = None


def get_model():
    global _model
    if _model is not None:
        return _model, None
    if not MODEL_PATH.exists():
        return None, (
            "**Model not found.** "
            f"Expected `{MODEL_PATH}`.  Run `python train.py` first."
        )
    try:
        from ultralytics import YOLO  # type: ignore[import]
        _model = YOLO(str(MODEL_PATH))
        print(f"[INFO] Model loaded: {MODEL_PATH}")
        return _model, None
    except Exception as exc:
        return None, f"**Error loading model:** {exc}"


def reload_model():
    """Force reload after fine-tuning."""
    global _model
    _model = None
    return get_model()


# ── inference ─────────────────────────────────────────────────────────────────

def classify_detection(cls_id: int, conf: float) -> tuple[str, bool]:
    name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "Unknown"
    if name == UNKNOWN_CLASS_NAME or conf < CONFIDENCE_THRESHOLD:
        return "Unknown Herb", True
    return name, False


def run_inference(image_np: np.ndarray):
    """
    Returns
    -------
    annotated_np  : numpy image with all detection boxes/masks drawn
    detections    : list of unique-per-species dicts {herb, confidence, count, status}
    raw_boxes     : list of raw per-box dicts (for feedback annotation)
    has_unknown   : bool
    """
    model, _ = get_model()
    if model is None:
        return image_np, [], [], True

    # Enhance before inference: CLAHE → bilateral → adaptive gamma.
    # Gradio delivers RGB; OpenCV works in BGR — convert round-trip.
    from preprocess import enhance_image
    image_np = enhance_image(image_np[..., ::-1])[..., ::-1]

    results = model(image_np, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result  = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return image_np, [], [], True

    # ── raw box data (saved as annotation when user submits feedback) ─────────
    raw_boxes: list[dict] = []
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        xywhn  = box.xywhn[0].tolist()       # [cx, cy, w, h] normalised

        # Prefer the actual segmentation polygon if the model produced one
        poly = None
        if result.masks is not None and i < len(result.masks.xyn):
            arr = np.array(result.masks.xyn[i])
            if arr.size >= 6:
                poly = arr.flatten().tolist()

        raw_boxes.append({"cls_id": cls_id, "conf": conf,
                          "xywhn": xywhn, "poly": poly})

    # ── deduplicate by class (one entry per species, best confidence) ─────────
    per_class: dict[str, dict] = {}
    for box in raw_boxes:
        name, unknown = classify_detection(box["cls_id"], box["conf"])
        if name not in per_class:
            per_class[name] = {"herb": name, "confidence": box["conf"],
                               "count": 1,
                               "status": "Unknown" if unknown else "Identified"}
        else:
            per_class[name]["count"] += 1
            if box["conf"] > per_class[name]["confidence"]:
                per_class[name]["confidence"] = box["conf"]

    detections  = list(per_class.values())
    has_unknown = any(d["status"] == "Unknown" for d in detections)

    annotated_np = result.plot()[..., ::-1]   # BGR → RGB
    return annotated_np, detections, raw_boxes, has_unknown


# ── result markdown ───────────────────────────────────────────────────────────

def build_result_markdown(detections: list[dict], model_err: str | None) -> str:
    if model_err:
        return model_err
    if not detections:
        return (
            "### No herbs detected\n"
            "The model found no known herb.  \n"
            "This may be an **unknown species** — use the Expert Consultation form below."
        )

    known   = [d for d in detections if d["status"] == "Identified"]
    unknown = [d for d in detections if d["status"] == "Unknown"]
    lines   = []

    if len(known) > 1:
        lines.append(f"> 🌿 **{len(known)} different herb species detected**\n")

    if known:
        lines.append("### ✅ Identified Herbs")
        for d in known:
            reg = f" — {d['count']} regions in image" if d["count"] > 1 else ""
            lines.append(f"- **{d['herb']}**{reg}  \n"
                         f"  Best confidence: `{d['confidence']:.1%}`")

    if unknown:
        lines.append("\n### ❓ Unknown / Unidentified Herb")
        for d in unknown:
            lines.append(
                f"- **Unrecognised herb** — confidence too low "
                f"(`{d['confidence']:.1%}`)  \n"
                "  *Use the Expert Consultation form below.*"
            )

    return "\n".join(lines)


# ── Gradio event handlers ─────────────────────────────────────────────────────

def detect(image):
    """Main detection handler — returns outputs for all UI components."""
    import gradio as gr

    empty_state = {"image": None, "raw_boxes": [], "detected_names": []}

    if image is None:
        return (
            None,
            "Please provide an image using the camera or upload.",
            gr.update(visible=False),
            gr.update(choices=[], value=None, interactive=False),
            gr.update(visible=False),
            empty_state,
        )

    _, model_err = get_model()
    if model_err:
        return (image, model_err,
                gr.update(visible=True),
                gr.update(choices=[], value=None, interactive=False),
                gr.update(visible=False),
                empty_state)

    annotated, detections, raw_boxes, has_unknown = run_inference(image)
    result_md = build_result_markdown(detections, model_err)

    detected_names = [d["herb"] for d in detections if d["status"] == "Identified"]

    state = {"image": image, "raw_boxes": raw_boxes,
             "detected_names": detected_names}

    return (
        annotated,
        result_md,
        gr.update(visible=has_unknown),                         # expert_group
        gr.update(choices=detected_names,                       # wrong_herb_dd
                  value=detected_names[0] if detected_names else None,
                  interactive=bool(detected_names)),
        gr.update(visible=bool(detected_names)),                # feedback_group
        state,                                                  # detection_state
    )


def submit_correction(state: dict, wrong_herb: str, correct_herb: str) -> str:
    """Save the user's correction and update the feedback log."""
    from retrain import save_correction, get_feedback_stats

    if state is None or not state.get("raw_boxes"):
        return "⚠️ No detection state found. Please run detection first."
    if not wrong_herb:
        return "⚠️ Please select which herb was wrong."
    if not correct_herb:
        return "⚠️ Please select the correct herb."
    if wrong_herb == correct_herb:
        return "ℹ️ The selected herbs are the same — no correction needed."

    # Map display names to class indices
    wrong_idx   = CLASS_NAMES.index(wrong_herb) if wrong_herb in CLASS_NAMES else -1
    if correct_herb == "Unknown / New species":
        correct_idx = CLASS_NAMES.index(UNKNOWN_CLASS_NAME)
    else:
        correct_idx = CLASS_NAMES.index(correct_herb) if correct_herb in CLASS_NAMES else -1

    if wrong_idx < 0 or correct_idx < 0:
        return "⚠️ Could not resolve class indices. Please try again."

    ref_id = save_correction(
        image_np=state["image"],
        raw_boxes=state["raw_boxes"],
        wrong_cls_id=wrong_idx,
        correct_cls_id=correct_idx,
    )
    stats = get_feedback_stats()

    # ── auto-trigger background fine-tuning ───────────────────────────────────
    global _retrain_thread
    with _retrain_lock:
        if _retrain_thread is None or not _retrain_thread.is_alive():
            _retrain_thread = threading.Thread(
                target=_background_retrain, args=("cpu",), daemon=True
            )
            _retrain_thread.start()
            retrain_note = "🔄 Fine-tuning started in background — model will update automatically."
        else:
            retrain_note = "⏳ A fine-tuning run is already in progress — this correction will be included in the next one."

    return (
        f"### ✅ Correction saved — `{ref_id}`\n\n"
        f"**Wrong label:** {wrong_herb}  \n"
        f"**Correct label:** {correct_herb}\n\n"
        f"{retrain_note}  \n"
        f"Total pending corrections: **{stats['pending']}**"
    )


def trigger_retrain(device: str):
    """Fine-tune the model with all pending feedback. Runs synchronously."""
    from retrain import retrain_with_feedback, get_feedback_stats

    stats = get_feedback_stats()
    if stats["pending"] == 0:
        yield "ℹ️ No pending corrections to train on yet."
        return

    yield f"⏳ Fine-tuning on **{stats['pending']}** corrections …  *(this may take a few minutes)*"

    success = retrain_with_feedback(epochs=40, device=device)

    if success:
        reload_model()   # hot-reload the updated weights
        new_stats = get_feedback_stats()
        yield (
            "### ✅ Fine-tuning complete\n\n"
            f"Model updated at `{MODEL_PATH}`  \n"
            f"Total corrections merged: **{new_stats['merged']}**  \n\n"
            "*The model has been reloaded. Run a new detection to see the improvement.*"
        )
    else:
        yield "⚠️ Fine-tuning did not complete — check the terminal for errors."


def submit_expert_query(state, description: str, location: str, contact: str) -> str:
    if not description.strip():
        return "⚠️ Please describe the plant before submitting."
    queries = []
    if EXPERT_QUERIES_FILE.exists():
        with open(EXPERT_QUERIES_FILE) as f:
            queries = json.load(f)
    qid = len(queries) + 1
    ref = f"HRB-{qid:04d}"

    # Save image alongside the query so the expert can view it
    image_path = None
    if state and state.get("image") is not None:
        import cv2
        EXPERT_IMAGES_DIR.mkdir(exist_ok=True)
        img_file = EXPERT_IMAGES_DIR / f"{ref}.jpg"
        cv2.imwrite(str(img_file), state["image"][..., ::-1])   # RGB→BGR
        image_path = str(img_file)

    queries.append({
        "id":          qid,
        "ref":         ref,
        "timestamp":   datetime.now().isoformat(),
        "description": description.strip(),
        "location":    location.strip() or "Not provided",
        "contact":     contact.strip()  or "Not provided",
        "image_path":  image_path,
        "status":      "pending",
        "expert_label":  None,
        "expert_notes":  None,
        "reviewed_at":   None,
    })
    with open(EXPERT_QUERIES_FILE, "w") as f:
        json.dump(queries, f, indent=2)
    return (
        f"### ✅ Query submitted — `{ref}`\n\n"
        "An expert botanist will review your submission.  \n"
        "The plant has been logged as **UNKNOWN** pending identification."
    )


# ── expert review helpers ─────────────────────────────────────────────────────

def _load_queries() -> list[dict]:
    if EXPERT_QUERIES_FILE.exists():
        with open(EXPERT_QUERIES_FILE) as f:
            return json.load(f)
    return []


def _save_queries(queries: list[dict]):
    with open(EXPERT_QUERIES_FILE, "w") as f:
        json.dump(queries, f, indent=2)


def get_pending_choices() -> list[str]:
    """Return dropdown choices for pending queries."""
    return [
        f"{q['ref']} — {q['description'][:60]}{'…' if len(q['description']) > 60 else ''}"
        for q in _load_queries()
        if q.get("status") == "pending"
    ]


def load_query_for_review(choice: str):
    """Given a dropdown choice string, return (image_np_or_None, info_markdown)."""
    import gradio as gr
    if not choice:
        return None, "*Select a query above.*"

    ref = choice.split(" — ")[0].strip()
    queries = _load_queries()
    q = next((x for x in queries if x.get("ref") == ref), None)
    if q is None:
        return None, "⚠️ Query not found."

    info = (
        f"**Ref:** `{q['ref']}`  \n"
        f"**Submitted:** {q['timestamp'][:19].replace('T', ' ')}  \n"
        f"**Description:** {q['description']}  \n"
        f"**Location:** {q['location']}  \n"
        f"**Contact:** {q['contact']}  \n"
        f"**Status:** {q['status']}"
    )

    img_np = None
    if q.get("image_path") and Path(q["image_path"]).exists():
        import cv2
        bgr = cv2.imread(q["image_path"])
        if bgr is not None:
            img_np = bgr[..., ::-1]   # BGR→RGB for Gradio

    return img_np, info


def expert_label_query(choice: str, label: str, new_species: str, notes: str) -> str:
    """Save expert label, optionally add to retrain pipeline."""
    if not choice:
        return "⚠️ No query selected."
    if not label:
        return "⚠️ Please select a label."

    ref = choice.split(" — ")[0].strip()
    queries = _load_queries()
    q = next((x for x in queries if x.get("ref") == ref), None)
    if q is None:
        return "⚠️ Query not found."

    final_label = new_species.strip() if label == "New / Unknown Species" else label
    if not final_label:
        return "⚠️ Please enter a species name for 'New / Unknown Species'."

    q["status"]       = "labelled"
    q["expert_label"] = final_label
    q["expert_notes"] = notes.strip() or None
    q["reviewed_at"]  = datetime.now().isoformat()
    _save_queries(queries)

    # Feed into retrain pipeline when it's a known class
    retrain_note = ""
    if label != "New / Unknown Species" and label in CLASS_NAMES:
        cls_idx = CLASS_NAMES.index(label)
        img_path = q.get("image_path")
        if img_path and Path(img_path).exists():
            try:
                from retrain import save_expert_label_as_feedback
                fb_ref = save_expert_label_as_feedback(img_path, cls_idx, ref)
                retrain_note = (
                    f"\n\n🔄 Added to retraining queue as `{fb_ref}`. "
                    "Use **Force Retrain** in the main tab to update the model."
                )
            except Exception as exc:
                retrain_note = f"\n\n⚠️ Could not save to retrain queue: {exc}"
        else:
            retrain_note = "\n\n⚠️ No image was saved with this query — not added to retrain queue."

    return (
        f"### ✅ Query `{ref}` labelled\n\n"
        f"**Label assigned:** {final_label}  \n"
        f"**Notes:** {notes or '—'}"
        + retrain_note
    )


def expert_reject_query(choice: str, notes: str) -> str:
    """Mark a query as unresolvable."""
    if not choice:
        return "⚠️ No query selected."
    ref = choice.split(" — ")[0].strip()
    queries = _load_queries()
    q = next((x for x in queries if x.get("ref") == ref), None)
    if q is None:
        return "⚠️ Query not found."
    q["status"]      = "rejected"
    q["expert_notes"] = notes.strip() or None
    q["reviewed_at"] = datetime.now().isoformat()
    _save_queries(queries)
    return f"### ❌ Query `{ref}` marked as unresolvable."


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui():
    import gradio as gr

    _, model_err = get_model()
    from retrain import get_feedback_stats
    stats = get_feedback_stats()

    herb_list = ", ".join(n for n in CLASS_NAMES if n != UNKNOWN_CLASS_NAME)
    expert_label_choices = [n for n in CLASS_NAMES if n != UNKNOWN_CLASS_NAME] + ["New / Unknown Species"]

    _CSS = """
    /* ── hero: always white text, theme changes the background ─── */
    .hero {
        border-radius: 12px;
        padding: 1.4rem 1.2rem;
        margin-bottom: 0.6rem;
        transition: background 0.4s ease;
        /* default (first paint, before JS sets data-hs-theme) */
        background: linear-gradient(135deg, #122a1a 0%, #1c3a26 100%);
    }
    /* Force all text inside the hero to white — overrides Gradio defaults */
    .hero, .hero *, .hero h1, .hero h2, .hero h3, .hero p, .hero span,
    #title, #title *, #subtitle, #subtitle * {
        color: #ffffff !important;
    }
    /* Subtitle slightly softer */
    #subtitle, #subtitle * { color: rgba(255,255,255,0.80) !important; }

    #title    { text-align: center; }
    #subtitle { text-align: center; }

    /* ── per-theme hero backgrounds ───────────────────────────── */
    /* Dark — deep forest green */
    html[data-hs-theme="dark"] .hero {
        background: linear-gradient(135deg, #0d1f12 0%, #1c3a26 100%);
    }
    /* Light — vivid meadow (dark enough for white text) */
    html[data-hs-theme="light"] .hero {
        background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%);
    }
    /* Nature — warm bark / earth */
    html[data-hs-theme="nature"] .hero {
        background: linear-gradient(135deg, #4e342e 0%, #795548 100%);
    }
    /* Ocean — deep coastal */
    html[data-hs-theme="ocean"] .hero {
        background: linear-gradient(135deg, #01579b 0%, #0288d1 100%);
    }
    /* System — tracks OS; default dark unless OS is light */
    html[data-hs-theme="system"] .hero {
        background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%);
    }
    @media (prefers-color-scheme: dark) {
        html[data-hs-theme="system"] .hero {
            background: linear-gradient(135deg, #0d1f12 0%, #1c3a26 100%);
        }
    }

    /* ── theme-bar ────────────────────────────────────────────── */
    #theme-bar {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 6px;
        padding: 2px 4px 8px 4px;
        flex-wrap: wrap;
    }
    #theme-bar label { font-size: 0.78rem; opacity: 0.7; margin-right: 4px; }
    """

    # Gradio 5 supports ?__theme=dark in the URL to fully switch component colours.
    # On first visit we redirect to append it; on theme-switch we update it.
    _INIT_JS = """
    () => {
        const params  = new URLSearchParams(window.location.search);
        const urlMode = params.get('__theme');          // 'dark'|'light'|null

        /* Map our named themes to Gradio's dark/light URL param */
        const GRADIO_MODE = {
            Dark: 'dark', Light: 'light',
            Nature: 'light', Ocean: 'light', System: null,
        };

        /* Resolve which hs-theme to show */
        let saved = 'Dark';
        try { saved = localStorage.getItem('hs_theme') || 'Dark'; } catch(e) {}

        /* Apply data-hs-theme for our CSS */
        document.documentElement.setAttribute('data-hs-theme', saved.toLowerCase());

        /* If Gradio URL param doesn't match what we want, redirect once */
        const wantedMode = GRADIO_MODE[saved];
        if (wantedMode !== null && urlMode !== wantedMode) {
            const url = new URL(window.location);
            url.searchParams.set('__theme', wantedMode);
            window.location.replace(url.toString());
            return;
        }
        /* System theme: let Gradio follow OS if no __theme param */
        if (saved === 'System' && params.has('__theme')) {
            const url = new URL(window.location);
            url.searchParams.delete('__theme');
            window.location.replace(url.toString());
            return;
        }

        /* Register switcher used by the radio button */
        window._hsSetTheme = (name) => {
            try { localStorage.setItem('hs_theme', name); } catch(e) {}
            const mode = GRADIO_MODE[name];
            const url  = new URL(window.location);
            if (mode !== null) {
                url.searchParams.set('__theme', mode);
            } else {
                url.searchParams.delete('__theme');
            }
            window.location.replace(url.toString());
        };

        /* System media query listener */
        try {
            window.matchMedia('(prefers-color-scheme: dark)')
                  .addEventListener('change', () => {
                      if ((localStorage.getItem('hs_theme') || 'Dark') === 'System') {
                          window.location.reload();
                      }
                  });
        } catch(e) {}
    }
    """

    _THEME_JS = "(t) => { window._hsSetTheme && window._hsSetTheme(t); }"

    with gr.Blocks(
        title="Herbal Plant Recognition",
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald"),
        css=_CSS,
        js=_INIT_JS,
    ) as demo:

        # ── header ────────────────────────────────────────────────────────────
        with gr.Column(elem_classes="hero"):
            gr.Markdown("# 🌿 Herbal Plant Recognition System", elem_id="title")
            gr.Markdown(
                "Identify medicinal herbs from a live photo or uploaded image using AI",
                elem_id="subtitle",
            )
        with gr.Row(elem_id="theme-bar"):
            gr.Markdown("**Theme:**")
            theme_radio = gr.Radio(
                choices=["Dark", "Light", "Nature", "Ocean", "System"],
                value="Dark",
                show_label=False,
                container=False,
            )

        if model_err:
            gr.Markdown(f"> ⚠️ {model_err}")

        with gr.Tabs():
            # ══════════════════════════════════════════════════════════════════
            # Tab 1 — Detection
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🔍 Identify Herb"):

                # ── detection row ─────────────────────────────────────────────
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        gr.Markdown("### 📷 Input Image")
                        image_in = gr.Image(
                            label="Take a photo or upload",
                            sources=["webcam", "upload"],
                            type="numpy", height=380,
                        )
                        detect_btn = gr.Button("🔍  Identify Herb", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### 🌱 Detection Result")
                        image_out = gr.Image(
                            label="Annotated image", type="numpy",
                            height=300, interactive=False,
                        )
                        result_md = gr.Markdown("*Results will appear here after detection.*")

                # ── hidden state ──────────────────────────────────────────────
                detection_state = gr.State(value=None)

                # ── feedback section (visible after a detection) ──────────────
                with gr.Group(visible=False) as feedback_group:
                    gr.Markdown("---")
                    gr.Markdown(
                        "## ✏️ Correct a Wrong Prediction\n"
                        "If the model identified an herb incorrectly, use this form "
                        "to submit the correction.  Corrections are used to improve "
                        "the model via **continuous fine-tuning**."
                    )
                    with gr.Row():
                        with gr.Column(scale=2):
                            wrong_herb_dd = gr.Dropdown(
                                label="Which herb was wrongly identified?",
                                choices=[], value=None, interactive=False,
                            )
                            correct_herb_dd = gr.Dropdown(
                                label="What is the correct herb?",
                                choices=CORRECT_LABEL_CHOICES,
                                value=None, interactive=True,
                            )
                            correction_btn = gr.Button("💾  Submit Correction", variant="secondary")

                        with gr.Column(scale=2):
                            correction_result = gr.Markdown(
                                "*Select the wrongly-predicted herb and the correct one, "
                                "then click Submit Correction.*"
                            )

                    gr.Markdown("---")
                    gr.Markdown(
                        "## 🔄 Manual Retrain Override\n"
                        f"Corrections collected: **{stats['total']}** — "
                        f"already merged: **{stats['merged']}**\n\n"
                        "> Fine-tuning runs **automatically in the background** each time "
                        "you submit a correction above.  Use this button only if you want "
                        "to force an immediate retrain (e.g. after collecting several "
                        "corrections while offline)."
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            device_dd = gr.Dropdown(
                                label="Device", choices=["cpu", "0", "mps"],
                                value="cpu", interactive=True,
                            )
                            retrain_btn = gr.Button("🚀  Force Retrain Now", variant="secondary")
                        with gr.Column(scale=3):
                            retrain_result = gr.Markdown(
                                "*Retraining output will appear here.*"
                            )

                # ── expert consultation (unknown herbs only) ───────────────────
                with gr.Group(visible=False) as expert_group:
                    gr.Markdown("---")
                    gr.Markdown(
                        "## 🔬 Expert Consultation\n"
                        "This herb could not be identified. "
                        "Submit details for expert review."
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            desc_box = gr.Textbox(
                                label="Plant Description *",
                                placeholder="Leaf shape, colour, smell, stem, flower …",
                                lines=4,
                            )
                            with gr.Row():
                                loc_box     = gr.Textbox(label="Location (optional)",
                                                         placeholder="e.g. Kampala, Uganda")
                                contact_box = gr.Textbox(label="Contact Email (optional)",
                                                         placeholder="you@example.com")
                            expert_btn = gr.Button("📤  Submit to Expert", variant="secondary")
                        with gr.Column(scale=2):
                            expert_result = gr.Markdown(
                                "*Reference ID will appear here after submitting.*"
                            )

                # ── species list ──────────────────────────────────────────────
                with gr.Accordion("Supported herb species", open=False):
                    gr.Markdown(herb_list)

            # ══════════════════════════════════════════════════════════════════
            # Tab 2 — Expert Review
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🔬 Expert Review"):
                gr.Markdown(
                    "## Expert Review Panel\n"
                    "Review submitted unknown-plant queries, view the uploaded image, "
                    "and assign a species label. Labelling a known class automatically "
                    "queues the image for model retraining."
                )

                with gr.Row():
                    refresh_btn   = gr.Button("🔄 Refresh Pending Queries", variant="secondary")
                    pending_count = gr.Markdown("*Click Refresh to load.*")

                query_dd = gr.Dropdown(
                    label="Select a pending query",
                    choices=[], value=None, interactive=True,
                )

                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        review_img = gr.Image(
                            label="Submitted Image",
                            type="numpy", interactive=False, height=380,
                        )

                    with gr.Column(scale=1):
                        query_info_md = gr.Markdown("*Select a query above.*")

                        gr.Markdown("### Assign Label")
                        review_label_dd = gr.Dropdown(
                            label="Species",
                            choices=expert_label_choices,
                            value=None, interactive=True,
                        )
                        new_species_box = gr.Textbox(
                            label="New species name (fill in if selecting 'New / Unknown Species')",
                            placeholder="e.g. Vernonia amygdalina",
                            visible=False,
                        )
                        review_notes_box = gr.Textbox(
                            label="Expert Notes (optional)",
                            placeholder="Distinguishing features observed, confidence level …",
                            lines=3,
                        )
                        with gr.Row():
                            label_btn  = gr.Button("✅ Submit Label", variant="primary")
                            reject_btn = gr.Button("❌ Cannot Identify", variant="stop")
                        review_result = gr.Markdown()

        gr.Markdown(
            "---\n*Model: YOLOv8n-seg — Herbal Plants SpeciesInstSeg dataset (CC BY 4.0)*"
        )

        # ── event wiring ──────────────────────────────────────────────────────
        detect_btn.click(
            fn=detect,
            inputs=[image_in],
            outputs=[image_out, result_md, expert_group,
                     wrong_herb_dd, feedback_group, detection_state],
        )

        correction_btn.click(
            fn=submit_correction,
            inputs=[detection_state, wrong_herb_dd, correct_herb_dd],
            outputs=[correction_result],
        )

        retrain_btn.click(
            fn=trigger_retrain,
            inputs=[device_dd],
            outputs=[retrain_result],
        )

        expert_btn.click(
            fn=submit_expert_query,
            inputs=[detection_state, desc_box, loc_box, contact_box],
            outputs=[expert_result],
        )

        theme_radio.change(
            fn=None,
            inputs=[theme_radio],
            js=_THEME_JS,
        )

        # ── expert review wiring ───────────────────────────────────────────────
        def _refresh():
            choices = get_pending_choices()
            count   = f"**{len(choices)}** pending quer{'y' if len(choices)==1 else 'ies'}"
            return gr.update(choices=choices, value=None), count

        refresh_btn.click(
            fn=_refresh,
            outputs=[query_dd, pending_count],
        )

        query_dd.change(
            fn=load_query_for_review,
            inputs=[query_dd],
            outputs=[review_img, query_info_md],
        )

        def _toggle_new_species(label):
            return gr.update(visible=(label == "New / Unknown Species"))

        review_label_dd.change(
            fn=_toggle_new_species,
            inputs=[review_label_dd],
            outputs=[new_species_box],
        )

        def _submit_label(choice, label, new_species, notes):
            msg = expert_label_query(choice, label, new_species, notes)
            # Refresh the dropdown after labelling
            choices = get_pending_choices()
            return msg, gr.update(choices=choices, value=None), None, "*Select a query above.*"

        label_btn.click(
            fn=_submit_label,
            inputs=[query_dd, review_label_dd, new_species_box, review_notes_box],
            outputs=[review_result, query_dd, review_img, query_info_md],
        )

        def _reject(choice, notes):
            msg = expert_reject_query(choice, notes)
            choices = get_pending_choices()
            return msg, gr.update(choices=choices, value=None), None, "*Select a query above.*"

        reject_btn.click(
            fn=_reject,
            inputs=[query_dd, review_notes_box],
            outputs=[review_result, query_dd, review_img, query_info_md],
        )

    return demo


# ── entry point ───────────────────────────────────────────────────────────────

def _disable_brotli_middleware() -> None:
    """
    Replace Gradio's BrotliMiddleware with a no-op pass-through.

    Gradio ≥ 5 adds a BrotliMiddleware that occasionally sends fewer bytes than
    its declared Content-Length, causing h11 to raise
    'LocalProtocolError: Too little data for declared Content-Length'.
    Patching the name in gradio.routes (where add_middleware reads it) before
    demo.launch() is called prevents the middleware from ever being installed.
    """
    try:
        import gradio.routes as _routes  # type: ignore[import]

        class _PassThrough:
            def __init__(self, app, **_):
                self.app = app
            async def __call__(self, scope, receive, send):
                await self.app(scope, receive, send)

        _routes.BrotliMiddleware = _PassThrough
        print("[INFO] BrotliMiddleware disabled (h11 content-length workaround).")
    except Exception as exc:
        print(f"[WARN] Could not patch BrotliMiddleware: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    _disable_brotli_middleware()
    demo = build_ui()
    demo.queue()          # required for generator-based retrain progress
    demo.launch(server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
