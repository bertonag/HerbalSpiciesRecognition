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


def submit_expert_query(description: str, location: str, contact: str) -> str:
    if not description.strip():
        return "⚠️ Please describe the plant before submitting."
    queries  = []
    if EXPERT_QUERIES_FILE.exists():
        with open(EXPERT_QUERIES_FILE) as f:
            queries = json.load(f)
    qid = len(queries) + 1
    queries.append({
        "id": qid, "timestamp": datetime.now().isoformat(),
        "description": description.strip(),
        "location": location.strip() or "Not provided",
        "contact":  contact.strip()  or "Not provided",
        "status": "pending_expert_review",
    })
    with open(EXPERT_QUERIES_FILE, "w") as f:
        json.dump(queries, f, indent=2)
    return (
        f"### ✅ Query submitted — `HRB-{qid:04d}`\n\n"
        "An expert botanist will review your submission.  \n"
        "The plant has been logged as **UNKNOWN** pending identification."
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui():
    import gradio as gr

    _, model_err = get_model()
    from retrain import get_feedback_stats
    stats = get_feedback_stats()

    herb_list = ", ".join(n for n in CLASS_NAMES if n != UNKNOWN_CLASS_NAME)

    with gr.Blocks(
        title="Herbal Plant Recognition",
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald"),
        css="""
        #title    { text-align:center; }
        #subtitle { text-align:center; color:#555; }
        .hero { background:linear-gradient(135deg,#e8f5e9,#f1f8e9);
                border-radius:12px; padding:1.2rem 1rem; margin-bottom:0.5rem; }
        """,
    ) as demo:

        # ── header ────────────────────────────────────────────────────────────
        with gr.Column(elem_classes="hero"):
            gr.Markdown("# 🌿 Herbal Plant Recognition System", elem_id="title")
            gr.Markdown(
                "Identify medicinal herbs from a live photo or uploaded image using AI",
                elem_id="subtitle",
            )

        if model_err:
            gr.Markdown(f"> ⚠️ {model_err}")

        # ── detection row ─────────────────────────────────────────────────────
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

        # ── hidden state ──────────────────────────────────────────────────────
        detection_state = gr.State(value=None)

        # ── feedback section (visible after a successful detection) ───────────
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

        # ── expert consultation (unknown herbs only) ───────────────────────────
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

        # ── species list ──────────────────────────────────────────────────────
        with gr.Accordion("Supported herb species", open=False):
            gr.Markdown(herb_list)

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
            inputs=[desc_box, loc_box, contact_box],
            outputs=[expert_result],
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
