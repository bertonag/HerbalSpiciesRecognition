"""
Herbal Plant Recognition - Feedback & Incremental Learning
==========================================================
Handles two responsibilities:
  1. Saving user corrections (wrong prediction → correct label + image)
  2. Fine-tuning the existing model on the accumulated feedback data

Feedback layout on disk
-----------------------
feedback/
  images/          <-- original (un-annotated) images
  labels/          <-- corrected YOLO-seg polygon annotations
  feedback_log.json

Usage (standalone)
------------------
    python retrain.py              # fine-tune with all pending feedback
    python retrain.py --epochs 40  # custom epoch count
    python retrain.py --device 0   # use first GPU
"""

import argparse
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

BASE_DIR     = Path(__file__).parent
FEEDBACK_DIR = BASE_DIR / "feedback"
FEEDBACK_LOG = FEEDBACK_DIR / "feedback_log.json"
DATASET_DIR  = BASE_DIR / "dataset"
YAML_PATH    = DATASET_DIR / "data.yaml"
BEST_PT      = BASE_DIR / "runs" / "train" / "weights" / "best.pt"


# ── helpers ──────────────────────────────────────────────────────────────────

def _ensure_dirs():
    (FEEDBACK_DIR / "images").mkdir(parents=True, exist_ok=True)
    (FEEDBACK_DIR / "labels").mkdir(parents=True, exist_ok=True)


def load_feedback_log() -> list[dict]:
    if FEEDBACK_LOG.exists():
        with open(FEEDBACK_LOG) as f:
            return json.load(f)
    return []


def save_feedback_log(entries: list[dict]):
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_LOG, "w") as f:
        json.dump(entries, f, indent=2)


def get_feedback_stats() -> dict:
    entries   = load_feedback_log()
    merged    = [e for e in entries if e.get("merged")]
    pending   = [e for e in entries if not e.get("merged")]
    return {"total": len(entries), "pending": len(pending), "merged": len(merged)}


def load_class_names() -> list[str]:
    if YAML_PATH.exists():
        with open(YAML_PATH) as f:
            return yaml.safe_load(f)["names"]
    return []


def _bbox_to_polygon(cx: float, cy: float, w: float, h: float) -> list[float]:
    """Bounding box → 4-corner polygon (normalised, clamped to [0,1])."""
    pts = [
        cx - w / 2, cy - h / 2,
        cx + w / 2, cy - h / 2,
        cx + w / 2, cy + h / 2,
        cx - w / 2, cy + h / 2,
    ]
    return [max(0.0, min(1.0, v)) for v in pts]


# ── feedback saving ───────────────────────────────────────────────────────────

def save_correction(
    image_np: np.ndarray,
    raw_boxes: list[dict],
    wrong_cls_id: int,
    correct_cls_id: int,
) -> str:
    """
    Persist one user correction.

    Parameters
    ----------
    image_np      : original numpy image (H×W×3 RGB)
    raw_boxes     : list of dicts from detection state:
                    {cls_id, conf, xywhn: [cx,cy,w,h], poly: [x,y,...] or None}
    wrong_cls_id  : the class the model predicted incorrectly
    correct_cls_id: the class the user says it actually is

    Returns
    -------
    Reference ID string, e.g. "FB-0007"
    """
    _ensure_dirs()
    import cv2

    # Build corrected annotation lines
    label_lines: list[str] = []
    for box in raw_boxes:
        cls_id = box["cls_id"]
        # Apply the correction: if this box was the wrong class, fix it
        if cls_id == wrong_cls_id:
            cls_id = correct_cls_id

        poly = box.get("poly")
        if poly and len(poly) >= 6:
            pts = poly
        else:
            cx, cy, w, h = box["xywhn"]
            pts = _bbox_to_polygon(cx, cy, w, h)

        label_lines.append(str(cls_id) + " " + " ".join(f"{v:.6f}" for v in pts))

    # Unique filename based on UUID
    uid      = uuid.uuid4().hex[:10]
    img_name = f"fb_{uid}.jpg"
    lbl_name = f"fb_{uid}.txt"

    # Save image (RGB → BGR for cv2)
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(FEEDBACK_DIR / "images" / img_name), img_bgr)

    # Save corrected label
    (FEEDBACK_DIR / "labels" / lbl_name).write_text(
        "\n".join(label_lines) + "\n", encoding="utf-8"
    )

    # Append to log
    log = load_feedback_log()
    ref_id = f"FB-{len(log) + 1:04d}"
    class_names = load_class_names()
    log.append({
        "ref_id":       ref_id,
        "timestamp":    datetime.now().isoformat(),
        "image_file":   img_name,
        "label_file":   lbl_name,
        "wrong_class":  class_names[wrong_cls_id]  if wrong_cls_id  < len(class_names) else str(wrong_cls_id),
        "correct_class":class_names[correct_cls_id] if correct_cls_id < len(class_names) else str(correct_cls_id),
        "merged":       False,
    })
    save_feedback_log(log)
    print(f"[FEEDBACK] Saved correction {ref_id}: "
          f"'{class_names[wrong_cls_id]}' → '{class_names[correct_cls_id]}'")
    return ref_id


# ── merging feedback into training split ──────────────────────────────────────

def _merge_pending_feedback() -> int:
    """
    Copy un-merged feedback images + labels into dataset/train/.
    Marks each entry as merged in the log.
    Returns number of items copied.
    """
    log   = load_feedback_log()
    pending = [e for e in log if not e.get("merged")]
    if not pending:
        return 0

    train_img = DATASET_DIR / "train" / "images"
    train_lbl = DATASET_DIR / "train" / "labels"

    copied = 0
    for entry in pending:
        src_img = FEEDBACK_DIR / "images" / entry["image_file"]
        src_lbl = FEEDBACK_DIR / "labels" / entry["label_file"]
        dst_img = train_img / entry["image_file"]
        dst_lbl = train_lbl / entry["label_file"]

        if src_img.exists() and not dst_img.exists():
            shutil.copy(src_img, dst_img)
        if src_lbl.exists() and not dst_lbl.exists():
            shutil.copy(src_lbl, dst_lbl)

        entry["merged"] = True
        copied += 1

    # Invalidate YOLO label caches so it rescans
    for cache in DATASET_DIR.rglob("labels.cache"):
        cache.unlink(missing_ok=True)

    save_feedback_log(log)
    print(f"[INFO] Merged {copied} feedback samples into training set.")
    return copied


# ── fine-tuning ───────────────────────────────────────────────────────────────

def retrain_with_feedback(epochs: int = 40, device: str = "cpu") -> bool:
    """
    Merge pending feedback into the training split, then fine-tune the current
    best model.  Returns True on success.

    Why fine-tune rather than train from scratch?
    - Only a handful of corrections exist → training from scratch would
      over-emphasise feedback and forget original knowledge (catastrophic forgetting).
    - Fine-tuning with a low learning rate (lr0=0.0005) nudges the model toward
      the corrections without disrupting the weights learned on the full dataset.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    stats = get_feedback_stats()
    if stats["pending"] == 0:
        print("[INFO] No pending feedback to train on.")
        return False

    n = _merge_pending_feedback()
    if n == 0:
        print("[INFO] Nothing new was merged.")
        return False

    if not BEST_PT.exists():
        print("[ERROR] No trained model found. Run python train.py first.")
        return False

    # Ensure yaml paths are correct
    with open(YAML_PATH) as f:
        data = yaml.safe_load(f)
    data["train"] = str(DATASET_DIR / "train" / "images")
    data["val"]   = str(DATASET_DIR / "valid" / "images")
    data["test"]  = str(DATASET_DIR / "test"  / "images")
    with open(YAML_PATH, "w") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    print(f"\n[INFO] Fine-tuning on {n} new corrections for {epochs} epochs …")
    print(f"       Starting from: {BEST_PT}\n")

    model = YOLO(str(BEST_PT))   # resume from existing best weights
    model.train(
        data=str(YAML_PATH),
        epochs=epochs,
        imgsz=640,
        batch=4,
        lr0=0.0005,       # low LR — prevents catastrophic forgetting
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=15,
        device=device,
        project=str(BASE_DIR / "runs"),
        name="train",
        exist_ok=True,
        # Keep augmentation light for fine-tuning
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.3,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
    )

    print(f"\n[SUCCESS] Fine-tuning complete. Updated model at: {BEST_PT}")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain with feedback data")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    stats = get_feedback_stats()
    print(f"Feedback stats: {stats['total']} total, "
          f"{stats['pending']} pending, {stats['merged']} already merged")

    retrain_with_feedback(epochs=args.epochs, device=args.device)
