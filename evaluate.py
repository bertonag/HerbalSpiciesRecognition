"""
Herbal Plant Recognition - Model Evaluation
============================================
Generates a comprehensive evaluation report including:
  - Per-class Precision, Recall, F1, mAP50, mAP50-95
  - Confusion matrix (saved as PNG + printed to console)
  - Test-set predictions with annotated images
  - Inference speed benchmark

Usage:
    python evaluate.py                        # evaluate on validation split
    python evaluate.py --split test           # evaluate on test split
    python evaluate.py --split test --save    # also save prediction images
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

BASE_DIR  = Path(__file__).parent
YAML_PATH = BASE_DIR / "dataset" / "data.yaml"
BEST_PT   = BASE_DIR / "runs" / "train" / "weights" / "best.pt"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_class_names() -> list[str]:
    with open(YAML_PATH) as f:
        return yaml.safe_load(f)["names"]


def find_model() -> Path:
    if BEST_PT.exists():
        return BEST_PT
    # fallback: search any best.pt under runs/
    candidates = sorted((BASE_DIR / "runs").rglob("best.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "No trained model found. Run python train.py first."
    )


# ── metrics table ─────────────────────────────────────────────────────────────

def print_metrics_table(metrics, class_names: list[str]):
    """Pretty-print per-class metrics from a YOLO Results object."""
    print("\n" + "=" * 80)
    print(" PER-CLASS DETECTION METRICS")
    print("=" * 80)

    header = f"{'Class':<38} {'Images':>7} {'Instances':>10} {'P':>7} {'R':>7} {'F1':>7} {'mAP50':>8} {'mAP50-95':>10}"
    print(header)
    print("-" * 80)

    box = metrics.box
    # box.ap_class_index: class indices that appeared in validation
    for i, cls_idx in enumerate(box.ap_class_index):
        name  = class_names[cls_idx] if cls_idx < len(class_names) else f"cls{cls_idx}"
        p     = box.p[i]
        r     = box.r[i]
        f1    = 2 * p * r / (p + r + 1e-9)
        ap50  = box.ap50[i]
        ap    = box.ap[i]  # mAP50-95
        # images/instances not directly available per-class in all versions
        print(f"{name:<38} {'—':>7} {'—':>10} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {ap50:>8.3f} {ap:>10.3f}")

    print("-" * 80)
    mp  = float(box.mp)
    mr  = float(box.mr)
    mf1 = 2 * mp * mr / (mp + mr + 1e-9)
    print(f"{'ALL (mean)':<38} {'':>7} {'':>10} {mp:>7.3f} {mr:>7.3f} {mf1:>7.3f} "
          f"{float(box.map50):>8.3f} {float(box.map):>10.3f}")
    print("=" * 80)


# ── confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(metrics, class_names: list[str], save_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("[SKIP] matplotlib not installed — confusion matrix skipped.")
        return

    # YOLO stores the raw confusion matrix in metrics.confusion_matrix.matrix
    cm_obj = metrics.confusion_matrix
    if cm_obj is None:
        print("[INFO] Confusion matrix not available for this split.")
        return

    matrix = cm_obj.matrix.astype(int)   # shape: (nc+1, nc+1) incl. background
    nc = len(class_names)

    # Trim to (nc, nc) – drop background row/col if present
    if matrix.shape[0] == nc + 1:
        matrix = matrix[:nc, :nc]

    # Normalise rows → recall per class
    row_sums = matrix.sum(axis=1, keepdims=True).clip(min=1)
    norm = matrix / row_sums

    fig, ax = plt.subplots(figsize=(max(10, nc), max(8, nc - 2)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax.set_xticks(range(nc))
    ax.set_yticks(range(nc))
    labels = [n[:20] for n in class_names]   # truncate long names
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True (Actual)", fontsize=11)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=13)

    # Annotate cells
    for i in range(nc):
        for j in range(nc):
            colour = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color=colour)

    plt.tight_layout()
    out_path = save_dir / "confusion_matrix_eval.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved: {out_path}")


# ── F1-confidence curve ───────────────────────────────────────────────────────

def plot_f1_curve(metrics, class_names: list[str], save_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    box = metrics.box
    # px = confidence thresholds, py = F1 values (nc, len(px))
    if not hasattr(box, "curves_results"):
        print("[INFO] F1-curve data not available in this ultralytics version.")
        return

    # curves_results entries are (px, py, label) — py may be 2-D [nc, len(px)]
    entry = box.curves_results[2]   # index 2 is F1
    conf  = entry[0]
    py    = entry[1]
    import numpy as _np
    f1_mean = _np.array(py).mean(axis=0) if _np.array(py).ndim == 2 else _np.array(py)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(conf, f1_mean, color="tab:green", linewidth=2, label="All classes (mean)")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("F1")
    ax.set_title("F1-Confidence Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = save_dir / "f1_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] F1 curve saved: {out_path}")


# ── prediction samples ────────────────────────────────────────────────────────

def save_prediction_samples(model, split: str, save_dir: Path, n: int = 8):
    """Run inference on a few images and save annotated outputs."""
    from PIL import Image as PILImage
    import shutil

    split_dir = BASE_DIR / "dataset" / split / "images"
    if not split_dir.exists():
        print(f"[SKIP] Image folder not found: {split_dir}")
        return

    images = sorted(split_dir.glob("*.jpg"))[:n]
    pred_dir = save_dir / "sample_predictions"
    pred_dir.mkdir(exist_ok=True)

    print(f"[INFO] Saving {len(images)} annotated prediction samples …")
    for img_path in images:
        results = model(str(img_path), conf=0.25, verbose=False)
        annotated = results[0].plot()   # BGR numpy
        from PIL import Image
        import cv2
        out = pred_dir / img_path.name
        cv2.imwrite(str(out), annotated)
    print(f"[INFO] Samples saved to: {pred_dir}")


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate(split: str = "val", save_samples: bool = False):
    from ultralytics import YOLO

    model_path = find_model()
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))

    class_names = load_class_names()
    save_dir = model_path.parent.parent  # runs/train/

    print(f"[INFO] Evaluating on '{split}' split …\n")
    metrics = model.val(
        data=str(YAML_PATH),
        split=split,
        verbose=True,
        plots=True,           # saves PR curve, confusion matrix, etc. to save_dir
        save_dir=str(save_dir),
    )

    # ── printed table ────────────────────────────────────────────────────────
    print_metrics_table(metrics, class_names)

    # ── overall summary ──────────────────────────────────────────────────────
    box = metrics.box
    mp  = float(box.mp)
    mr  = float(box.mr)
    mf1 = 2 * mp * mr / (mp + mr + 1e-9)

    print("\n OVERALL SUMMARY")
    print(f"  mAP50       : {float(box.map50):.4f}")
    print(f"  mAP50-95    : {float(box.map):.4f}")
    print(f"  mean Precision : {mp:.4f}")
    print(f"  mean Recall    : {mr:.4f}")
    print(f"  mean F1        : {mf1:.4f}")
    print(f"\n  Results & plots saved to: {save_dir}\n")

    # ── extra plots ───────────────────────────────────────────────────────────
    plot_confusion_matrix(metrics, class_names, save_dir)
    plot_f1_curve(metrics, class_names, save_dir)

    if save_samples:
        save_prediction_samples(model, split, save_dir)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate herbal plant model")
    parser.add_argument("--split",  default="val", choices=["val", "test"],
                        help="Dataset split to evaluate on (default: val)")
    parser.add_argument("--save",   action="store_true",
                        help="Save annotated prediction images")
    args = parser.parse_args()

    evaluate(split=args.split, save_samples=args.save)
