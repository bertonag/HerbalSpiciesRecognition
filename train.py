"""
Herbal Plant Recognition - Training Script
==========================================
Steps:
  1. Extract dataset from zip
  2. Convert any bbox-only annotations to 4-corner polygons
     so ALL 307 annotations are usable by the segmentation model
  3. Fix data.yaml paths
  4. Train YOLOv8n-seg with transfer learning

Usage:
    python train.py
    python train.py --epochs 150 --batch 4 --device 0   # GPU
"""
import argparse
import zipfile
from pathlib import Path

import yaml

BASE_DIR    = Path(__file__).parent
ZIP_FILE    = BASE_DIR / "Herbal Plants SpeciesInstSeg.yolov8.zip"
DATASET_DIR = BASE_DIR / "dataset"
YAML_PATH   = DATASET_DIR / "data.yaml"
BEST_PT     = BASE_DIR / "runs" / "train" / "weights" / "best.pt"


# ── dataset helpers ──────────────────────────────────────────────────────────

def extract_dataset():
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        print(f"[INFO] Dataset already extracted: {DATASET_DIR}")
        return
    if not ZIP_FILE.exists():
        raise FileNotFoundError(f"Zip not found: {ZIP_FILE}")
    print("[INFO] Extracting dataset …")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_FILE, "r") as z:
        z.extractall(DATASET_DIR)
    print(f"[INFO] Extracted to: {DATASET_DIR}")


def fix_yaml_paths():
    """Write absolute paths into data.yaml."""
    with open(YAML_PATH) as f:
        data = yaml.safe_load(f)
    data["train"] = str(DATASET_DIR / "train" / "images")
    data["val"]   = str(DATASET_DIR / "valid" / "images")
    data["test"]  = str(DATASET_DIR / "test"  / "images")
    with open(YAML_PATH, "w") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    print(f"[INFO] data.yaml updated ({data['nc']} classes).")
    return data


# ── annotation conversion ────────────────────────────────────────────────────

def bbox_to_polygon(cx: float, cy: float, w: float, h: float) -> list[float]:
    """
    Convert a YOLOv8 bounding box (cx cy w h, normalised) to a 4-corner
    polygon (x1 y1 x2 y2 x3 y3 x4 y4, normalised, clockwise from top-left).
    Coordinates are clamped to [0, 1].
    """
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy - h / 2
    x3, y3 = cx + w / 2, cy + h / 2
    x4, y4 = cx - w / 2, cy + h / 2
    pts = [x1, y1, x2, y2, x3, y3, x4, y4]
    return [max(0.0, min(1.0, v)) for v in pts]


def convert_label_file(path: Path) -> int:
    """
    Rewrite a label file so that bbox-only lines (5 tokens) become
    4-corner polygon lines (9 tokens).  Returns the number of lines converted.
    """
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    converted, n_conv = [], 0
    for line in lines:
        tokens = line.strip().split()
        if not tokens:
            continue
        if len(tokens) == 5:           # bbox format: class cx cy w h
            cls_id = tokens[0]
            cx, cy, w, h = map(float, tokens[1:])
            poly = bbox_to_polygon(cx, cy, w, h)
            converted.append(cls_id + " " + " ".join(f"{v:.6f}" for v in poly))
            n_conv += 1
        else:                           # already a polygon – keep as-is
            converted.append(line)
    path.write_text("\n".join(converted) + "\n", encoding="utf-8")
    return n_conv


def convert_all_labels():
    """Convert every bbox-only annotation in the dataset to polygon format."""
    label_files = list(DATASET_DIR.rglob("labels/*.txt"))
    total_conv = 0
    for lf in label_files:
        total_conv += convert_label_file(lf)
    print(f"[INFO] Converted {total_conv} bbox annotations → 4-corner polygons "
          f"across {len(label_files)} label files.")


# ── training ─────────────────────────────────────────────────────────────────

def train(epochs: int, batch: int, device: str, resume: bool):
    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The ultralytics package is required to train the model. "
            "Install it with: pip install ultralytics"
        ) from exc

    if BEST_PT.exists() and not resume:
        print(f"[INFO] Model already trained: {BEST_PT}")
        print("       Use --resume to continue, or delete runs/ to retrain.")
        return BEST_PT

    extract_dataset()
    convert_all_labels()   # bbox → polygon so seg model sees ALL annotations
    fix_yaml_paths()

    print(f"\n[INFO] Training YOLOv8n-seg for up to {epochs} epochs …")
    print(f"       batch={batch}  device={device}\n")

    model = YOLO("yolov8n-seg.pt")   # nano instance-segmentation, COCO pre-trained

    model.train(
        data=str(YAML_PATH),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        patience=40,         # slightly more patience for small dataset
        device=device,
        project=str(BASE_DIR / "runs"),
        name="train",
        exist_ok=True,
        resume=resume,
        # ── augmentation (critical for 74-image dataset) ──────────────────
        hsv_h=0.02,          # hue shift  – herb colour varies by season/lighting
        hsv_s=0.75,          # saturation – handle over/under-saturated photos
        hsv_v=0.4,           # value      – handle different lighting conditions
        flipud=0.3,          # vertical flip – plants photographed upside-down
        fliplr=0.5,          # horizontal flip
        mosaic=1.0,          # mosaic augmentation – most impactful on small sets
        mixup=0.15,          # mix two images – improves generalisation
        copy_paste=0.2,      # copy-paste from other images – helps rare classes
        degrees=15.0,        # rotation – camera tilt
        translate=0.1,       # slight translation
        scale=0.6,           # scale jitter
        shear=5.0,           # shear – perspective distortion
    )

    if BEST_PT.exists():
        print(f"\n[SUCCESS] Best model saved at: {BEST_PT}")
    else:
        print("\n[WARNING] best.pt not found after training – check logs above.")

    return BEST_PT


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train herbal plant recognition model")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Max training epochs (default 150; early-stop kicks in)")
    parser.add_argument("--batch",  type=int, default=8,
                        help="Batch size (default 8; use 4 on low-RAM machines)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu | 0 (first GPU) | mps (Apple Silicon)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    train(epochs=args.epochs, batch=args.batch,
          device=args.device, resume=args.resume)
    print("\nNext: python app.py")
