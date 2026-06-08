# HerbScan — Medicinal Herbal Plant Recognition

A real-time medicinal herb identification system using **YOLOv8 instance segmentation**, served through a browser-based Gradio interface. The system identifies 16 East African medicinal herb species, handles unknown plants with an expert consultation pathway, and continuously improves via a background fine-tuning loop triggered by user corrections.

**GitHub:** https://github.com/bertonag/HerbalSpiciesRecognition

---

## Features

- **Instance segmentation** — YOLOv8n-seg draws pixel-accurate masks around each detected herb
- **Camera & upload support** — use your device camera or upload a JPEG/PNG image
- **Per-class deduplication** — each detected species is shown once with its highest confidence score and a count of how many instances were found
- **Unknown herb detection** — detections below confidence 0.35 or labelled as the *New* class trigger an expert consultation form with a unique reference ID
- **Feedback & correction** — select a wrong prediction and the correct species; a corrected YOLO label is saved automatically
- **Background fine-tuning** — submitting a correction immediately starts a daemon thread that fine-tunes the model without blocking the UI, then hot-reloads it
- **Full evaluation suite** — per-class metrics, confusion matrix, F1 curve, and annotated prediction images

---

## Herb Classes (16)

| # | Species | Local Name |
|---|---------|------------|
| 0 | Bitter Leaf | Mululuza |
| 1 | Boerhavia diffusa | Olweza |
| 2 | Ceratonia siliqua | Omutulika |
| 3 | Chenopodium album | Omwetango |
| 4 | False Daisy | Mutaayiza |
| 5 | Himalayan Balsam | Muzukizi |
| 6 | Hoslundia opposita | Kamunye |
| 7 | Justicia pectoralis | — |
| 8 | Leucaena leucocephala | Lusina |
| 9 | Momordica foetida | Ebombo |
| 10 | Perilla frutescens | — |
| 11 | Plectranthus prostratus | Mubiri |
| 12 | Plectrarithus cyaneus | Kibwankulata |
| 13 | Sugar cane | Kikajo |
| 14 | Peppermint | — |
| 15 | New | (unknown / unannotated) |

---

## Project Structure

```
HerbalRecognition/
├── train.py                  # Dataset extraction, annotation conversion, YOLOv8 training
├── app.py                    # Gradio web application
├── retrain.py                # Feedback storage and incremental fine-tuning
├── evaluate.py               # Validation metrics, confusion matrix, F1 curve
├── requirements.txt          # Python dependencies
├── .gitignore
│
├── Herbal Plants SpeciesInstSeg.yolov8.zip   # Source dataset (not pushed)
├── dataset/                  # Extracted dataset (train/valid/test splits) — git-ignored
├── runs/train/weights/
│   ├── best.pt               # Best model checkpoint — git-ignored
│   └── last.pt               # Last model checkpoint — git-ignored
└── feedback/                 # User corrections and fine-tuning data — git-ignored
    ├── images/
    ├── labels/
    └── feedback_log.json
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/bertonag/HerbalSpiciesRecognition.git
cd HerbalSpiciesRecognition
```

### 2. Create a virtual environment

```bash
python -m venv HerbalVenv

# Windows
HerbalVenv\Scripts\activate

# macOS / Linux
source HerbalVenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the dataset zip

Download or copy `Herbal Plants SpeciesInstSeg.yolov8.zip` into the project root. `train.py` extracts it automatically.

### 5. Train the model

```bash
python train.py
```

Default: 150 epochs, early stopping (patience=40), YOLOv8n-seg, CPU. Options:

```bash
python train.py --epochs 150 --batch 8 --device 0   # GPU (CUDA device 0)
```

Training saves the best weights to `runs/train/weights/best.pt`.

### 6. Launch the web app

```bash
python app.py
```

Open http://localhost:7860 in your browser.

---

## Usage — Web Interface

| Step | Action |
|------|--------|
| **Detect** | Click *Camera* or *Upload Image*, then press **Detect Herbs** |
| **Results** | Each detected species appears once with confidence score and instance count |
| **Unknown herb** | If a herb is flagged as unknown, fill in the *Expert Consultation* form and submit |
| **Correction** | If the model is wrong, select the incorrectly identified herb, choose the correct one, and click **Submit Correction** — fine-tuning starts automatically in the background |

---

## Evaluation

```bash
# Evaluate on validation split (default)
python evaluate.py

# Evaluate on test split and save annotated prediction images
python evaluate.py --split test --save
```

Outputs saved to `runs/train/`:

- `confusion_matrix_eval.png`
- `f1_curve.png`
- `sample_predictions/` (if `--save` is used)

### Validation Results

| Metric | Box | Mask |
|--------|-----|------|
| mAP50 | 0.465 | 0.418 |
| mAP50-95 | 0.313 | 0.259 |
| Precision | 0.451 | 0.403 |
| Recall | 0.480 | 0.460 |
| F1 | 0.465 | 0.430 |

Best classes: **False Daisy** and **Perilla frutescens** (mAP50 ≈ 0.995 each).

---

## Continuous Learning

When a user submits a correction via the web interface:

1. The original image and a corrected YOLO-format polygon label are saved to `feedback/`
2. A background daemon thread starts `retrain_with_feedback()` automatically
3. The model fine-tunes from `best.pt` using a low learning rate (`lr=0.0005`) for 40 epochs — this prevents catastrophic forgetting of original training data
4. After completion the model is hot-reloaded in the running application

To fine-tune manually:

```bash
python retrain.py
python retrain.py --epochs 40 --device 0   # GPU
```

---

## Dataset Notes

All annotations are polygon segmentation format (YOLO instance segmentation). The dataset has been fully re-annotated; no bounding-box-only labels remain.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `ultralytics>=8.0.0` | YOLOv8 training and inference |
| `gradio>=4.44.0,<6.0.0` | Web interface (pinned below 6.x to avoid a BrotliMiddleware/h11 bug) |
| `opencv-python>=4.8.0` | Image processing |
| `Pillow>=10.0.0` | Image I/O |
| `numpy>=1.24.0` | Numerical operations |
| `PyYAML>=6.0` | Dataset config parsing |
| `pandas>=2.0.0` | Metrics tables |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4-core, 2 GHz | Intel Core i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB |
| GPU | — (CPU-only works) | NVIDIA GPU with 4+ GB VRAM |
| Disk | 2 GB free | 5 GB free |

Average inference time on CPU: ~89 ms per 640×640 image.

---

## License

This project is released for academic and research use.  
Dataset sourced from [Roboflow Universe](https://universe.roboflow.com).

---

*Gilbert Nyakana — Makerere University, Kampala, Uganda — 2026*
