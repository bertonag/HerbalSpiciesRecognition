"""
Image enhancement pipeline for HerbScan.

Applies CLAHE → Bilateral filter → Adaptive gamma correction to improve
feature quality in field-captured herbal plant images.

Can be used two ways:
  1. Offline (train.py) — preprocess every image in the dataset once before
     training, writing enhanced images in-place.  A sentinel file prevents
     re-processing on subsequent runs.

  2. Real-time (app.py) — enhance each uploaded image immediately before
     passing it to the YOLOv8 inference call.

Enhancement order (matches Section 4.5 of the CVPR report):
    CLAHE  →  Bilateral filter  →  Adaptive gamma
"""
import cv2
import numpy as np
from pathlib import Path

# ── tuneable parameters ───────────────────────────────────────────────────────
CLAHE_CLIP_LIMIT  = 2.0        # contrast clip — prevents noise amplification
CLAHE_TILE_GRID   = (8, 8)     # grid size — local contrast equalisation tiles
BILATERAL_D       = 9          # filter diameter (pixels)
BILATERAL_SIGMA_C = 30         # colour sigma — edge preservation strength
BILATERAL_SIGMA_S = 5          # spatial sigma — neighbourhood reach
GAMMA_VALUE       = 0.45       # γ < 1 brightens shadows (I_out = I^γ)
GAMMA_DARK_THRESH = 100        # mean L* below this triggers gamma correction
_SENTINEL_NAME    = ".enhanced"


def enhance_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Enhance a single BGR image.

    Steps
    -----
    1. CLAHE on the L* channel — boosts local contrast and reveals venation /
       surface texture without distorting the hue-based colour features used
       to discriminate Perilla and Cape Gooseberry.
    2. Bilateral filter — edge-preserving denoise; keeps the leaf margins and
       serration detail that Fourier shape descriptors and Gabor filters rely on.
    3. Adaptive gamma — brightens only images whose mean L* falls below
       GAMMA_DARK_THRESH, expanding the shadow tonal range for images shot under
       dense canopy without clipping highlights in well-lit images.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR format (OpenCV convention), dtype uint8.

    Returns
    -------
    np.ndarray
        Enhanced image in BGR format, same shape and dtype as input.
    """
    # ── Step 1: CLAHE on L* ──────────────────────────────────────────────────
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_eq = clahe.apply(l)
    img_clahe = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

    # ── Step 2: bilateral filter ─────────────────────────────────────────────
    img_bilateral = cv2.bilateralFilter(
        img_clahe, BILATERAL_D, BILATERAL_SIGMA_C, BILATERAL_SIGMA_S
    )

    # ── Step 3: adaptive gamma ───────────────────────────────────────────────
    mean_l = float(l_eq.mean())
    if mean_l < GAMMA_DARK_THRESH:
        lut = np.array(
            [int(((i / 255.0) ** GAMMA_VALUE) * 255) for i in range(256)],
            dtype=np.uint8,
        )
        img_out = cv2.LUT(img_bilateral, lut)
    else:
        img_out = img_bilateral

    return img_out


# ── offline dataset preprocessing ─────────────────────────────────────────────

def preprocess_dataset(dataset_dir: Path) -> None:
    """
    Apply enhance_image() to every image in dataset_dir in-place.

    Idempotent: writes a sentinel file (.enhanced) so re-running train.py
    does not re-process already-enhanced images.  The sentinel is placed
    inside dataset_dir so it is removed automatically if the dataset is
    deleted and re-extracted from the zip.

    Parameters
    ----------
    dataset_dir : Path
        Root of the extracted dataset (contains train/, valid/, test/).
    """
    sentinel = dataset_dir / _SENTINEL_NAME
    if sentinel.exists():
        print("[INFO] Dataset already enhanced — skipping preprocessing.")
        return

    img_paths = list(dataset_dir.rglob("images/*.jpg")) + \
                list(dataset_dir.rglob("images/*.jpeg")) + \
                list(dataset_dir.rglob("images/*.png"))

    if not img_paths:
        print("[WARN] No images found to enhance — check dataset_dir path.")
        return

    print(f"[INFO] Enhancing {len(img_paths)} dataset images "
          f"(CLAHE + bilateral + adaptive gamma) …")

    failed = 0
    for path in img_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[WARN] Could not read {path} — skipping.")
            failed += 1
            continue
        enhanced = enhance_image(img)
        cv2.imwrite(str(path), enhanced)

    sentinel.write_text(
        f"Enhanced {len(img_paths) - failed} images.\n"
        f"Pipeline: CLAHE(clip={CLAHE_CLIP_LIMIT}, grid={CLAHE_TILE_GRID})"
        f" -> Bilateral(d={BILATERAL_D}, sC={BILATERAL_SIGMA_C}, sS={BILATERAL_SIGMA_S})"
        f" -> Adaptive gamma(g={GAMMA_VALUE}, thresh={GAMMA_DARK_THRESH})\n",
        encoding="utf-8",
    )
    print(f"[INFO] Enhancement complete. "
          f"{len(img_paths) - failed} images written, {failed} skipped.")
