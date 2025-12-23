# explainability/yolo_ocr.py

from __future__ import annotations

from typing import Dict, Any, Callable, Optional, Tuple, List
import time
import numpy as np
from PIL import Image, ImageOps, ImageFilter

from skimage.segmentation import slic
from skimage.color import rgb2lab
from skimage.filters import gaussian

from ocr.base import OCRBase


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def clamp(v: int, vmin: int, vmax: int) -> int:
    return max(vmin, min(v, vmax))


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    expand_ratio: float = 0.4,  # ⬅️ +40% pour lisibilité
) -> Tuple[int, int, int, int]:
    """
    Agrandit une bounding box de manière contrôlée.
    """
    W, H = image_size
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    dx = int(w * expand_ratio / 2)
    dy = int(h * expand_ratio / 2)

    nx1 = clamp(x1 - dx, 0, W - 1)
    ny1 = clamp(y1 - dy, 0, H - 1)
    nx2 = clamp(x2 + dx, nx1 + 1, W)
    ny2 = clamp(y2 + dy, ny1 + 1, H)

    return nx1, ny1, nx2, ny2


def crop_bbox(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    return image.crop(bbox)


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------
# ROI preparation (LISIBLE)
# ---------------------------------------------------------------------

def prepare_roi(
    roi_raw: Image.Image,
    target_size: int = 256,
) -> Image.Image:
    """
    ROI lisible :
    - upscale volontaire
    - interpolation nette
    - amélioration de contours
    """

    w, h = roi_raw.size
    scale = target_size / max(w, h)

    roi = roi_raw.resize(
        (int(w * scale), int(h * scale)),
        Image.Resampling.BICUBIC,   # ⬅️ net, pas flou
    )

    roi = ImageOps.autocontrast(roi)
    roi = roi.filter(ImageFilter.EDGE_ENHANCE_MORE)

    return roi


def mask_sp(img_f: np.ndarray, segments: np.ndarray, sp_id: int) -> np.ndarray:
    out = img_f.copy()
    out[segments == sp_id] = 0.0
    return out


# ---------------------------------------------------------------------
# Main explain function
# ---------------------------------------------------------------------

def explain_yolo_ocr(
    ocr_model: OCRBase,
    image: Image.Image,
    target_index: int,
    n_segments: int = 60,
    compactness: float = 10.0,
    expand_ratio: float = 0.4,
    target_size: int = 256,
    iou_threshold: float = 0.3,
    max_perturb_preview: int = 6,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:

    # ---- OCR baseline
    base = ocr_model.predict(image)
    boxes = base.get("boxes", [])
    probs = base.get("token_probs", [])
    labels = base.get("tokens", [])

    if not boxes:
        raise RuntimeError("YOLO OCR: aucune bounding box détectée.")

    target_box = tuple(map(int, boxes[target_index]))
    baseline_score = float(probs[target_index])
    label = labels[target_index]

    # ---- 1) BB élargie (lisible)
    expanded_box = expand_bbox(
        target_box,
        image.size,
        expand_ratio=expand_ratio,
    )

    roi_raw = crop_bbox(image, expanded_box)

    # ---- 2) ROI lisible
    roi_prepared = prepare_roi(
        roi_raw,
        target_size=target_size,
    )

    # ---- 3) Segmentation SLIC propre
    roi_np = np.array(roi_prepared)

    roi_lab = rgb2lab(roi_np)
    roi_lab = gaussian(roi_lab, sigma=0.8, channel_axis=-1)

    segments = slic(
        roi_lab,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
    )

    img_f = roi_np.astype(np.float32) / 255.0
    n_sp = int(segments.max() + 1)
    sp_importance = np.zeros(n_sp, dtype=np.float32)

    perturbed_images: List[Image.Image] = []

    # ---- 4) Occlusion
    for sp_id in range(n_sp):

        masked = mask_sp(img_f, segments, sp_id)
        masked_pil = Image.fromarray((masked * 255).astype(np.uint8))

        if len(perturbed_images) < max_perturb_preview:
            perturbed_images.append(masked_pil)

        # repatch à la taille BB élargie
        masked_small = masked_pil.resize(
            roi_raw.size,
            Image.Resampling.BICUBIC,
        )

        patched = image.copy()
        patched.paste(masked_small, expanded_box[:2])

        pert = ocr_model.predict(patched)

        best_iou = 0.0
        best_score = 0.0

        for b, s in zip(pert.get("boxes", []), pert.get("token_probs", [])):
            ov = iou(target_box, tuple(map(int, b)))
            if ov > best_iou:
                best_iou = ov
                best_score = float(s)

        sp_importance[sp_id] = (
            baseline_score - best_score
            if best_iou >= iou_threshold
            else baseline_score
        )

        if progress_callback:
            progress_callback((sp_id + 1) / n_sp)

    return {
        "image": roi_prepared,          # ROI lisible
        "roi_raw": roi_raw,
        "segments": segments,
        "sp_importance": sp_importance,
        "baseline_score": baseline_score,
        "bbox": expanded_box,
        "label": label,
        "perturbed_images": perturbed_images,
    }
