"""
lime_ocr.py
===========

Explicabilité OCR par perturbation (LIME-like).

Caractéristiques :
- calcule l'importance brute de CHAQUE superpixel
- ne fait AUCUNE sélection top-k
- ne construit AUCUNE heatmap finale
- support :
    - progression (progress_callback)
    - interruption utilisateur (stop_flag)
- retourne :
    - segments (SLIC)
    - sp_importance (importance LIME brute par superpixel)
    - images perturbées (debug / pédagogie)
    - scores OCR associés aux perturbations

Compatible TrOCR et OCRBase.
"""

from typing import Dict, Any, List, Optional, Callable
import numpy as np
from PIL import Image

from skimage.segmentation import slic
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from ocr.base import OCRBase



# ---------------------------------------------------------------------
# Utils image
# ---------------------------------------------------------------------

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """PIL -> numpy RGB float [0,1]"""
    return np.array(image.convert("RGB"), dtype=np.float32) / 255.0


def numpy01_to_pil(image_np01: np.ndarray) -> Image.Image:
    """numpy [0,1] -> PIL"""
    return Image.fromarray((np.clip(image_np01, 0, 1) * 255).astype(np.uint8))


def mask_image(
    image: np.ndarray,
    segments: np.ndarray,
    mask: np.ndarray,
    background: float = 0.0
) -> np.ndarray:
    """
    Masque les superpixels dont mask[i] == 0
    """
    out = image.copy()
    for seg_id in np.unique(segments):
        if mask[seg_id] == 0:
            out[segments == seg_id] = background
    return out


# ---------------------------------------------------------------------
# Score OCR pour un token donné
# ---------------------------------------------------------------------

def token_score(
    ocr_model: OCRBase,
    image_np01: np.ndarray,
    token_index: int
) -> float:
    """
    Retourne la probabilité du token donné pour une image perturbée.
    """
    image_pil = numpy01_to_pil(image_np01)
    result = ocr_model.predict(image_pil)

    if token_index >= len(result["token_probs"]):
        return 0.0

    return float(result["token_probs"][token_index])


# ---------------------------------------------------------------------
# LIME OCR (BACKEND PUR)
# ---------------------------------------------------------------------

def explain_token_lime(
    ocr_model: OCRBase,
    image: Image.Image,
    token_index: int,
    n_segments: int = 40,
    compactness: float = 10.0,
    n_samples: int = 100,
    kernel_width: float = 0.25,
    random_state: int = 42,
    progress_callback: Optional[Callable[[float], None]] = None,
    stop_flag: Optional[dict] = None,
    n_debug_images: int = 5,
    background_value: float = 0.0,
) -> Dict[str, Any]:
    """
    Calcule l'importance LIME brute de chaque superpixel
    pour un token OCR donné.

    Aucune sélection top-k ici.
    """

    rng = np.random.RandomState(random_state)
    image_np = pil_to_numpy(image)

    # -----------------------------------------------------------------
    # Segmentation SLIC
    # -----------------------------------------------------------------
    segments = slic(
        image_np,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1,
        start_label=0
    )

    n_segs = int(segments.max() + 1)

    # -----------------------------------------------------------------
    # Génération des perturbations
    # -----------------------------------------------------------------
    masks = rng.randint(0, 2, size=(n_samples, n_segs)).astype(np.int32)
    masks[0, :] = 1  # image originale (référence)

    scores = np.zeros(n_samples, dtype=np.float32)
    perturbed_images: List[np.ndarray] = []

    iterator = range(n_samples)
    if progress_callback is None:
        iterator = tqdm(iterator, desc="LIME perturbations")

    for i in iterator:

        if stop_flag is not None and stop_flag.get("stop", False):
            raise RuntimeError("LIME interrupted by user")

        perturbed = mask_image(
            image_np,
            segments,
            masks[i],
            background=background_value
        )

        scores[i] = token_score(
            ocr_model,
            perturbed,
            token_index
        )

        if len(perturbed_images) < n_debug_images:
            perturbed_images.append(perturbed)

        if progress_callback is not None:
            progress_callback((i + 1) / float(n_samples))

    # -----------------------------------------------------------------
    # Régression locale (LIME)
    # -----------------------------------------------------------------
    distances = cosine_distances(masks, masks[0].reshape(1, -1)).ravel()
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

    reg = Ridge(alpha=1.0)
    reg.fit(masks, scores, sample_weight=weights)

    sp_importance = reg.coef_.astype(np.float32)

    # -----------------------------------------------------------------
    # Retour BACKEND PUR (sans top-k, sans heatmap)
    # -----------------------------------------------------------------
    return {
        "segments": segments,
        "sp_importance": sp_importance,
        "perturbed_images": perturbed_images,
        "scores": scores,
    }


# ---------------------------------------------------------------------
# Test local (optionnel)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb
    from ocr.trocr_model import TrOCRModel

    image = Image.open("examples/test_anglais.png").convert("RGB")

    ocr = TrOCRModel()
    result = ocr.predict(image)

    print("Texte OCR :", result["text"])
    for i, t in enumerate(result["tokens"]):
        print(f"{i:02d} : {t} ({result['token_probs'][i]:.3f})")

    out = explain_token_lime(
        ocr_model=ocr,
        image=image,
        token_index=0,
        n_segments=40,
        compactness=10.0,
        n_samples=50,
    )

    print("Importances par superpixel :", out["sp_importance"].shape)

    plt.figure(figsize=(5, 5))
    plt.title("Segmentation")
    plt.imshow(label2rgb(out["segments"], np.array(image), bg_label=0))
    plt.axis("off")
    plt.show()
