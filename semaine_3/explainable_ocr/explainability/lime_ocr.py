"""
lime_ocr.py
===========

Explicabilité OCR par perturbation (LIME-like), version lisible :
- on affiche UNIQUEMENT les régions qui AIDENT la prédiction
- top-k superpixels les plus importants
- importance positive uniquement
- paramétrable (n_samples, n_segments, top_k)

Compatible TrOCR et OCRModel.
"""

from typing import Tuple
import numpy as np
from PIL import Image

from skimage.segmentation import slic
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

from ocr.base import OCRModel


# ---------------------------------------------------------------------
# Utils image
# ---------------------------------------------------------------------

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """PIL -> numpy RGB float [0,1]"""
    return np.array(image.convert("RGB"), dtype=np.float32) / 255.0


def mask_image(
    image: np.ndarray,
    segments: np.ndarray,
    mask: np.ndarray,
    background: float = 0.0
) -> np.ndarray:
    """Masque les superpixels dont mask[i] == 0"""
    out = image.copy()
    for seg_id in np.unique(segments):
        if mask[seg_id] == 0:
            out[segments == seg_id] = background
    return out


# ---------------------------------------------------------------------
# Score OCR pour un token donné
# ---------------------------------------------------------------------

def token_score(
    ocr_model: OCRModel,
    image_np: np.ndarray,
    token_index: int
) -> float:
    """Retourne P(token_k | image)"""
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    result = ocr_model.predict(image_pil)

    if token_index >= len(result["token_probs"]):
        return 0.0

    return result["token_probs"][token_index]


# ---------------------------------------------------------------------
# LIME OCR (POSITIF UNIQUEMENT)
# ---------------------------------------------------------------------

def explain_token_lime(
    ocr_model: OCRModel,
    image: Image.Image,
    token_index: int,
    n_segments: int = 40,
    n_samples: int = 10,
    kernel_width: float = 0.25,
    top_k: int = 3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Explique un token OCR en affichant UNIQUEMENT
    les régions qui AUGMENTENT sa probabilité.

    Returns
    -------
    heatmap : np.ndarray (H,W) ∈ [0,1]
    segments : np.ndarray (H,W)
    """

    rng = np.random.RandomState(random_state)

    image_np = pil_to_numpy(image)

    # --- Superpixels ---
    segments = slic(
        image_np,
        n_segments=n_segments,
        compactness=10,
        sigma=1,
        start_label=0
    )

    n_segs = segments.max() + 1

    # --- Masques binaires ---
    masks = rng.randint(0, 2, size=(n_samples, n_segs))
    masks[0, :] = 1  # image originale

    scores = np.zeros(n_samples)

    # --- OCR perturbé ---
    for i in tqdm(range(n_samples), desc="LIME perturbations"):
        perturbed = mask_image(
            image_np,
            segments,
            masks[i],
            background=0.0
        )
        scores[i] = token_score(
            ocr_model,
            perturbed,
            token_index
        )

    # --- Pondération locale ---
    distances = cosine_distances(masks, masks[0].reshape(1, -1)).ravel()
    weights = np.exp(-(distances ** 2) / kernel_width ** 2)

    # --- Régression locale ---
    reg = Ridge(alpha=1.0, fit_intercept=True)
    reg.fit(masks, scores, sample_weight=weights)

    sp_importance = reg.coef_

    # -----------------------------------------------------------------
    # >>> ICI : ON GARDE UNIQUEMENT CE QUI AIDE (importance positive)
    # -----------------------------------------------------------------

    positive_indices = np.where(sp_importance > 0)[0]

    if len(positive_indices) == 0:
        # aucun superpixel n'aide → heatmap vide
        return np.zeros_like(segments, dtype=np.float32), segments

    # Top-k parmi les positifs
    top_k = min(top_k, len(positive_indices))
    top_indices = positive_indices[
        np.argsort(sp_importance[positive_indices])[-top_k:]
    ]

    # --- Heatmap sparse (positif uniquement) ---
    heatmap = np.zeros_like(segments, dtype=np.float32)
    for seg_id in top_indices:
        heatmap[segments == seg_id] = sp_importance[seg_id]

    # Normalisation [0,1]
    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap, segments


# ---------------------------------------------------------------------
# Test local
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    from ocr.trocr_model import TrOCRModel

    image = Image.open("examples/test_anglais.png").convert("RGB")

    ocr = TrOCRModel()
    result = ocr.predict(image)

    print("\nTexte OCR :", result["text"])
    for i, t in enumerate(result["tokens"]):
        print(f"{i:02d} : {t} ({result['token_probs'][i]:.3f})")

    token_index = 1
    print(f"\nExplication du token #{token_index} -> '{result['tokens'][token_index]}'")

    heatmap, segments = explain_token_lime(
        ocr_model=ocr,
        image=image,
        token_index=token_index,
        n_segments=40,
        n_samples=10,
        top_k=3,
    )

    # --- Visualisation ---
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Heatmap (zones qui aident)")
    plt.imshow(heatmap, cmap="Reds", vmin=0, vmax=1)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(image)
    plt.imshow(heatmap, cmap="Reds", alpha=0.6, vmin=0, vmax=1)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("/home/jmerhrioui/soprasteria/bee/semaine_3/explainable_ocr/examples/lime_explanation.png", dpi=200)
    print("Figure sauvegardée dans lime_explanation.png")
