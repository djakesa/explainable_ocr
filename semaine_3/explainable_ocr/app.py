# ============================================================
# Explainable OCR — Application Streamlit
# ============================================================
#
# Cette application permet d'expliquer visuellement
# les décisions prises par un modèle OCR (TrOCR ou YOLO OCR)
# en utilisant des méthodes de perturbations locales
# inspirées de LIME.
#
# Objectif :
# - Comprendre POURQUOI un texte est reconnu
# - Fournir une explication lisible pour un humain
# - Outil d’audit, pas d’inférence industrielle
#
# ============================================================

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from skimage.segmentation import mark_boundaries

from ocr.trocr_model import TrOCRModel
from ocr.yolo_ocr_model import YOLOOCRModel
from explainability.lime_ocr import explain_token_lime
from explainability.yolo_ocr import explain_yolo_ocr


# ============================================================
# 1. Fonctions utilitaires
# ============================================================

def image_hash(img: Image.Image) -> str:
    """
    Calcule un hash MD5 à partir des pixels de l’image.

    Utilité :
    - détecter un changement d’image
    - réinitialiser proprement l’état Streamlit
    """
    return hashlib.md5(np.asarray(img).tobytes()).hexdigest()


def strong_red_overlay(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.85,
) -> np.ndarray:
    """
    Applique un overlay rouge fortement contrasté
    pour visualiser l'importance des régions.

    Rouge intense   → région critique
    Rouge modéré    → région contributrice
    Pas de rouge    → région ignorée
    """
    img = image_np.astype(np.float32) / 255.0
    heat = heatmap[..., None]

    overlay = img.copy()

    overlay[..., 0] = np.clip(
        (1 - alpha * heat[..., 0]) * overlay[..., 0] + alpha * heat[..., 0],
        0, 1
    )

    overlay[..., 1] *= (1 - 0.6 * heat[..., 0])
    overlay[..., 2] *= (1 - 0.6 * heat[..., 0])

    return np.clip(overlay, 0, 1)


def draw_boxes(image, boxes, labels=None, selected=None):
    """
    Dessine les bounding boxes YOLO sur l’image.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        is_sel = i == selected
        color = "red" if is_sel else "gray"
        width = 4 if is_sel else 2

        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        if labels:
            draw.text((x1, max(0, y1 - 14)), labels[i], fill=color)

    return img


# ============================================================
# 2. Gestion de l’état Streamlit
# ============================================================

def init_state():
    """
    Initialise toutes les variables persistantes.
    Streamlit est stateless par défaut.
    """
    defaults = dict(
        ocr_type=None,
        ocr_model=None,
        model_key=None,
        image_key=None,
        ocr_result=None,
        target_idx=None,
        seg_output=None,
        base_image=None,
        segmentation_done=False,
        explanation_done=False,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ============================================================
# 3. Configuration de la page
# ============================================================

st.set_page_config(page_title="Explicabilité et OCR", layout="wide")
st.title("Explicabilité et OCR")

st.markdown("""
### Objectif de l'application

Cette application permet de **comprendre visuellement**
les décisions prises par un modèle OCR.

👉 On ne cherche pas la performance maximale,  
👉 mais la **compréhension humaine** et l’**audit**.
""")

st.markdown("---")


# ============================================================
# 4. Barre latérale — paramètres utilisateur
# ============================================================

with st.sidebar:
    st.header("🧠 Moteur OCR")

    ocr_choice = st.radio(
        "Choisissez le modèle OCR",
        ["TrOCR (séquentiel)", "YOLO OCR (détection)"]
    )

    uploaded = None
    if ocr_choice.startswith("YOLO"):
        uploaded = st.file_uploader("Modèle YOLO (.pt)", type=["pt"])

    st.markdown("---")
    st.header("⚙️ Paramètres d’explication")

    n_samples = st.slider("Perturbations (TrOCR)", 20, 300, 100, 20)
    n_segments = st.slider("Nombre de superpixels", 3, 50, 25, 1)
    compactness = st.slider("Compacité des superpixels", 1.0, 40.0, 10.0)

    st.markdown("---")
    st.header("👁️ Affichage")

    top_k = st.slider("Régions affichées", 1, 10, 3)


# ============================================================
# 5. Chargement du modèle OCR
# ============================================================

if ocr_choice.startswith("TrOCR"):
    if st.session_state.ocr_type != "trocr":
        st.session_state.ocr_model = TrOCRModel()
        st.session_state.ocr_type = "trocr"
        st.session_state.segmentation_done = False
        st.session_state.explanation_done = False
else:
    if uploaded is None:
        st.info("Veuillez charger un modèle YOLO OCR.")
        st.stop()

    path = Path("uploaded_models") / uploaded.name
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        path.write_bytes(uploaded.read())

    key = f"yolo::{path.name}"
    if st.session_state.model_key != key:
        st.session_state.ocr_model = YOLOOCRModel(str(path))
        st.session_state.ocr_type = "yolo"
        st.session_state.model_key = key
        st.session_state.segmentation_done = False
        st.session_state.explanation_done = False

ocr_model = st.session_state.ocr_model


# ============================================================
# 6. Chargement de l’image
# ============================================================

st.markdown("## 1️⃣ Image d’entrée")

st.markdown("""
**Étape 1 :**  
Veuillez charger une image contenant du texte.
""")

img_file = st.file_uploader("Charger une image", type=["png", "jpg", "jpeg"])
if img_file is None:
    st.stop()

image = Image.open(img_file).convert("RGB")
img_key = image_hash(image)

if st.session_state.image_key != img_key:
    st.session_state.image_key = img_key
    st.session_state.segmentation_done = False
    st.session_state.explanation_done = False
    st.session_state.target_idx = None

st.image(image, use_column_width=True)


# ============================================================
# 7. Prédiction OCR
# ============================================================

st.markdown("## 2️⃣ Prédiction OCR")


if st.session_state.ocr_type == "trocr":
    st.markdown("""
### Comment fonctionne TrOCR ?  

#### Principe général
TrOCR est un modèle OCR **séquentiel** qui transforme une image en une **suite de tokens**
(un token peut contenir plusieurs caractères).

#### Comment le modèle décide
- Le modèle regarde **toute l’image**
- Il prédit le texte **token par token**
- Chaque token dépend :
  - de l’image
  - **des tokens déjà prédits**

#### Ce que l’on explique ici
Quand vous sélectionnez un token,  
l’explication montre **quelles zones de l’image ont le plus influencé sa prédiction**,  
en gardant **le contexte des tokens précédents fixe**.

On explique donc **l’influence visuelle sur une décision précise du modèle**.

""")

else:
    st.markdown("""
### Comment fonctionne YOLO OCR ?

### Comment fonctionne YOLO OCR ?

#### Principe général
YOLO OCR est un modèle OCR basé sur la **détection d’objets**.

Il ne lit pas le texte comme une séquence globale,
mais **identifie des zones de texte indépendantes** dans l’image.

#### Comment le modèle décide
- Le modèle analyse l’image entière
- Il détecte des **bounding boxes** correspondant à des zones de texte
- Chaque zone est reconnue **indépendamment des autres**, avec un score de confiance

####  Ce que l’on explique ici
Quand vous sélectionnez une bounding box,  
l’explication montre **quelles régions à l’intérieur de cette zone**
ont le plus influencé la reconnaissance du texte associé.

 YOLO OCR base sa décision sur la **localisation et le contenu visuel local**,
sans dépendance au contexte des autres zones.

""")


if st.button("Lancer l’OCR"):
    with st.spinner("OCR en cours..."):
        st.session_state.ocr_result = ocr_model.predict(image)
        st.session_state.segmentation_done = False
        st.session_state.explanation_done = False

if st.session_state.ocr_result is None:
    st.stop()

ocr = st.session_state.ocr_result


# ============================================================
# 8. Affichage du résultat OCR
# ============================================================

if st.session_state.ocr_type == "trocr":
    st.markdown("**Texte reconnu :**")
    st.write(ocr["text"])
else:
    boxes = ocr.get("boxes", [])
    tokens = ocr.get("tokens", [])

    if not boxes:
        st.warning("Aucune détection.")
        st.stop()

    st.markdown("Zones détectées :")
    st.image(draw_boxes(image, boxes, tokens), use_column_width=True)


# ============================================================
# 9. Sélection de la cible à expliquer
# ============================================================

st.markdown("## 3️⃣ Sélection de la décision à expliquer")

st.markdown("""
On choisit **une décision locale précise** :
- un token (TrOCR)
- ou une bounding box (YOLO)
""")

if st.session_state.ocr_type == "trocr":
    idx = st.selectbox(
        "Token à expliquer",
        range(len(ocr["tokens"])),
        format_func=lambda i: f"{i} — '{ocr['tokens'][i]}' (p={ocr['token_probs'][i]:.2f})"
    )
else:
    idx = st.selectbox(
        "Bounding box à expliquer",
        range(len(ocr["boxes"])),
        format_func=lambda i: f"{i} — {ocr['tokens'][i]} (conf={ocr['token_probs'][i]:.2f})"
    )
    st.image(draw_boxes(image, ocr["boxes"], ocr["tokens"], selected=idx),
             use_column_width=True)

st.session_state.target_idx = idx


# ============================================================
# 10. Segmentation & perturbations
# ============================================================

st.markdown("## 4️⃣ Segmentation & perturbations")

st.markdown("""
Lors de cette étape, on applique une méthode d’explicabilité locale basée sur des perturbations.

L’image est d’abord découpée en **superpixels**, c’est-à-dire en régions cohérentes visuellement.
Ces régions servent d’unités d’analyse interprétables.

Ensuite, différentes **perturbations locales** sont générées en masquant ou modifiant certains superpixels.
Pour chaque perturbation, on observe comment la **prédiction du modèle change**.

En analysant l’impact de chaque superpixel sur la probabilité du token (ou de la zone) sélectionné(e),
on peut estimer **l’importance de chaque région de l’image dans la décision du modèle**.

""")

if st.button("Lancer la segmentation"):
    bar = st.progress(0.0)

    def progress(p):
        bar.progress(p)

    with st.spinner("Segmentation et perturbations..."):
        if st.session_state.ocr_type == "trocr":
            out = explain_token_lime(
                ocr_model,
                image,
                idx,
                n_segments,
                compactness,
                n_samples,
                progress_callback=progress,
            )
            base_image = image
        else:
            out = explain_yolo_ocr(
                ocr_model,
                image,
                idx,
                n_segments=n_segments,
                compactness=compactness,
                progress_callback=progress,
            )
            base_image = out["image"]

    st.session_state.seg_output = out
    st.session_state.base_image = base_image
    st.session_state.segmentation_done = True
    st.session_state.explanation_done = False

    st.success("Segmentation terminée")

if not st.session_state.segmentation_done:
    st.stop()


# ============================================================
# 11. Visualisation de la segmentation
# ============================================================

segments = st.session_state.seg_output["segments"]

roi_image = image if st.session_state.ocr_type == "trocr" else st.session_state.seg_output["image"]

st.markdown("### 🔍 Visualisation des superpixels")

seg_vis = mark_boundaries(
    np.asarray(roi_image),
    segments,
    color=(1, 0, 0),
    mode="thick",
)

st.image(seg_vis, use_column_width=True)


# ============================================================
# 12. Explication visuelle finale
# ============================================================

st.markdown("## 5️⃣ Explication visuelle finale")

st.markdown("""
Cette visualisation met en évidence les régions de l’image
qui contribuent le plus à la décision du modèle pour la prédiction sélectionnée.

Les zones colorées avec une intensité plus forte
correspondent aux régions ayant **l’impact le plus important**
sur la probabilité du token ou de la zone expliquée.
 
""")

if st.button("Afficher l’explication"):
    st.session_state.explanation_done = True

if not st.session_state.explanation_done:
    st.stop()

importance = st.session_state.seg_output["sp_importance"]
order = np.argsort(importance)[::-1][:top_k]

heatmap = np.zeros_like(segments, dtype=np.float32)
for i in order:
    heatmap[segments == i] = importance[i]

heatmap /= heatmap.max() + 1e-8

overlay = strong_red_overlay(
    np.asarray(st.session_state.base_image),
    heatmap,
)

st.image(overlay, use_column_width=True)


# ============================================================
# 13. Tableau d’importance
# ============================================================

st.markdown("### Importance des régions")

abs_imp = importance[order]
rel_imp = abs_imp / abs_imp.sum()

df = pd.DataFrame({
    "Rang": range(1, len(order) + 1),
    "Superpixel": order,
    "Importance absolue": abs_imp.round(6),
    "Importance relative (%)": (100 * rel_imp).round(2),
})

st.dataframe(df, use_container_width=True)


# ============================================================
# 14. Conclusion pédagogique
# ============================================================

st.markdown("""
## Conclusion

Cette application montre **comment** et **où**
un modèle OCR regarde pour prendre une décision.

Conçue pour :
- interprétabilité
- audit
- démonstration scientifique

""")
