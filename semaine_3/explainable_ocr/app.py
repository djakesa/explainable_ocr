import streamlit as st
import numpy as np
from PIL import Image

from ocr.trocr_model import TrOCRModel
from explainability.lime_ocr import explain_token_lime

# ---------------------------------------------------------------------
# Configuration Streamlit
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Explainable OCR (TrOCR + LIME)",
    layout="wide"
)

st.title("🔍 Explainable OCR")
st.write(
    """
Cette interface permet d'expliquer **visuellement** la prédiction d’un token OCR
à l’aide d’une méthode par perturbation (LIME).

👉 Seules les **régions qui aident la prédiction** sont affichées.
"""
)

# ---------------------------------------------------------------------
# Sidebar — paramètres utilisateur
# ---------------------------------------------------------------------

st.sidebar.header("⚙️ Paramètres LIME")

n_samples = st.sidebar.slider(
    "Nombre de perturbations",
    min_value=20,
    max_value=300,
    value=100,
    step=20,
    help="Plus élevé = explication plus stable mais plus lente"
)

n_segments = st.sidebar.slider(
    "Nombre de superpixels",
    min_value=10,
    max_value=80,
    value=40,
    step=5,
    help="Contrôle la finesse spatiale de l’explication"
)

top_k = st.sidebar.slider(
    "Nombre de régions explicatives",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help="Nombre de zones visuelles affichées"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "🟥 Les régions rouges correspondent aux zones **qui augmentent la probabilité** du token sélectionné."
)

# ---------------------------------------------------------------------
# Chargement image
# ---------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Déposez une image contenant du texte",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is None:
    st.info("Veuillez déposer une image pour commencer.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

# ---------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------

@st.cache_resource
def load_ocr():
    return TrOCRModel()

ocr_model = load_ocr()

with st.spinner("🔎 Reconnaissance du texte (OCR en cours)…"):
    ocr_result = ocr_model.predict(image)

tokens = ocr_result["tokens"]
token_probs = ocr_result["token_probs"]

# ---------------------------------------------------------------------
# Affichage OCR
# ---------------------------------------------------------------------

st.subheader("📄 Texte reconnu")
st.write(f"**{ocr_result['text']}**")

st.subheader("🔤 Tokens détectés")

token_labels = [
    f"{i:02d} — '{t}' (p={token_probs[i]:.2f})"
    for i, t in enumerate(tokens)
]

token_choice = st.selectbox(
    "Sélectionnez le token à expliquer",
    options=list(range(len(tokens))),
    format_func=lambda i: token_labels[i]
)

# ---------------------------------------------------------------------
# Bouton explication
# ---------------------------------------------------------------------

if st.button("🔥 Expliquer ce token"):
    with st.spinner("🧠 Calcul de l’explication (LIME)…"):
        heatmap, _ = explain_token_lime(
            ocr_model=ocr_model,
            image=image,
            token_index=token_choice,
            n_segments=n_segments,
            n_samples=n_samples,
            top_k=top_k,
        )

    # -----------------------------------------------------------------
    # Affichage des résultats
    # -----------------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Image originale")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader(" Explication visuelle")
        overlay = np.array(image).astype(np.float32) / 255.0
        overlay[..., 0] += heatmap * 0.8  # rouge
        overlay = np.clip(overlay, 0, 1)

        st.image(overlay, use_column_width=True)

    st.markdown("---")

    st.subheader("📖 Interprétation")
    st.write(
        f"""
Le token **'{tokens[token_choice]}'** est principalement influencé par  
les **{top_k} régions rouges** affichées ci-dessus.

Ces régions correspondent aux zones de l’image qui **augmentent la probabilité**
de ce token selon le modèle OCR.
"""
    )
