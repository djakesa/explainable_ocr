import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import label2rgb
from skimage.segmentation import slic
import hashlib
import time

from ocr.trocr_model import TrOCRModel
from explainability.lime_ocr import explain_token_lime


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def image_hash(img: Image.Image) -> str:
    return hashlib.md5(np.array(img).tobytes()).hexdigest()


def build_heatmap(segments, sp_importance, top_k):
    heatmap = np.zeros_like(segments, dtype=np.float32)

    positive_idx = np.where(sp_importance > 0)[0]
    if len(positive_idx) == 0:
        return heatmap, np.array([], dtype=int)

    sorted_idx = positive_idx[np.argsort(sp_importance[positive_idx])[::-1]]
    top_idx = sorted_idx[:top_k]

    for seg_id in top_idx:
        heatmap[segments == seg_id] = sp_importance[seg_id]

    heatmap /= heatmap.max()
    return heatmap, top_idx


def generate_masked_examples(image, segments, n_examples=4, seed=0):
    rng = np.random.RandomState(seed)
    img_np = np.array(image).astype(np.float32) / 255.0
    n_sp = int(segments.max() + 1)

    examples = []
    for _ in range(n_examples):
        mask = rng.randint(0, 2, size=n_sp)
        masked = img_np.copy()
        for sp_id in range(n_sp):
            if mask[sp_id] == 0:
                masked[segments == sp_id] = 0.0
        examples.append(masked)

    return examples


# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Explainable OCR",
    page_icon="🔍",
    layout="wide"
)

st.markdown("# 🔍 Explainable OCR")
st.markdown("---")


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

for key, default in {
    "ocr_result": None,
    "ocr_image_hash": None,
    "segments": None,
    "segmentation_done": False,
    "lime_output": None,
    "explanation_done": False,
    "stop_lime": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Paramètres d’explication")

    n_samples = st.slider("Perturbations", 20, 300, 100, 20)
    n_segments = st.slider("Superpixels", 10, 80, 40, 5)
    compactness = st.slider("Compacité", 1.0, 50.0, 10.0)
    top_k = st.slider("Régions affichées", 1, 10, 3)

    if st.button("🛑 Stop calcul"):
        st.session_state.stop_lime = True


# ---------------------------------------------------------------------
# 1. Image upload
# ---------------------------------------------------------------------

st.markdown("## 1. Image d’entrée")

uploaded_file = st.file_uploader(
    "Déposez une image contenant du texte",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is None:
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
img_hash = image_hash(image)


# ---------------------------------------------------------------------
# 2. OCR (caché)
# ---------------------------------------------------------------------

@st.cache_resource
def load_ocr():
    return TrOCRModel()

ocr_model = load_ocr()

if st.session_state.ocr_image_hash != img_hash:
    with st.spinner("🔎 OCR en cours…"):
        st.session_state.ocr_result = ocr_model.predict(image)
        st.session_state.ocr_image_hash = img_hash
        st.session_state.segmentation_done = False
        st.session_state.explanation_done = False

ocr_result = st.session_state.ocr_result
tokens = ocr_result["tokens"]
token_probs = ocr_result["token_probs"]


# ---------------------------------------------------------------------
# 3. Résultat OCR
# ---------------------------------------------------------------------

st.markdown("## 2. Résultat OCR")

c1, c2 = st.columns([1, 1.2])
with c1:
    st.image(image, caption="Image d’entrée", use_column_width=True)
with c2:
    st.info(ocr_result["text"])


# ---------------------------------------------------------------------
# 4. Segmentation visuelle (PERSISTANTE)
# ---------------------------------------------------------------------

st.markdown("## 🧩 3. Segmentation visuelle")

if st.button("▶️ Générer segmentation"):
    with st.spinner("Segmentation en cours…"):
        img_np = np.array(image).astype(np.float32) / 255.0
        st.session_state.segments = slic(
            img_np,
            n_segments=n_segments,
            compactness=compactness,
            sigma=1,
            start_label=0
        )
    st.session_state.segmentation_done = True
    st.session_state.explanation_done = False

# 🔒 AFFICHAGE PERSISTANT
if st.session_state.segmentation_done:

    segments = st.session_state.segments

    seg_vis = label2rgb(segments, np.array(image), bg_label=0)
    st.image(seg_vis, caption="Segmentation (superpixels)", use_column_width=True)

    st.markdown("### 🔍 Exemples de masquage des superpixels")

    masked_examples = generate_masked_examples(image, segments)
    cols = st.columns(len(masked_examples))
    for col, img in zip(cols, masked_examples):
        col.image(img, use_column_width=True)


# ---------------------------------------------------------------------
# 5. Choix du token
# ---------------------------------------------------------------------

st.markdown("## 🔤 4. Choix du token")

if not st.session_state.segmentation_done:
    st.info("Veuillez d’abord effectuer la segmentation.")
    st.stop()

token_choice = st.selectbox(
    "Token à expliquer",
    range(len(tokens)),
    format_func=lambda i: f"{i:02d} — '{tokens[i]}' (p={token_probs[i]:.2f})"
)


# ---------------------------------------------------------------------
# 6. LIME (RECALCULÉ PAR TOKEN)
# ---------------------------------------------------------------------

st.markdown("## 🔥 5. Explication de la prédiction")

if st.button("▶️ Expliquer ce token"):

    progress_bar = st.progress(0.0)
    eta_text = st.empty()
    start = time.time()

    def progress_cb(p):
        progress_bar.progress(p)
        elapsed = time.time() - start
        if p > 0:
            eta = elapsed * (1 - p) / p
            eta_text.markdown(f"⏳ Temps restant estimé : `{eta:.1f}s`")

    out = explain_token_lime(
        ocr_model=ocr_model,
        image=image,
        token_index=token_choice,
        n_segments=n_segments,
        compactness=compactness,
        n_samples=n_samples,
        progress_callback=progress_cb,
        stop_flag={"stop": st.session_state.stop_lime},
    )

    eta_text.empty()
    st.session_state.lime_output = out
    st.session_state.explanation_done = True


if not st.session_state.explanation_done:
    st.stop()


# ---------------------------------------------------------------------
# 7. Visualisation finale
# ---------------------------------------------------------------------

segments = st.session_state.segments
sp_importance = st.session_state.lime_output["sp_importance"]

heatmap, top_idx = build_heatmap(segments, sp_importance, top_k)

img_np = np.array(image).astype(np.float32) / 255.0
gray = 1 - np.mean(img_np, axis=2)
text_mask = gray > 0.3

overlay = img_np.copy()
overlay[text_mask] = (
    overlay[text_mask] * (1 - heatmap[text_mask][:, None])
    + np.array([1.0, 0.0, 0.0]) * heatmap[text_mask][:, None]
)
overlay = np.clip(overlay, 0, 1)

st.image(
    overlay,
    caption=f"Zones explicatives (top {top_k})",
    use_column_width=True
)

# ---------------------------------------------------------------------
# Tableau quantitatif
# ---------------------------------------------------------------------

st.markdown("### 📊 Importance quantitative des régions")

top_scores = sp_importance[top_idx]
relative = top_scores / top_scores.sum()

df = pd.DataFrame({
    "Rang": range(1, len(top_idx) + 1),
    "Superpixel ID": top_idx,
    "Importance brute": top_scores,
    "Importance relative (%)": (100 * relative).round(2),
})

st.dataframe(df, use_container_width=True, hide_index=True)
