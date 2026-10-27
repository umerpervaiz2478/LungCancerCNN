"""LungAI: educational CT image classification dashboard."""
import io
from datetime import datetime
from pathlib import Path
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title="LungAI | CT Scan Analyzer", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

LABELS = ["Malignant", "Normal", "Benign"]  # MUST match the training label-index mapping
MODEL_PATH = Path(__file__).resolve().parent / "Lung_Model.h5"

st.markdown("""<style>
[data-testid="stAppViewContainer"] {background:linear-gradient(135deg,#071320,#0b2233 65%,#0b1b2b);color:#edf7ff}
[data-testid="stSidebar"] {background:#0a1c2b;border-right:1px solid #23445a}
.block-container {max-width:1350px;padding-top:2rem}
h1,h2,h3 {letter-spacing:-.03em}
.hero {padding:30px;border-radius:22px;border:1px solid #275568;background:linear-gradient(105deg,#113d4e,#10253a);margin-bottom:20px}
.hero h1 {color:#f1fbff;font-size:2.4rem;margin:8px 0}
.hero p {color:#bad2df;margin:0}
.eyebrow {font-size:.78rem;letter-spacing:2px;color:#5ce4cb;font-weight:800}
.result {padding:24px;border-radius:18px;background:#122e40;border:1px solid #32617a}
.result strong {font-size:2rem}
[data-testid="stMetric"] {border:1px solid #285065;border-radius:14px;padding:15px;background:#102a3c}
[data-testid="stFileUploader"] {border:1px dashed #43748a;border-radius:14px;padding:10px;background:#10283a}
.stButton button[kind="primary"] {background:#16bca8;color:#041a23;border:none;font-weight:750}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading LungAI model...")
def load_model(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)

def preprocess(image_bytes: bytes):
    import tensorflow as tf
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB").resize((244, 244))
        batch = np.asarray(rgb, dtype=np.float32)[None, ...]
    return tf.keras.applications.mobilenet_v2.preprocess_input(batch)

def scores_from_prediction(prediction):
    values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if values.size != 3 or not np.all(np.isfinite(values)):
        raise ValueError("Model output must contain exactly three finite class scores.")
    if np.all(values >= 0) and np.all(values <= 1) and np.isclose(values.sum(), 1, atol=.02):
        return values / values.sum()
    exp = np.exp(values - values.max())
    return exp / exp.sum()

def make_pdf(result):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    buf = io.BytesIO()
    styles = getSampleStyleSheet()
    story = [Paragraph("LungAI | Research Classification Report", styles["Title"]), Spacer(1, 18)]
    details = [f"File: {escape(result['filename'])}", f"Time: {result['timestamp']}",
               f"Predicted class: {result['label']}", f"Highest model score: {result['confidence']:.2f}%"]
    details += [f"{label}: {score:.2f}%" for label, score in zip(LABELS, result["scores"])]
    details.append("Research use only. Not a medical diagnosis. Model scores are not calibrated disease probabilities.")
    for line in details:
        story.extend([Paragraph(line, styles["Normal"]), Spacer(1, 10)])
    SimpleDocTemplate(buf, pagesize=A4).build(story)
    return buf.getvalue()

if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None

with st.sidebar:
    st.title("🫁 LungAI")
    st.caption("AI-POWERED CT IMAGE ANALYSIS")
    st.divider()
    st.markdown("**Dashboard sections**")
    st.caption("The Scan Analyzer is always visible on the main page. Use the tabs above to switch sections.")
    st.divider()
    st.markdown("**Model configuration**")
    st.write("**Input:** 244 × 244 RGB")
    st.write("**Classes:** Malignant · Normal · Benign")
    st.write("**Preprocessing:** MobileNetV2")
    st.divider()
    st.warning("Educational research prototype — not for clinical diagnosis.")

st.markdown('<div class="hero"><div class="eyebrow">DEEP LEARNING · MEDICAL IMAGING</div><h1>🫁 Lung CT Scan Analyzer</h1><p>Upload a CT image, run your trained Keras model, and explore its three classification scores.</p></div>', unsafe_allow_html=True)

scan_tab, history_tab, about_tab = st.tabs(["🔬 Scan Analyzer", "📊 Analysis History", "ℹ️ About LungAI"])

with scan_tab:
    a,b,c = st.columns(3)
    a.metric("Classification classes", "03")
    b.metric("Image resolution", "244 × 244")
