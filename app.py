"""LungAI educational CT image classification dashboard."""
import io
from pathlib import Path
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="LungAI | CT Analyzer", page_icon="🫁", layout="wide", initial_sidebar_state="expanded")

import numpy as np
from PIL import Image, UnidentifiedImageError

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "Lung_Model.h5"
LABELS = ["Malignant", "Normal", "Benign"]  # Confirm against training class indices.

st.markdown("""<style>
.stApp {background:linear-gradient(125deg,#081827,#0c2636);color:#eef8fc}
[data-testid="stSidebar"] {background:#0b2030}
.block-container {max-width:1250px;padding-top:1.8rem}
.hero {background:linear-gradient(100deg,#124558,#132c43);border:1px solid #2b6275;border-radius:18px;padding:25px;margin-bottom:18px}
.hero h1 {color:white;margin:6px 0;font-size:2.25rem}
.hero p {color:#c5dce8;margin:0}
.eyebrow {color:#69e7d3;letter-spacing:2px;font-size:.8rem;font-weight:800}
[data-testid="stFileUploader"] {background:#102d3e;border-radius:12px;padding:10px;border:1px dashed #528297}
[data-testid="stMetric"] {background:#123247;border:1px solid #2c6074;padding:15px;border-radius:14px}
.stButton button[kind="primary"] {background:#20c7b3;color:#04222a;border:none;font-weight:bold}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading trained model...")
def get_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)

def infer(data):
    import tensorflow as tf
    with Image.open(io.BytesIO(data)) as im:
        rgb = im.convert("RGB").resize((244, 244))
        arr = np.asarray(rgb, dtype=np.float32)[None, ...]
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    pred = np.asarray(get_model(str(MODEL_PATH)).predict(arr, verbose=0), dtype=float).reshape(-1)
    if len(pred) != 3 or not np.isfinite(pred).all():
        raise ValueError("Expected three finite output scores; check model output shape.")
    if not (np.all(pred >= 0) and np.all(pred <= 1) and np.isclose(pred.sum(), 1, atol=.02)):
        pred = np.exp(pred - pred.max())
    return pred / pred.sum()

def report_pdf(record):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from xml.sax.saxutils import escape
    buf = io.BytesIO()
    styles = getSampleStyleSheet()
    story = [Paragraph("LungAI | Research Report", styles["Title"]), Spacer(1, 18)]
    for line in [f"File: {escape(record['file'])}", f"Time: {record['time']}",
                 f"Predicted class: {record['label']}", f"Highest model score: {record['confidence']:.2f}%",
                 *[f"{k}: {v:.2f}%" for k,v in record['scores'].items()],
                 "Research only. Not a medical diagnosis. Scores are not calibrated clinical probabilities."]:
        story.extend([Paragraph(line, styles["Normal"]), Spacer(1, 9)])
    SimpleDocTemplate(buf, pagesize=A4).build(story)
    return buf.getvalue()

if "lung_history" not in st.session_state:
    st.session_state.lung_history = []
if "lung_result" not in st.session_state:
    st.session_state.lung_result = None

with st.sidebar:
    st.title("🫁 LungAI")
    st.caption("AI-POWERED CT IMAGE ANALYSIS")
    st.divider()
    section = st.radio("Navigate", ["🔬 Scan Analyzer", "📊 Analysis History", "ℹ️ About LungAI"], index=0)
    st.divider()
    st.caption("Model input: 244 × 244 RGB")
    st.caption("Preprocessing: MobileNetV2")
    st.caption("Classes: Malignant / Normal / Benign")
    st.warning("Research prototype — not a clinical diagnostic tool.")

st.markdown('<div class="hero"><div class="eyebrow">DEEP LEARNING · MEDICAL IMAGING</div><h1>🫁 Lung CT Scan Analyzer</h1><p>Upload a CT image to explore your trained model’s classification scores.</p></div>', unsafe_allow_html=True)

if section == "🔬 Scan Analyzer":
    st.subheader("Upload CT Scan")
    st.caption("Supported formats: PNG, JPG, JPEG. Upload an image to start.")
    uploaded = st.file_uploader("Select a CT image", type=["png", "jpg", "jpeg"], key="lung_upload")
    left, right = st.columns([1, 1], gap="large")
    with left:
        data = None
        if uploaded is not None:
            try:
                data = uploaded.getvalue()
                with Image.open(io.BytesIO(data)) as im:
                    preview = im.convert("RGB")
                st.image(preview, caption=uploaded.name, use_container_width=True)
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("This file could not be opened as an image.")
