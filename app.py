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
LABELS = ["Malignant", "Normal", "benign"]

# Custom UI styling with clean hex codes
style_css = """<style>
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
</style>"""
st.markdown(style_css, unsafe_allow_html=True)

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
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from xml.sax.saxutils import escape
    except ImportError:
        return b"ReportLab library is required to generate the PDF report."
    buf = io.BytesIO()
    styles = getSampleStyleSheet()
    story = [Paragraph("LungAI | Research Report", styles["Title"]), Spacer(1, 18)]
    for line in [f"File: {escape(record["file"])}", f"Time: {record["time"]}",
                 f"Predicted class: {record["label"]}", f"Highest model score: {record["confidence"]:.2f}%",
                 *[f"{k}: {v:.2f}%" for k,v in record["scores"].items()],
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
    st.caption("Classes: Malignant / Normal / benign")
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
    with right:
        if data is not None:
            if st.button("Run Classification Analysis", type="primary"):
                with st.spinner("Processing CT scans... "):
                    try:
                        scores = infer(data)
                        best_idx = int(np.argmax(scores))
                        predicted_label = LABELS[best_idx]
                        confidence_pct = scores[best_idx] * 100
                        st.session_state.lung_result = {
                            "file": uploaded.name,
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "label": predicted_label,
                            "confidence": confidence_pct,
                            "scores": {LABELS[i]: float(scores[i]*100) for i in range(len(LABELS))}
                        }
                        st.session_state.lung_history.append(st.session_state.lung_result)
                    except Exception as e:
                        st.error(f"Error performing inference: {str(e)}")
            if st.session_state.lung_result is not None:
                res = st.session_state.lung_result
                st.success(f"Classification Result: {res["label"]}")
                st.metric(label="Highest Classification Confidence", value=f"{res["confidence"]:.2f}%")
                st.subheader("Class Probability Metrics")
                for class_name, score_pct in res["scores"].items():
                    st.progress(score_pct / 100.0)
                    st.write(f"{class_name}: **{score_pct:.2f}%**")
                try:
                    import reportlab
                    pdf_data = report_pdf(res)
                    st.download_button(
                        label="📥 Download Analysis PDF Report",
                        data=pdf_data,
                        file_name=f"lungai_report_{datetime.now().strftime('%Y%m%d%H%M')}.pdf",
                        mime="application/pdf"
                    )
                except ImportError:
                    st.info("Note: Install 'reportlab' in your python environment to enable PDF report downloading.")

elif section == "📊 Analysis History":
    st.subheader("Analysis Session Logs")
    if not st.session_state.lung_history:
        st.info("No images have been analyzed in this session yet. Results will appear here once you run a scan classification.")
    else:
        for i, item in enumerate(reversed(st.session_state.lung_history)):
            st.markdown(f"""**Record #{len(st.session_state.lung_history)-i}** - *{item["time"]}*
                        f"- **File Name:** {item[\"file\"]}"
                        f"- **Top Prediction:** `{item[\"label\"]}` ({item[\"confidence\"]:.2f}%)""")
            st.write(item["scores"])
            st.divider()

elif section == "ℹ️ About LungAI":
    st.subheader("Project Specifications & Model Configuration")
    st.write("""This application utilizes a custom Convolutional Neural Network (CNN) trained directly on lung CT scan data.
    The user uploads an image, which is processed to run classification across three categories:
    - **Malignant**: High probability patterns indicating tumorous changes.
    - **Normal**: Clean scan tissue structure.
    - **Benign**: Growth patterns not indicating aggressive malignancy.""")
    st.markdown("""**Disclaimer**: This tool is created for educational and research prototyping purposes only.
    It should not be used as clinical advice or for medical diagnosis of any kind.""")
