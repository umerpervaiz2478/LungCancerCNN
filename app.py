
import io
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError

st.set_page_config(
    page_title="LungAI | CT Scan Analyzer",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Lung_Model.h5"
CLASS_LABELS = ["Malignant", "Normal", "Benign"]

st.markdown("""
<style>
.stApp {
    background: linear-gradient(125deg, #081827, #0c2636);
    color: #eef8fc;
}
[data-testid="stSidebar"] {
    background: #0b2030;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}
.hero {
    background: linear-gradient(110deg, #124558, #132c43);
    border: 1px solid #2b6275;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 22px;
}
.hero h1 {
    color: white;
    font-size: 2.3rem;
    margin: 10px 0;
}
.hero p {
    color: #c5dce8;
}
.eyebrow {
    color: #69e7d3;
    letter-spacing: 2px;
    font-size: 0.8rem;
    font-weight: 800;
}
[data-testid="stFileUploader"] {
    background: #102d3e;
    border: 1px dashed #528297;
    border-radius: 14px;
    padding: 12px;
}
[data-testid="stMetric"] {
    background: #123247;
    border: 1px solid #2c6074;
    padding: 18px;
    border-radius: 14px;
}
.stButton button[kind="primary"] {
    background: #20c7b3;
    color: #04222a;
    border: none;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AI model...")
def load_lung_model():
    import tensorflow as tf
    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


def predict_image(image_bytes):
    import tensorflow as tf

    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image = image.resize((244, 244))
        image_array = np.asarray(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    processed = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    model = load_lung_model()
    output = model.predict(processed, verbose=0)
    scores = np.asarray(output, dtype=float).reshape(-1)

    if len(scores) != 3 or not np.isfinite(scores).all():
        raise ValueError(
            "Model must return three valid classification scores."
        )

    if (
        np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(scores.sum(), 1, atol=0.02)
    ):
        scores = np.exp(scores - np.max(scores))

    scores = scores / scores.sum()

    return scores


def generate_pdf(result):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer
    )
    from reportlab.lib.styles import getSampleStyleSheet
    from xml.sax.saxutils import escape

    buffer = io.BytesIO()
    styles = getSampleStyleSheet()

    document = SimpleDocTemplate(buffer, pagesize=A4)

    story = [
        Paragraph("LungAI - Analysis Report", styles["Title"]),
        Spacer(1, 20)
    ]

    lines = [
        "File: " + escape(result["file"]),
        "Date: " + result["time"],
        "Prediction: " + result["label"],
        "Highest model score: "
        + f'{result["confidence"]:.2f}%'
    ]

    for label, score in result["scores"].items():
        lines.append(f"{label}: {score:.2f}%")

    lines.append(
        "Educational research only. Not a medical diagnosis. "
        "Model scores are not calibrated clinical probabilities."
    )

    for line in lines:
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 12))

    document.build(story)
    return buffer.getvalue()


if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

if "last_file" not in st.session_state:
    st.session_state.last_file = None


with st.sidebar:
    st.title("🫁 LungAI")
    st.caption("AI-POWERED CT IMAGE ANALYSIS")
    st.divider()

    section = st.radio(
        "Dashboard Navigation",
        [
            "🔬 Scan Analyzer",
            "📊 Analysis History",
            "ℹ️ About LungAI"
        ],
        index=0
    )

    st.divider()
    st.subheader("Model Configuration")
    st.write("Input: 244 × 244 RGB")
    st.write("Preprocessing: MobileNetV2")
    st.write("Classes: Malignant, Normal, Benign")

    st.warning(
        "Research prototype. Not for clinical diagnosis."
    )


st.markdown("""
<div class="hero">
    <div class="eyebrow">
        DEEP LEARNING · MEDICAL IMAGING
    </div>
    <h1>🫁 Lung CT Scan Analyzer</h1>
    <p>
        Upload a CT scan image and explore classification
        results from your trained deep learning model.
    </p>
</div>
""", unsafe_allow_html=True)


if section == "🔬 Scan Analyzer":

    st.subheader("Upload CT Scan Image")
    st.caption("Supported formats: PNG, JPG, JPEG")

    uploaded_file = st.file_uploader(
        "Choose a CT scan image",
        type=["png", "jpg", "jpeg"],
        key="ct_upload"
    )

    if uploaded_file is None:
        st.info("Upload a CT scan image to begin analysis.")

    else:
        image_bytes = uploaded_file.getvalue()
        file_id = (
            uploaded_file.name,
            len(image_bytes),
            hash(image_bytes)
        )

        if st.session_state.last_file != file_id:
            st.session_state.result = None
            st.session_state.last_file = file_id

        left, right = st.columns([1, 1], gap="large")

        with left:
            st.subheader("CT Scan Preview")

            try:
                image = Image.open(io.BytesIO(image_bytes))
                image = image.convert("RGB")

                st.image(
                    image,
                    caption=uploaded_file.name,
                    use_container_width=True
                )
                valid_image = True

            except (UnidentifiedImageError, OSError):
                st.error("Invalid or corrupted image.")
                valid_image = False

        with right:
            st.subheader("AI Classification")

            if not MODEL_PATH.exists():
                st.error(
                    "Lung_Model.h5 is missing. "
                    "Upload your trained model to GitHub."
                )

            elif valid_image:

                if st.button(
                    "🔬 Analyze CT Scan",
                    type="primary",
                    use_container_width=True
                ):
                    with st.spinner("Analyzing image..."):
                        try:
                            probabilities = predict_image(
                                image_bytes
                            )

                            best_index = int(
                                np.argmax(probabilities)
                            )

                            result = {
                                "file": uploaded_file.name,
                                "time": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                                "label": CLASS_LABELS[best_index],
                                "confidence": float(
                                    probabilities[best_index] * 100
                                ),
                                "scores": {
                                    CLASS_LABELS[i]: float(
                                        probabilities[i] * 100
                                    )
                                    for i in range(3)
                                }
                            }

                            st.session_state.result = result
                            st.session_state.history.append(
                                result.copy()
                            )

                        except Exception as error:
                            st.error(
                                f"Prediction failed: {error}"
                            )

                if st.session_state.result is not None:
                    result = st.session_state.result

                    st.success(
                        "Predicted Class: " + result["label"]
                    )

                    st.metric(
                        "Highest Model Score",
                        f'{result["confidence"]:.2f}%'
                    )

                    st.subheader("Classification Scores")

                    for label, score in result["scores"].items():
                        st.write(
                            f"**{label}: {score:.2f}%**"
                        )
                        st.progress(
                            min(max(score / 100, 0.0), 1.0)
                        )

                    try:
                        pdf_bytes = generate_pdf(result)

                        st.download_button(
                            "📥 Download PDF Report",
                            data=pdf_bytes,
                            file_name="LungAI_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

                    except ImportError:
                        st.warning(
                            "Install reportlab to enable PDF reports."
                        )

    st.divider()
    st.caption(
        "LungAI is an educational AI project. "
        "Predictions must not be used for medical decisions."
    )


elif section == "📊 Analysis History":

    st.subheader("Analysis History")

    if not st.session_state.history:
        st.info(
            "No scans analyzed yet in this session."
        )

    else:
        import pandas as pd

        records = []

        for item in st.session_state.history:
            row = {
                "File": item["file"],
                "Time": item["time"],
                "Prediction": item["label"],
                "Highest Score (%)": round(
                    item["confidence"], 2
                )
            }
            row.update(item["scores"])
            records.append(row)

        dataframe = pd.DataFrame(records)

        st.dataframe(
            dataframe,
            use_container_width=True
        )

        st.download_button(
            "📥 Export History as CSV",
            data=dataframe.to_csv(index=False),
            file_name="LungAI_History.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()


elif section == "ℹ️ About LungAI":

    st.subheader("About LungAI")

    st.write("""
    LungAI is a deep learning research dashboard
    designed to classify lung CT scan images
    into three categories using a trained Keras model.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Classification Classes", "03")

    with col2:
        st.metric("Image Resolution", "244 × 244")

    with col3:
        st.metric("Model Format", "Keras H5")

    st.subheader("Classification Categories")

    st.write("**Malignant:** Model category for malignant images.")
    st.write("**Normal:** Model category for normal images.")
    st.write("**Benign:** Model category for benign images.")

    st.subheader("Model Processing Pipeline")

    st.write("""
    1. Upload CT scan image.
    2. Convert image to RGB.
    3. Resize image to 244 × 244.
    4. Apply MobileNetV2 preprocessing.
    5. Run prediction using the trained Keras model.
    6. Display classification scores.
    7. Generate a research report.
    """)

    st.warning(
        "This dashboard is for educational and research "
        "purposes only. It is not clinically validated."
    )
