
import io
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="LungAI | Lung Cancer Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Lung_Model.h5"

# IMPORTANT: This order must match the training notebook.
CLASS_LABELS = ["Malignant", "Normal", "benign"]

# --------------------------------------------------
# PROJECT LINKS
# --------------------------------------------------

PROJECT_LINKS = {
    "Dataset": (
        "https://drive.google.com/drive/folders/"
        "1y9Fbp7loKAUFjUHMzMjLe1xWzoYOpYCn?usp=sharing"
    ),
    "Notebook": (
        "https://colab.research.google.com/drive/"
        "1FMufGBdUeYz2mTFQhaEqF5LYU5xNElTm?usp=sharing"
    ),
    "GitHub": (
        "https://github.com/umerpervaiz2478/LungCancerCNN/"
    ),
    "Deployment": (
        "https://lungcancercnn.streamlit.app/"
    ),
    "YouTube": (
        "https://youtu.be/M5Emf1WuQlM"
    ),
    "Presentation": (
        "https://docs.google.com/presentation/d/"
        "1vw5HB88hDIkM7C_iTH9mIAvWOv_ql7Vq/edit"
        "?usp=sharing&ouid=100880821762278039102"
        "&rtpof=true&sd=true"
    )
}

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

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
    max-width: 1300px;
    padding-top: 2rem;
}

.hero {
    background: linear-gradient(110deg, #124558, #132c43);
    border: 1px solid #2b6275;
    border-radius: 18px;
    padding: 30px;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 2.4rem;
    margin: 10px 0;
}

.hero p {
    color: #c5dce8;
    font-size: 1.05rem;
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

.resource-title {
    color: #69e7d3;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #9bb7c7;
    padding: 25px;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading trained AI model...")
def load_lung_model():
    import tensorflow as tf

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


# --------------------------------------------------
# MODEL PREDICTION
# --------------------------------------------------

def predict_image(image_bytes):
    import tensorflow as tf

    model = load_lung_model()

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height, width = input_shape[1], input_shape[2]

    if height is None or width is None:
        raise ValueError(
            "Model input dimensions are undefined."
        )

    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image = image.resize((int(width), int(height)))
        image_array = np.asarray(
            image,
            dtype=np.float32
        )

    image_array = np.expand_dims(image_array, axis=0)

    # Match preprocessing with the training notebook.
    # If the model was trained with rescale=1./255,
    # use this normalization instead of MobileNetV2.
    processed = image_array / 255.0

    output = model.predict(processed, verbose=0)

    scores = np.asarray(
        output,
        dtype=float
    ).reshape(-1)

    if len(scores) != len(CLASS_LABELS):
        raise ValueError(
            "Model output does not match the three classes."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Model returned invalid prediction values."
        )

    if (
        np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(scores.sum(), 1.0, atol=0.02)
    ):
        scores = np.exp(scores - np.max(scores))

    scores = scores / scores.sum()

    return scores


# --------------------------------------------------
# GENERATE PDF REPORT
# --------------------------------------------------

def generate_pdf(result):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer
    )
    from reportlab.lib.styles import getSampleStyleSheet
    from xml.sax.saxutils import escape

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph(
            "LungAI - CT Scan Classification Report",
            styles["Title"]
        ),
        Spacer(1, 20)
    ]

    lines = [
        "Image: " + escape(result["file"]),
        "Date: " + result["time"],
        "Predicted Class: " + result["label"],
        (
            "Highest Model Score: "
            f'{result["confidence"]:.2f}%'
        )
    ]

    for label, score in result["scores"].items():
        lines.append(
            f"{label.title()}: {score:.2f}%"
        )

    lines.append(
        "Disclaimer: This report is generated by an "
        "educational AI model and is not a medical diagnosis."
    )

    for line in lines:
        story.append(
            Paragraph(line, styles["Normal"])
        )
        story.append(Spacer(1, 12))

    document.build(story)

    return buffer.getvalue()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

if "last_file" not in st.session_state:
    st.session_state.last_file = None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🫁 LungAI")
    st.caption("AI-POWERED LUNG CT ANALYSIS")

    st.divider()

    section = st.radio(
        "Dashboard Navigation",
        [
            "🔬 Scan Analyzer",
            "📊 Analysis History",
            "ℹ️ About LungAI",
            "🔗 Project Resources"
        ],
        index=0
    )

    st.divider()

    st.subheader("Model Information")

    st.write("Architecture: CNN")
    st.write("Framework: TensorFlow / Keras")
    st.write("Classes: 3")
    st.write("Model: Lung_Model.h5")

    st.divider()

    st.link_button(
        "💻 GitHub Repository",
        PROJECT_LINKS["GitHub"],
        use_container_width=True
    )

    st.link_button(
        "🎥 Watch Demonstration",
        PROJECT_LINKS["YouTube"],
        use_container_width=True
    )

    st.warning(
        "Research prototype. Not for clinical diagnosis."
    )


# --------------------------------------------------
# HERO HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">
        DEEP LEARNING · MEDICAL IMAGING
    </div>

    <h1>🫁 LungAI - Lung Cancer Detection</h1>

    <p>
        AI-powered lung CT scan classification using
        a trained Convolutional Neural Network.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SCAN ANALYZER PAGE
# --------------------------------------------------

if section == "🔬 Scan Analyzer":

    st.subheader("🔬 CT Scan Analyzer")

    st.write(
        "Upload a lung CT scan image to classify it "
        "as Malignant, Normal, or Benign."
    )

    uploaded_file = st.file_uploader(
        "Upload CT Scan",
        type=["png", "jpg", "jpeg"],
        key="lung_upload"
    )

    if uploaded_file is None:
        st.info(
            "Upload a PNG, JPG, or JPEG CT scan image "
            "to begin analysis."
        )

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

        col1, col2 = st.columns(
            [1, 1],
            gap="large"
        )

        with col1:

            st.subheader("CT Scan Preview")

            try:
                image = Image.open(
                    io.BytesIO(image_bytes)
                )

                image = image.convert("RGB")

                st.image(
                    image,
                    caption=uploaded_file.name,
                    use_container_width=True
                )

                valid_image = True

            except (
                UnidentifiedImageError,
                OSError
            ):
                st.error(
                    "The uploaded image is invalid or corrupted."
                )

                valid_image = False

        with col2:

            st.subheader("AI Classification")

            if not MODEL_PATH.exists():

                st.error(
                    "Lung_Model.h5 was not found. "
                    "Place your trained model beside app.py."
                )

            elif valid_image:

                if st.button(
                    "🚀 Analyze CT Scan",
                    type="primary",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing CT scan..."
                    ):

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
                        "Predicted Class: "
                        + result["label"].title()
                    )

                    st.metric(
                        "Highest Model Score",
                        f'{result["confidence"]:.2f}%'
                    )

                    st.subheader(
                        "Classification Scores"
                    )

                    for label, score in result["scores"].items():

                        st.write(
                            f"**{label.title()}: {score:.2f}%**"
                        )

                        st.progress(
                            min(
                                max(score / 100, 0.0),
                                1.0
                            )
                        )

                    try:

                        pdf_bytes = generate_pdf(
                            result
                        )

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
        "This AI model is intended for educational "
        "research and is not clinically validated."
    )


# --------------------------------------------------
# ANALYSIS HISTORY PAGE
# --------------------------------------------------

elif section == "📊 Analysis History":

    st.subheader("📊 Analysis History")

    st.write(
        "Review CT scan classification results "
        "generated during the current session."
    )

    if not st.session_state.history:

        st.info(
            "No scans have been analyzed yet."
        )

    else:

        import pandas as pd

        records = []

        for item in st.session_state.history:

            row = {
                "File": item["file"],
                "Time": item["time"],
                "Prediction": item["label"].title(),
                "Highest Score (%)": round(
                    item["confidence"],
                    2
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

        if st.button("🗑️ Clear History"):

            st.session_state.history = []
            st.session_state.result = None
            st.rerun()


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif section == "ℹ️ About LungAI":

    st.title("ℹ️ About LungAI")

    st.write("""
    LungAI is a Deep Learning semester project
    designed to classify lung CT scan images using
    a Convolutional Neural Network (CNN).

    The system processes uploaded CT images and
    generates classification scores for three categories.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Classification Categories",
            "03"
        )

    with col2:
        st.metric(
            "Dataset Images",
            "1,097"
        )

    with col3:
        st.metric(
            "Reported Test Accuracy",
            "97.88%"
        )

    st.caption(
        "The reported accuracy comes from the training "
        "notebook and requires independent validation."
    )

    st.subheader("Classification Categories")

    st.write(
        "**Malignant:** Dataset category for malignant lesions."
    )

    st.write(
        "**Normal:** Dataset category for normal lung scans."
    )

    st.write(
        "**Benign:** Dataset category for benign lesions."
    )

    st.subheader("Dataset Information")

    st.write(
        "Dataset: IQ-OTH/NCCD Lung Cancer Dataset"
    )

    st.write(
        "Total images: 1,097"
    )

    st.write(
        "Classes: Malignant, Normal, Benign"
    )

    st.subheader("Model Architecture")

    st.write(
        "Custom Convolutional Neural Network (CNN)"
    )

    st.write(
        "Framework: TensorFlow / Keras"
    )

    st.write(
        "Output: Three-class classification"
    )

    st.subheader("Processing Pipeline")

    st.markdown("""
    1. Upload a lung CT scan image.
    2. Convert the image to RGB.
    3. Resize it to the model's expected input dimensions.
    4. Normalize the image values.
    5. Run inference through the CNN.
    6. Display the predicted class and scores.
    7. Generate a downloadable analysis report.
    """)

    st.subheader("Project Documentation")

    st.link_button(
        "📓 Open Training Notebook",
        PROJECT_LINKS["Notebook"]
    )

    st.link_button(
        "💻 View Source Code",
        PROJECT_LINKS["GitHub"]
    )

    st.link_button(
        "📊 View Presentation",
        PROJECT_LINKS["Presentation"]
    )

    st.warning(
        "LungAI is an educational research project "
        "and must not be used for clinical decisions."
    )


# --------------------------------------------------
# PROJECT RESOURCES PAGE
# --------------------------------------------------

elif section == "🔗 Project Resources":

    st.title("🔗 LungAI Project Resources")

    st.write(
        "Access the dataset, source code, training "
        "notebook, deployed application, project "
        "presentation, and YouTube demonstration."
    )

    st.divider()

    resources = [
        {
            "title": "📂 Dataset",
            "description": (
                "IQ-OTH/NCCD lung CT scan dataset "
                "used for training and evaluation."
            ),
            "url": PROJECT_LINKS["Dataset"]
        },
        {
            "title": "📓 Google Colab Notebook",
            "description": (
                "Complete model training, preprocessing, "
                "architecture, and evaluation notebook."
            ),
            "url": PROJECT_LINKS["Notebook"]
        },
        {
            "title": "💻 GitHub Repository",
            "description": (
                "LungAI source code, model integration, "
                "and project implementation."
            ),
            "url": PROJECT_LINKS["GitHub"]
        },
        {
            "title": "🌐 Live Deployment",
            "description": (
                "Access the deployed LungAI "
                "Streamlit web application."
            ),
            "url": PROJECT_LINKS["Deployment"]
        },
        {
            "title": "🎥 YouTube Demonstration",
            "description": (
                "Watch the project presentation "
                "and demonstration video."
            ),
            "url": PROJECT_LINKS["YouTube"]
        },
        {
            "title": "📊 Project Presentation",
            "description": (
                "Deep Learning semester project "
                "PowerPoint presentation."
            ),
            "url": PROJECT_LINKS["Presentation"]
        }
    ]

    for resource in resources:

        with st.container(border=True):

            st.subheader(
                resource["title"]
            )

            st.write(
                resource["description"]
            )

            st.link_button(
                "Open Resource ↗",
                resource["url"],
                use_container_width=True
            )

    st.divider()

    st.subheader("🎥 Project Video Demonstration")

    st.video(
        PROJECT_LINKS["YouTube"]
    )

    st.subheader("Project Repository")

    st.code(
        PROJECT_LINKS["GitHub"],
        language=None
    )

    st.info(
        "All project resources are provided "
        "for academic evaluation and demonstration."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown("""
<div class="footer">
    LungAI | Deep Learning Semester Project<br>
    Powered by TensorFlow, Keras and Streamlit<br>
    For Educational and Research Purposes Only
</div>
""", unsafe_allow_html=True)
