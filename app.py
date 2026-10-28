
import io
import hashlib
import textwrap
from pathlib import Path
from datetime import datetime
from xml.sax.saxutils import escape

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LungAI | Intelligent Lung CT Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Lung_Model.h5"

CLASS_LABELS = ["Malignant", "Normal", "benign"]

# ============================================================
# 2. PROJECT LINKS
# ============================================================

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
    ),
    "Project Report": (
        "https://docs.google.com/document/d/"
        "1XAgCHND2BPV4SZxJ1ugP7j8jjOvkO-YH/edit"
        "?usp=sharing&ouid=100880821762278039102"
        "&rtpof=true&sd=true"
    )
}

RESOURCES = [
    (
        "Dataset",
        "Dataset",
        "IQ-OTH/NCCD lung CT image dataset."
    ),
    (
        "Google Colab Notebook",
        "Notebook",
        "Model architecture, training, and evaluation."
    ),
    (
        "GitHub Repository",
        "GitHub",
        "Source code and project implementation."
    ),
    (
        "Live Deployment",
        "Deployment",
        "Deployed Streamlit dashboard."
    ),
    (
        "Project Report",
        "Project Report",
        "Detailed methodology, architecture, and results."
    ),
    (
        "Project Presentation",
        "Presentation",
        "Project slides and research overview."
    ),
    (
        "YouTube Demonstration",
        "YouTube",
        "Recorded project demonstration."
    )
]

# ============================================================
# 3. PROFESSIONAL DARK THEME
# ============================================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(
            ellipse at 85% 0%,
            rgba(29, 91, 113, 0.22),
            transparent 45%
        ),
        linear-gradient(
            145deg,
            #071421,
            #0b2030,
            #0d2938
        );
    color: #eef8fc;
}

[data-testid="stSidebar"] {
    background: #091c2b;
    border-right: 1px solid #28495b;
}

.block-container {
    max-width: 1350px;
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 45px;
    border-radius: 22px;
    background:
        radial-gradient(
            circle at 90% 20%,
            rgba(88, 224, 203, 0.15),
            transparent 35%
        ),
        linear-gradient(
            115deg,
            #102f43,
            #143f51,
            #185363
        );
    border: 1px solid #356477;
    margin-bottom: 25px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.15);
}

.hero::after {
    content: "";
    position: absolute;
    width: 300px;
    height: 300px;
    right: -100px;
    top: -110px;
    border-radius: 50%;
    border: 1px solid rgba(112, 235, 217, 0.18);
    box-shadow:
        0 0 0 45px rgba(112, 235, 217, 0.04),
        0 0 0 90px rgba(112, 235, 217, 0.025);
    pointer-events: none;
}

.hero-eyebrow {
    display: inline-block;
    color: #8af2e1;
    background: rgba(82, 224, 202, 0.10);
    border: 1px solid rgba(82, 224, 202, 0.26);
    padding: 8px 14px;
    border-radius: 30px;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 1.6px;
    margin-bottom: 22px;
}

.hero h1 {
    color: #ffffff;
    font-size: clamp(2.1rem, 4vw, 3.4rem);
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -1.5px;
    margin: 0 0 18px;
}

.hero h1 span {
    display: block;
    color: #75ead7;
    font-size: 0.75em;
    margin-top: 10px;
    letter-spacing: -0.5px;
}

.hero p {
    color: #d2e6ed;
    font-size: 1.05rem;
    line-height: 1.8;
    max-width: 750px;
    margin: 0;
}

.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 26px;
}

.hero-tag {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.13);
    border-radius: 8px;
    padding: 8px 13px;
    color: #e0f0f3;
    font-size: 0.8rem;
    font-weight: 600;
}

.section-label {
    color: #6fe4d1;
    font-size: 0.78rem;
    letter-spacing: 1.5px;
    font-weight: 800;
}

[data-testid="stMetric"] {
    background: linear-gradient(
        140deg,
        #102b3e,
        #14384a
    );
    border: 1px solid #31576b;
    border-radius: 15px;
    padding: 18px;
}

[data-testid="stFileUploader"] {
    background: #102c3d;
    border: 1px dashed #527d90;
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 15px !important;
    border-color: #315367 !important;
}

.stButton button[kind="primary"] {
    background: #55d8c5;
    color: #07232b;
    border: none;
    font-weight: 800;
}

.stButton button,
.stLinkButton a,
.stDownloadButton button {
    border-radius: 10px !important;
    font-weight: 650;
}

.footer {
    color: #a4becb;
    text-align: center;
    padding: 20px;
    font-size: 0.84rem;
    line-height: 1.9;
}

@media (max-width: 768px) {
    .hero {
        padding: 28px 22px;
    }

    .hero h1 {
        letter-spacing: -0.5px;
    }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 4. ORIGINAL WORKING MODEL LOADING
# ============================================================

@st.cache_resource
def load_lung_model():
    # Preserve the model-loading behavior of the original app.
    model = tf.keras.models.load_model(
        str(MODEL_PATH)
    )
    return model

# ============================================================
# 5. ORIGINAL WORKING PREDICTION PIPELINE
# ============================================================

def predict_image(image_bytes):
    """
    Uses the same image processing and prediction
    operations as the original working app.py.

    Do not change the image size or preprocessing
    without checking the original training pipeline.
    """

    model = load_lung_model()

    # Original: Image.open(uploaded_file)
    image = Image.open(io.BytesIO(image_bytes))

    # Original: fixed resize to 244 x 244.
    image = image.resize((244, 244))

    # Original: convert PIL image to NumPy array.
    image_array = np.array(image)

    # Original: remove alpha channel if present.
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]

    # Original: add batch dimension.
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Original: MobileNetV2 preprocessing.
    processed_image = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(image_array)
    )

    # Original: direct model prediction.
    predictions = model.predict(
        processed_image,
        verbose=0
    )

    # Original: direct class selection.
    predicted_class_index = np.argmax(
        predictions,
        axis=1
    )[0]

    # Preserve original model scores.
    probabilities = predictions[0]

    diagnostics = {
        "model_input_shape": str(model.input_shape),
        "image_resize": "244 x 244",
        "preprocessing": "MobileNetV2",
        "raw_output": probabilities.tolist(),
        "predicted_index": int(predicted_class_index),
        "predicted_class": CLASS_LABELS[
            predicted_class_index
        ]
    }

    return probabilities, diagnostics

# ============================================================
# 6. PDF REPORT GENERATION
# ============================================================

def generate_pdf(result):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle
    )

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=45,
        rightMargin=45,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph(
            "LungAI - CT Scan Analysis Report",
            styles["Title"]
        ),
        Spacer(1, 15),
        Paragraph(
            "Deep Learning Research Prototype",
            styles["Heading2"]
        ),
        Spacer(1, 20)
    ]

    details = [
        ("Image File", escape(result["file"])),
        ("Analysis Date", escape(result["time"])),
        ("Predicted Class", escape(result["label"])),
        (
            "Highest Model Score",
            f'{result["confidence"]:.2f}%'
        )
    ]

    for title, value in details:
        story.append(
            Paragraph(
                f"<b>{title}:</b> {value}",
                styles["Normal"]
            )
        )
        story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Classification Scores",
            styles["Heading2"]
        )
    )

    table_data = [
        ["Category", "Model Score"]
    ]

    for label, score in result["scores"].items():
        table_data.append([
            label,
            f"{score:.2f}%"
        ])

    table = Table(
        table_data,
        colWidths=[260, 170]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#123e50")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#cad9df")
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                11
            )
        ])
    )

    story.append(Spacer(1, 10))
    story.append(table)
    story.append(Spacer(1, 25))

    story.append(
        Paragraph(
            "This is an educational research result, "
            "not a medical diagnosis. Model scores "
            "are not calibrated clinical probabilities.",
            styles["Normal"]
        )
    )

    document.build(story)

    return buffer.getvalue()

# ============================================================
# 7. SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

if "last_file" not in st.session_state:
    st.session_state.last_file = None

if "diagnostics" not in st.session_state:
    st.session_state.diagnostics = None

# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("LungAI")

    st.caption(
        "INTELLIGENT LUNG CT ANALYSIS"
    )

    st.divider()

    section = st.radio(
        "Dashboard Navigation",
        [
            "Scan Analyzer",
            "Analysis History",
            "About LungAI",
            "Project Resources"
        ],
        index=0
    )

    st.divider()

    st.subheader("Model Configuration")

    st.write("Architecture: Custom CNN")
    st.write("Framework: TensorFlow / Keras")
    st.write("Classes: 3")
    st.write("Input Processing: 244 x 244")
    st.write("Preprocessing: MobileNetV2")

    st.divider()

    st.subheader("Quick Access")

    for title, key, description in RESOURCES:
        st.link_button(
            title,
            PROJECT_LINKS[key],
            use_container_width=True
        )

    st.divider()

    st.caption(
        "Research prototype. Not intended "
        "for clinical diagnosis."
    )


# ============================================================
# 9. PROFESSIONAL HERO SECTION - FIXED
# ============================================================

hero_html = """
<div class="hero">
<div class="hero-eyebrow">DEEP LEARNING RESEARCH PLATFORM</div>
<h1>LungAI <span>Intelligent Lung CT Analysis</span></h1>
<p>A research-focused medical imaging platform powered by a custom
Convolutional Neural Network. Analyze lung CT images, explore
classification results, and access complete project documentation
through one integrated dashboard.</p>
<div class="hero-tags">
<span class="hero-tag">Custom CNN</span>
<span class="hero-tag">TensorFlow / Keras</span>
<span class="hero-tag">Three-Class Classification</span>
<span class="hero-tag">Academic Research</span>
</div>
</div>
"""

st.markdown(
    hero_html.strip(),
    unsafe_allow_html=True
)


# ============================================================
# 10. PAGE: SCAN ANALYZER
# ============================================================

if section == "Scan Analyzer":

    st.markdown(
        '<div class="section-label">'
        'IMAGE CLASSIFICATION'
        '</div>',
        unsafe_allow_html=True
    )

    st.title("CT Scan Analyzer")

    st.write(
        "Upload a lung CT image to analyze it using "
        "the trained CNN model."
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric("Dataset Images", "1,097")

    with metric2:
        st.metric("Classification Classes", "3")

    with metric3:
        st.metric("Reported Test Accuracy", "97.88%")

    st.caption(
        "Accuracy reported in the original notebook. "
        "The dataset splitting procedure requires "
        "independent validation."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Lung CT Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is None:

        st.info(
            "Upload a CT scan image to begin analysis."
        )

    else:

        image_bytes = uploaded_file.getvalue()

        file_hash = hashlib.sha256(
            image_bytes
        ).hexdigest()

        if st.session_state.last_file != file_hash:
            st.session_state.last_file = file_hash
            st.session_state.result = None
            st.session_state.diagnostics = None

        left, right = st.columns(
            [1, 1],
            gap="large"
        )

        with left:

            st.subheader("CT Scan Preview")

            valid_image = False

            try:

                with Image.open(
                    io.BytesIO(image_bytes)
                ) as image:
                    preview = image.copy()

                st.image(
                    preview,
                    caption=uploaded_file.name,
                    use_container_width=True
                )

                st.caption(
                    f"Original image dimensions: "
                    f"{preview.width} x {preview.height}"
                )

                valid_image = True

            except (
                UnidentifiedImageError,
                OSError,
                ValueError
            ):

                st.error(
                    "Invalid or corrupted image."
                )

        with right:

            st.subheader("CNN Classification")

            if not MODEL_PATH.exists():

                st.error(
                    "Lung_Model.h5 was not found. "
                    "Place the trained model beside app.py."
                )

            elif valid_image:

                if st.button(
                    "Run CNN Analysis",
                    type="primary",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Running model inference..."
                    ):

                        try:

                            (
                                probabilities,
                                diagnostics
                            ) = predict_image(image_bytes)

                            best_index = int(
                                np.argmax(probabilities)
                            )

                            predicted_class = CLASS_LABELS[
                                best_index
                            ]

                            result = {
                                "file": uploaded_file.name,
                                "time": (
                                    datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    )
                                ),
                                "label": predicted_class.title(),
                                "confidence": float(
                                    probabilities[best_index] * 100
                                ),
                                "scores": {
                                    CLASS_LABELS[i].title(): float(
                                        probabilities[i] * 100
                                    )
                                    for i in range(3)
                                }
                            }

                            st.session_state.result = result

                            st.session_state.diagnostics = (
                                diagnostics
                            )

                            st.session_state.history.append(
                                result.copy()
                            )

                        except Exception as error:

                            st.session_state.result = None
                            st.session_state.diagnostics = None

                            st.error(
                                f"Prediction failed: {error}"
                            )

                if st.session_state.result is not None:

                    result = st.session_state.result

                    st.success(
                        "Predicted Class: "
                        + result["label"]
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
                            f"**{label}: {score:.2f}%**"
                        )

                        st.progress(
                            min(
                                max(score / 100.0, 0.0),
                                1.0
                            )
                        )

                    st.caption(
                        "Model scores are not calibrated "
                        "clinical probabilities."
                    )

                    try:

                        pdf_bytes = generate_pdf(
                            result
                        )

                        st.download_button(
                            "Download PDF Analysis Report",
                            data=pdf_bytes,
                            file_name="LungAI_Analysis_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

                    except ImportError:

                        st.warning(
                            "Install reportlab to enable "
                            "PDF report generation."
                        )

                    with st.expander(
                        "Model Diagnostics",
                        expanded=False
                    ):

                        st.json(
                            st.session_state.diagnostics
                        )

                        st.caption(
                            "Class order: "
                            "0 = Malignant, "
                            "1 = Normal, "
                            "2 = Benign."
                        )

    st.divider()

    st.warning(
        "LungAI is an educational research prototype. "
        "Do not use its results for medical diagnosis "
        "or treatment decisions."
    )

# ============================================================
# 11. PAGE: ANALYSIS HISTORY
# ============================================================

elif section == "Analysis History":

    st.markdown(
        '<div class="section-label">'
        'SESSION ACTIVITY'
        '</div>',
        unsafe_allow_html=True
    )

    st.title("Analysis History")

    st.write(
        "Review predictions generated during "
        "your current application session."
    )

    if not st.session_state.history:

        st.info(
            "No images have been analyzed yet."
        )

    else:

        records = []

        for item in st.session_state.history:

            row = {
                "File": item["file"],
                "Date and Time": item["time"],
                "Prediction": item["label"],
                "Highest Score (%)": round(
                    item["confidence"],
                    2
                )
            }

            for label, score in item["scores"].items():
                row[f"{label} (%)"] = round(
                    score,
                    2
                )

            records.append(row)

        history_df = pd.DataFrame(records)

        st.metric(
            "Total Analyses",
            len(history_df)
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                "Export History as CSV",
                data=history_df.to_csv(index=False),
                file_name="LungAI_Analysis_History.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:

            if st.button(
                "Clear History",
                use_container_width=True
            ):

                st.session_state.history = []
                st.session_state.result = None
                st.session_state.diagnostics = None

                st.rerun()

    st.caption(
        "Analysis history is stored only in "
        "the current Streamlit session."
    )

# ============================================================
# 12. PAGE: ABOUT LUNGAI
# ============================================================

elif section == "About LungAI":

    st.markdown(
        '<div class="section-label">'
        'PROJECT OVERVIEW'
        '</div>',
        unsafe_allow_html=True
    )

    st.title("About LungAI")

    st.write(
        "LungAI is a Deep Learning semester project "
        "designed to classify lung CT scan images "
        "using a custom Convolutional Neural Network "
        "developed with TensorFlow and Keras."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset Images",
            "1,097"
        )

    with col2:
        st.metric(
            "Classification Categories",
            "3"
        )

    with col3:
        st.metric(
            "Reported Test Accuracy",
            "97.88%"
        )

    st.subheader("Problem Statement")

    st.write(
        "Lung CT scans contain complex visual patterns. "
        "This project explores the use of CNN-based "
        "feature learning to distinguish between "
        "Malignant, Normal, and Benign image categories."
    )

    st.subheader("Dataset Information")

    st.write(
        "Dataset: IQ-OTH/NCCD Lung Cancer CT Dataset"
    )

    dataset_df = pd.DataFrame({
        "Category": [
            "Malignant",
            "Normal",
            "Benign"
        ],
        "Images": [
            561,
            416,
            120
        ]
    })

    st.dataframe(
        dataset_df,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        dataset_df.set_index("Category")
    )

    st.subheader("CNN Architecture")

    st.write(
        "The project uses a custom CNN with "
        "convolutional layers, batch normalization, "
        "max pooling, dense layers, dropout, "
        "and a three-class softmax output."
    )

    architecture_df = pd.DataFrame({
        "Parameter": [
            "Framework",
            "Architecture",
            "Optimizer",
            "Learning Rate",
            "Loss Function",
            "Epochs",
            "Batch Size",
            "Input Processing",
            "Preprocessing",
            "Output Activation"
        ],
        "Value": [
            "TensorFlow / Keras",
            "Custom CNN",
            "SGD",
            "0.001",
            "Categorical Cross-Entropy",
            "10",
            "4",
            "244 x 244",
            "MobileNetV2",
            "Softmax"
        ]
    })

    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Model Processing Pipeline")

    st.markdown("""
1. Upload a lung CT scan image.
2. Resize the image to 244 x 244 pixels.
3. Convert the image into a NumPy array.
4. Remove the alpha channel if present.
5. Apply MobileNetV2 preprocessing.
6. Run inference using the trained CNN.
7. Identify the highest-scoring class.
8. Display the classification scores.
9. Generate a PDF analysis report.
""")

    st.subheader("Reported Model Results")

    st.write(
        "The original notebook reports approximately "
        "97.88% test accuracy after 10 training epochs. "
        "This result is preliminary because the original "
        "training and test splits may overlap."
    )

    st.subheader("Project Documentation")

    col1, col2 = st.columns(2)

    with col1:

        st.link_button(
            "Read Project Report",
            PROJECT_LINKS["Project Report"],
            use_container_width=True
        )

        st.link_button(
            "Open Training Notebook",
            PROJECT_LINKS["Notebook"],
            use_container_width=True
        )

    with col2:

        st.link_button(
            "View Presentation",
            PROJECT_LINKS["Presentation"],
            use_container_width=True
        )

        st.link_button(
            "View GitHub Repository",
            PROJECT_LINKS["GitHub"],
            use_container_width=True
        )

    st.warning(
        "This project has not been clinically validated."
    )

# ============================================================
# 13. PAGE: PROJECT RESOURCES
# ============================================================

elif section == "Project Resources":

    st.markdown(
        '<div class="section-label">'
        'RESEARCH MATERIALS'
        '</div>',
        unsafe_allow_html=True
    )

    st.title("Project Resources")

    st.write(
        "Access all project materials, including "
        "the dataset, notebook, source code, "
        "deployment, report, presentation, "
        "and video demonstration."
    )

    st.divider()

    for start in range(0, len(RESOURCES), 2):

        columns = st.columns(
            2,
            gap="large"
        )

        for offset, column in enumerate(columns):

            index = start + offset

            if index >= len(RESOURCES):
                continue

            title, key, description = RESOURCES[index]

            with column:

                with st.container(border=True):

                    st.subheader(title)

                    st.write(description)

                    st.link_button(
                        "Open Resource",
                        PROJECT_LINKS[key],
                        use_container_width=True
                    )

    st.divider()

    st.subheader("Video Demonstration")

    st.video(
        PROJECT_LINKS["YouTube"]
    )

    st.divider()

    st.subheader("GitHub Repository")

    st.code(
        PROJECT_LINKS["GitHub"],
        language=None
    )

    st.caption(
        "Ensure that externally hosted documents "
        "are accessible to your project evaluator."
    )

# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

footer_html = textwrap.dedent("""
<div class="footer">
    <strong>LungAI</strong>
    | Intelligent Lung CT Analysis
    <br>
    Deep Learning Semester Project
    <br>
    Built with Python, TensorFlow, Keras and Streamlit
    <br>
    Educational and Research Use Only
</div>
""").strip()

st.markdown(
    footer_html,
    unsafe_allow_html=True
)
