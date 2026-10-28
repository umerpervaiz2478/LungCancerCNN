
import io
import hashlib
from pathlib import Path
from datetime import datetime
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LungAI | Intelligent Lung CT Analysis",
    page_icon="L",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Lung_Model.h5"

# Exact class order from the original training notebook.
CLASS_LABELS = ["Malignant", "Normal", "benign"]

CLASS_DISPLAY = {
    "Malignant": "Malignant",
    "Normal": "Normal",
    "benign": "Benign",
}

# The original notebook uses MobileNetV2 preprocessing
# with a custom CNN architecture.
PREPROCESSING = "mobilenet_v2"

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
    ),
}

RESOURCE_DETAILS = [
    {
        "title": "Dataset",
        "key": "Dataset",
        "description": (
            "IQ-OTH/NCCD lung CT image dataset used "
            "for model development and evaluation."
        ),
        "category": "DATA",
    },
    {
        "title": "Training Notebook",
        "key": "Notebook",
        "description": (
            "Google Colab notebook containing CNN architecture, "
            "preprocessing, training, and evaluation."
        ),
        "category": "DEVELOPMENT",
    },
    {
        "title": "GitHub Repository",
        "key": "GitHub",
        "description": (
            "Project source code, application implementation, "
            "and associated repository files."
        ),
        "category": "SOURCE CODE",
    },
    {
        "title": "Live Application",
        "key": "Deployment",
        "description": (
            "Public Streamlit deployment of the "
            "LungAI research dashboard."
        ),
        "category": "DEPLOYMENT",
    },
    {
        "title": "Project Report",
        "key": "Project Report",
        "description": (
            "Comprehensive documentation of the problem, "
            "dataset, architecture, methodology, and results."
        ),
        "category": "DOCUMENTATION",
    },
    {
        "title": "Project Presentation",
        "key": "Presentation",
        "description": (
            "Presentation covering the project workflow, "
            "implementation, and research findings."
        ),
        "category": "PRESENTATION",
    },
    {
        "title": "Video Demonstration",
        "key": "YouTube",
        "description": (
            "Recorded explanation and demonstration "
            "of the LungAI project."
        ),
        "category": "VIDEO",
    },
]

# ============================================================
# 3. PROFESSIONAL THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #071421;
        --panel: #102638;
        --panel-2: #143247;
        --border: #29485a;
        --text: #f1f7fa;
        --muted: #a9c0ce;
        --accent: #56d8c4;
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at 85% 0%,
                rgba(27, 91, 112, 0.22),
                transparent 42%
            ),
            linear-gradient(
                145deg,
                #071421 0%,
                #0a1b2b 60%,
                #0c2533 100%
            );
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: #091c2b;
        border-right: 1px solid #234052;
    }

    .block-container {
        max-width: 1390px;
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 48px 46px;
        border-radius: 24px;
        background:
            radial-gradient(
                circle at 90% 30%,
                rgba(63, 212, 194, 0.16),
                transparent 30%
            ),
            linear-gradient(
                115deg,
                #102f43 0%,
                #123e51 55%,
                #164758 100%
            );
        border: 1px solid #326175;
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.16);
        margin-bottom: 24px;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 310px;
        height: 310px;
        border: 1px solid rgba(113, 224, 210, 0.16);
        border-radius: 50%;
        right: -85px;
        top: -100px;
        box-shadow:
            0 0 0 48px rgba(113, 224, 210, 0.035),
            0 0 0 96px rgba(113, 224, 210, 0.025);
        pointer-events: none;
    }

    .hero-eyebrow {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 100px;
        background: rgba(82, 224, 202, 0.10);
        border: 1px solid rgba(82, 224, 202, 0.25);
        color: #8af2e1;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 1.6px;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: clamp(2.2rem, 4.5vw, 3.6rem);
        font-weight: 800;
        line-height: 1.12;
        letter-spacing: -1.7px;
        margin: 0 0 16px;
        max-width: 760px;
    }

    .hero h1 span {
        color: #75ead7;
    }

    .hero p {
        color: #d3e7ee;
        max-width: 720px;
        font-size: 1.04rem;
        line-height: 1.85;
        margin: 0;
    }

    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 26px;
    }

    .hero-tag {
        padding: 7px 13px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.075);
        border: 1px solid rgba(255, 255, 255, 0.13);
        color: #dceef2;
        font-size: 0.79rem;
        font-weight: 600;
    }

    .section-kicker {
        color: #6fe4d1;
        font-size: 0.76rem;
        letter-spacing: 1.7px;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .section-description {
        color: #adc5d0;
        font-size: 0.95rem;
        line-height: 1.7;
        margin-bottom: 18px;
    }

    .small-note {
        color: #9cb5c2;
        font-size: 0.83rem;
        line-height: 1.65;
    }

    .resource-category {
        color: #6fe4d1;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 1.2px;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(
            140deg,
            #102c3e,
            #12384a
        );
        border: 1px solid #31576b;
        border-radius: 16px;
        padding: 17px 19px;
        min-height: 115px;
    }

    [data-testid="stMetricLabel"] {
        color: #bdd3dd;
    }

    [data-testid="stFileUploader"] {
        background: #102b3c;
        border: 1px dashed #527c8e;
        border-radius: 16px;
        padding: 12px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #315367 !important;
        border-radius: 16px !important;
        background: rgba(15, 41, 57, 0.55);
    }

    .stButton button[kind="primary"] {
        background: #55d8c5;
        color: #07232b;
        border: none;
        font-weight: 800;
    }

    .stButton button,
    .stDownloadButton button,
    .stLinkButton a {
        border-radius: 10px !important;
        font-weight: 650;
    }

    .footer {
        text-align: center;
        padding: 25px 10px 12px;
        color: #9db7c3;
        font-size: 0.82rem;
        line-height: 1.9;
    }

    .footer strong {
        color: #b9f4e8;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 30px 24px;
            border-radius: 18px;
        }

        .hero h1 {
            letter-spacing: -0.8px;
        }
        .hero h1 span {
    display: block;
    color: #75ead7;
    margin-top: 8px;
}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 4. MODEL LOADING
# ============================================================

@st.cache_resource(
    show_spinner="Loading trained CNN model..."
)
def load_lung_model():
    import tensorflow as tf

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Lung_Model.h5 is missing from the application directory."
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )


def get_model_input_dimensions(model):
    shape = model.input_shape

    if isinstance(shape, list):
        if len(shape) != 1:
            raise ValueError(
                "The dashboard supports single-input models only."
            )
        shape = shape[0]

    if len(shape) != 4:
        raise ValueError(
            "Expected model input shape: "
            "(batch, height, width, channels)."
        )

    height, width, channels = shape[1:4]

    if height is None or width is None:
        raise ValueError(
            "The saved model does not define fixed image dimensions."
        )

    if channels != 3:
        raise ValueError(
            "The saved model must accept RGB images."
        )

    return int(height), int(width)


# ============================================================
# 5. IMAGE PREPROCESSING AND PREDICTION
# ============================================================

def predict_image(image_bytes):
    import tensorflow as tf

    model = load_lung_model()

    height, width = get_model_input_dimensions(model)

    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image = image.resize(
            (width, height),
            Image.Resampling.BILINEAR,
        )

        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

    image_array = np.expand_dims(
        image_array,
        axis=0,
    )

    # IMPORTANT:
    # The training notebook uses MobileNetV2 preprocessing.
    # This converts pixel values from [0, 255] to [-1, 1].
    #
    # The architecture is still a custom CNN, not MobileNetV2.
    processed = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(image_array.copy())
    )

    output = model.predict(
        processed,
        verbose=0,
    )

    raw_scores = np.asarray(
        output,
        dtype=np.float64,
    ).reshape(-1)

    if raw_scores.size != 3:
        raise ValueError(
            "Expected three output scores: "
            "Malignant, Normal, and benign."
        )

    if not np.all(np.isfinite(raw_scores)):
        raise ValueError(
            "The model returned non-finite prediction scores."
        )

    # The notebook's final layer uses softmax.
    # Normally the model already returns probabilities.
    if (
        np.any(raw_scores < 0)
        or np.any(raw_scores > 1)
        or not np.isclose(
            raw_scores.sum(),
            1.0,
            atol=0.02,
        )
    ):
        shifted = raw_scores - np.max(raw_scores)
        exp_values = np.exp(shifted)
        probabilities = exp_values / exp_values.sum()
    else:
        probabilities = raw_scores / raw_scores.sum()

    diagnostics = {
        "model_input_shape": str(model.input_shape),
        "preprocessing": "MobileNetV2 [-1, 1]",
        "processed_min": float(processed.min()),
        "processed_max": float(processed.max()),
        "raw_output": raw_scores.tolist(),
    }

    return probabilities, diagnostics


# ============================================================
# 6. PDF REPORT GENERATION
# ============================================================

def generate_pdf(result):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib import colors

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=50,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    story = [
        Paragraph(
            "LungAI | CT Scan Classification Report",
            styles["Title"],
        ),
        Spacer(1, 12),
        Paragraph(
            "Deep Learning Research Prototype",
            styles["Heading2"],
        ),
        Spacer(1, 18),
    ]

    details = [
        ("Image File", escape(result["file"])),
        ("Analysis Date", escape(result["time"])),
        ("Predicted Class", escape(result["label"])),
        (
            "Highest Model Score",
            f'{result["confidence"]:.2f}%',
        ),
    ]

    for label, value in details:
        story.append(
            Paragraph(
                f"<b>{label}:</b> {value}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 11))

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Classification Scores",
            styles["Heading2"],
        )
    )

    table_data = [
        ["Category", "Model Score"]
    ]

    for label, score in result["scores"].items():
        table_data.append([
            label,
            f"{score:.2f}%",
        ])

    table = Table(
        table_data,
        colWidths=[270, 170],
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#123e50"),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor("#cad9df"),
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#edf5f7"),
                ],
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                11,
            ),
        ])
    )

    story.append(table)
    story.append(Spacer(1, 24))

    story.append(
        Paragraph(
            "Research Disclaimer",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            "This report is generated by an educational "
            "deep learning model. It is not a clinical "
            "diagnosis or a substitute for professional "
            "medical evaluation. Model scores are not "
            "calibrated clinical probabilities.",
            styles["Normal"],
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
            "Project Resources",
        ],
        index=0,
    )

    st.divider()

    st.subheader("Model Configuration")

    st.write("Architecture: Custom CNN")
    st.write("Framework: TensorFlow / Keras")
    st.write("Output Classes: 3")
    st.write("Preprocessing: MobileNetV2")
    st.write("Model Format: Keras H5")

    st.divider()

    st.subheader("Quick Access")

    quick_links = [
        ("Dataset", "Dataset"),
        ("Google Colab Notebook", "Notebook"),
        ("GitHub Repository", "GitHub"),
        ("Live Deployment", "Deployment"),
        ("Project Report", "Project Report"),
        ("Project Presentation", "Presentation"),
        ("YouTube Demonstration", "YouTube"),
    ]

    for title, key in quick_links:
        st.link_button(
            title,
            PROJECT_LINKS[key],
            use_container_width=True,
        )

    st.divider()

    st.caption(
        "Research prototype only. "
        "Not intended for clinical diagnosis."
    )


# ============================================================
# 9. PROFESSIONAL HERO SECTION - FIXED
# ============================================================

import textwrap

hero_html = textwrap.dedent("""
<div class="hero">
    <div class="hero-eyebrow">
        DEEP LEARNING RESEARCH PLATFORM
    </div>

    <h1>
        LungAI
        <span>Intelligent Lung CT Analysis</span>
    </h1>

    <p>
        An intelligent medical imaging research platform
        powered by a custom Convolutional Neural Network.
        Analyze lung CT scans, explore classification
        results, and access complete research documentation
        through one integrated dashboard.
    </p>

    <div class="hero-tags">
        <span class="hero-tag">
            Custom CNN
        </span>
        <span class="hero-tag">
            TensorFlow / Keras
        </span>
        <span class="hero-tag">
            Three-Class Classification
        </span>
        <span class="hero-tag">
            Academic Research
        </span>
    </div>
</div>
""").strip()

st.markdown(
    hero_html,
    unsafe_allow_html=True
)

# ============================================================
# 10. PAGE: SCAN ANALYZER
# ============================================================

if section == "Scan Analyzer":

    st.markdown(
        '<div class="section-kicker">'
        'IMAGE CLASSIFICATION'
        '</div>',
        unsafe_allow_html=True,
    )

    st.title("CT Scan Analyzer")

    st.markdown(
        '<div class="section-description">'
        'Upload a lung CT image to run inference using '
        'the trained CNN. Review the predicted category, '
        'model scores, and analysis details.'
        '</div>',
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            "Dataset Images",
            "1,097",
        )

    with metric2:
        st.metric(
            "Classification Categories",
            "3",
        )

    with metric3:
        st.metric(
            "Reported Test Accuracy",
            "97.88%",
        )

    st.caption(
        "The reported test accuracy is from the original "
        "notebook and is not an independently validated "
        "generalization estimate because the dataset "
        "splits may overlap."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Lung CT Image",
        type=["png", "jpg", "jpeg"],
        help=(
            "Supported image formats: PNG, JPG, JPEG. "
            "Use research images rather than identifiable "
            "patient records."
        ),
    )

    if uploaded_file is None:
        st.info(
            "Upload a CT scan image to begin model analysis."
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
            gap="large",
        )

        with left:
            st.subheader("Image Preview")

            valid_image = False

            try:
                with Image.open(
                    io.BytesIO(image_bytes)
                ) as image:
                    preview = image.convert("RGB")

                st.image(
                    preview,
                    caption=uploaded_file.name,
                    use_container_width=True,
                )

                st.caption(
                    f"Image dimensions: "
                    f"{preview.width} x {preview.height}"
                )

                valid_image = True

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
            ):
                st.error(
                    "The uploaded file is not a valid image."
                )

        with right:
            st.subheader("Model Analysis")

            if not MODEL_PATH.is_file():
                st.error(
                    "The trained model file Lung_Model.h5 "
                    "was not found. Add it to the application "
                    "directory before running predictions."
                )

            elif valid_image:
                if st.button(
                    "Run CNN Analysis",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner(
                        "Processing image and running inference..."
                    ):
                        try:
                            (
                                probabilities,
                                diagnostics,
                            ) = predict_image(image_bytes)

                            best_index = int(
                                np.argmax(probabilities)
                            )

                            predicted_label = CLASS_DISPLAY[
                                CLASS_LABELS[best_index]
                            ]

                            result = {
                                "file": uploaded_file.name,
                                "time": (
                                    datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    )
                                ),
                                "label": predicted_label,
                                "confidence": float(
                                    probabilities[best_index] * 100
                                ),
                                "scores": {
                                    CLASS_DISPLAY[
                                        CLASS_LABELS[i]
                                    ]: float(
                                        probabilities[i] * 100
                                    )
                                    for i in range(3)
                                },
                            }

                            st.session_state.result = result
                            st.session_state.diagnostics = diagnostics

                            st.session_state.history.append(
                                result.copy()
                            )

                        except Exception as error:
                            st.session_state.result = None
                            st.session_state.diagnostics = None

                            st.error(
                                f"Model inference failed: {error}"
                            )

                if st.session_state.result is not None:
                    result = st.session_state.result

                    st.success(
                        "Predicted Class: "
                        + result["label"]
                    )

                    st.metric(
                        "Highest Model Score",
                        f'{result["confidence"]:.2f}%',
                    )

                    st.subheader(
                        "Classification Score Distribution"
                    )

                    for label, score in result["scores"].items():
                        st.write(
                            f"**{label}: {score:.2f}%**"
                        )

                        st.progress(
                            min(
                                max(score / 100.0, 0.0),
                                1.0,
                            )
                        )

                    st.caption(
                        "These values are model outputs, "
                        "not clinical probabilities."
                    )

                    try:
                        pdf_bytes = generate_pdf(result)

                        st.download_button(
                            "Download PDF Analysis Report",
                            data=pdf_bytes,
                            file_name="LungAI_Analysis_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )

                    except ImportError:
                        st.warning(
                            "Install reportlab to enable "
                            "PDF report generation."
                        )

                    with st.expander(
                        "Model Diagnostics",
                        expanded=False,
                    ):
                        st.write(
                            "Use this information to investigate "
                            "unexpected or repeated predictions."
                        )

                        st.json(
                            st.session_state.diagnostics
                        )

                        st.caption(
                            "Output index order: "
                            "0 = Malignant, "
                            "1 = Normal, "
                            "2 = Benign."
                        )

    st.divider()

    st.warning(
        "LungAI is an educational research prototype. "
        "Do not use its predictions for diagnosis, "
        "screening, or treatment decisions."
    )

# ============================================================
# 11. PAGE: ANALYSIS HISTORY
# ============================================================

elif section == "Analysis History":

    st.markdown(
        '<div class="section-kicker">'
        'SESSION ACTIVITY'
        '</div>',
        unsafe_allow_html=True,
    )

    st.title("Analysis History")

    st.write(
        "Review the CT scan predictions generated "
        "during the current application session."
    )

    if not st.session_state.history:
        st.info(
            "No images have been analyzed in this session."
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
                    2,
                ),
            }

            for label, score in item["scores"].items():
                row[f"{label} (%)"] = round(
                    score,
                    2,
                )

            records.append(row)

        history_df = pd.DataFrame(records)

        st.metric(
            "Total Analyses",
            len(history_df),
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "Export History as CSV",
                data=history_df.to_csv(index=False),
                file_name="LungAI_Analysis_History.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col2:
            if st.button(
                "Clear History",
                use_container_width=True,
            ):
                st.session_state.history = []
                st.session_state.result = None
                st.session_state.diagnostics = None
                st.rerun()

    st.caption(
        "History is stored in Streamlit session state. "
        "It is not a permanent medical record."
    )

# ============================================================
# 12. PAGE: ABOUT LUNGAI
# ============================================================

elif section == "About LungAI":

    st.markdown(
        '<div class="section-kicker">'
        'PROJECT OVERVIEW'
        '</div>',
        unsafe_allow_html=True,
    )

    st.title("About LungAI")

    st.write(
        "LungAI is a Deep Learning semester project "
        "that investigates the classification of lung "
        "CT scan images using a custom Convolutional "
        "Neural Network developed with TensorFlow and Keras."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Dataset Images",
            "1,097",
        )

    with col2:
        st.metric(
            "Classification Classes",
            "3",
        )

    with col3:
        st.metric(
            "Reported Test Accuracy",
            "97.88%",
        )

    st.subheader("Problem Statement")

    st.write(
        "Lung CT images contain complex patterns that "
        "can be challenging to distinguish visually. "
        "This project explores how a CNN can learn "
        "image features and classify scans into "
        "Malignant, Normal, and Benign categories."
    )

    st.subheader("Dataset")

    st.write(
        "The project uses the IQ-OTH/NCCD lung "
        "cancer CT image dataset."
    )

    dataset_df = pd.DataFrame({
        "Category": [
            "Malignant",
            "Normal",
            "Benign",
        ],
        "Images": [
            561,
            416,
            120,
        ],
    })

    st.dataframe(
        dataset_df,
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        dataset_df.set_index("Category")
    )

    st.subheader("CNN Architecture")

    st.write(
        "The notebook defines a custom CNN containing "
        "convolutional layers, batch normalization, "
        "max-pooling, fully connected layers, dropout, "
        "and a three-unit softmax output layer."
    )

    architecture_df = pd.DataFrame({
        "Parameter": [
            "Framework",
            "Architecture",
            "Optimizer",
            "Learning Rate",
            "Loss Function",
            "Training Epochs",
            "Batch Size",
            "Output Classes",
            "Preprocessing",
        ],
        "Value": [
            "TensorFlow / Keras",
            "Custom CNN",
            "SGD",
            "0.001",
            "Categorical Cross-Entropy",
            "10",
            "4",
            "3",
            "MobileNetV2 preprocessing",
        ],
    })

    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Reported Results")

    st.write(
        "The original notebook reports approximately "
        "97.88% test accuracy after training for "
        "10 epochs. This result requires careful "
        "interpretation because independently created "
        "training and testing splits may contain "
        "overlapping images."
    )

    st.subheader("Project Workflow")

    st.markdown(
        """
        1. Load and organize CT scan images.
        2. Prepare images for CNN training.
        3. Apply MobileNetV2 image preprocessing.
        4. Train the custom CNN architecture.
        5. Evaluate model predictions.
        6. Save the trained Keras model.
        7. Deploy the model using Streamlit.
        8. Provide project documentation and resources.
        """
    )

    st.subheader("Project Documentation")

    doc1, doc2 = st.columns(2)

    with doc1:
        st.link_button(
            "Read Project Report",
            PROJECT_LINKS["Project Report"],
            use_container_width=True,
        )

        st.link_button(
            "Open Training Notebook",
            PROJECT_LINKS["Notebook"],
            use_container_width=True,
        )

    with doc2:
        st.link_button(
            "View Presentation",
            PROJECT_LINKS["Presentation"],
            use_container_width=True,
        )

        st.link_button(
            "View GitHub Repository",
            PROJECT_LINKS["GitHub"],
            use_container_width=True,
        )

    st.warning(
        "This project is not clinically validated "
        "and must not be used for medical decisions."
    )

# ============================================================
# 13. PAGE: PROJECT RESOURCES
# ============================================================

elif section == "Project Resources":

    st.markdown(
        '<div class="section-kicker">'
        'RESEARCH MATERIALS'
        '</div>',
        unsafe_allow_html=True,
    )

    st.title("Project Resources")

    st.write(
        "Access all LungAI project deliverables, "
        "including the dataset, notebook, source code, "
        "live application, project report, presentation, "
        "and video demonstration."
    )

    st.divider()

    for start in range(0, len(RESOURCE_DETAILS), 2):
        columns = st.columns(2, gap="large")

        for offset, column in enumerate(columns):
            index = start + offset

            if index >= len(RESOURCE_DETAILS):
                continue

            resource = RESOURCE_DETAILS[index]

            with column:
                with st.container(border=True):
                    st.markdown(
                        '<div class="resource-category">'
                        + resource["category"]
                        + '</div>',
                        unsafe_allow_html=True,
                    )

                    st.subheader(resource["title"])

                    st.write(resource["description"])

                    st.link_button(
                        "Open Resource",
                        PROJECT_LINKS[resource["key"]],
                        use_container_width=True,
                    )

    st.divider()

    st.subheader("Project Video Demonstration")

    st.video(
        PROJECT_LINKS["YouTube"]
    )

    st.divider()

    st.subheader("Source Repository")

    st.code(
        PROJECT_LINKS["GitHub"],
        language=None,
    )

    st.caption(
        "Ensure all externally hosted project files "
        "are shared with appropriate viewing permissions "
        "for academic evaluation."
    )

# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)
