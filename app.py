
import io
import hashlib
from pathlib import Path
from datetime import datetime
from xml.sax.saxutils import escape

import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="LungAI | Lung Cancer Classification",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Lung_Model.h5"

# Must match the training class indices.
CLASS_LABELS = ["Malignant", "Normal", "benign"]

# Verify this against the training notebook.
# Supported: "rescale_255", "mobilenet_v2", "none"
PREPROCESSING = "rescale_255"

# ==================================================
# PROJECT LINKS
# ==================================================

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

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        125deg, #081827, #0c2636
    );
    color: #eef8fc;
}

[data-testid="stSidebar"] {
    background: #0b2030;
}

.block-container {
    max-width: 1300px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(
        110deg, #124558, #132c43
    );
    border: 1px solid #2b6275;
    border-radius: 18px;
    padding: 32px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #ffffff;
    font-size: 2.5rem;
    margin: 10px 0;
    font-weight: 800;
}

.hero p {
    color: #c5dce8;
    font-size: 1.05rem;
    line-height: 1.7;
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

.stButton button {
    border-radius: 10px;
}

[data-testid="stDownloadButton"] button {
    border-radius: 10px;
}

.section-heading {
    color: #69e7d3;
    font-size: 1.3rem;
    font-weight: 700;
    margin-top: 12px;
}

.footer {
    text-align: center;
    color: #9bb7c7;
    padding: 25px;
    font-size: 0.85rem;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# MODEL LOADING
# ==================================================

@st.cache_resource(
    show_spinner="Loading trained CNN model..."
)
def load_lung_model():
    import tensorflow as tf

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Lung_Model.h5 was not found."
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


# ==================================================
# IMAGE PREPROCESSING AND PREDICTION
# ==================================================

def predict_image(image_bytes):
    import tensorflow as tf

    model = load_lung_model()

    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 4:
        raise ValueError(
            "Expected a 4D image input model."
        )

    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]

    if height is None or width is None:
        raise ValueError(
            "Model image dimensions are undefined."
        )

    if channels != 3:
        raise ValueError(
            "This dashboard expects a 3-channel RGB model."
        )

    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")

        image = image.resize(
            (int(width), int(height)),
            Image.Resampling.BILINEAR
        )

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    if PREPROCESSING == "rescale_255":
        processed = image_array / 255.0

    elif PREPROCESSING == "mobilenet_v2":
        processed = (
            tf.keras.applications.mobilenet_v2
            .preprocess_input(image_array)
        )

    elif PREPROCESSING == "none":
        processed = image_array

    else:
        raise ValueError(
            "Unsupported preprocessing configuration."
        )

    prediction = model.predict(
        processed,
        verbose=0
    )

    scores = np.asarray(
        prediction,
        dtype=np.float64
    ).reshape(-1)

    if len(scores) != len(CLASS_LABELS):
        raise ValueError(
            "Model output must contain exactly 3 scores."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Model returned invalid prediction values."
        )

    # Convert logits to softmax probabilities if needed.
    if (
        np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(scores.sum(), 1, atol=0.02)
    ):
        exp_scores = np.exp(
            scores - np.max(scores)
        )
        scores = exp_scores / exp_scores.sum()

    else:
        scores = scores / scores.sum()

    return scores


# ==================================================
# PDF REPORT GENERATION
# ==================================================

def generate_pdf(result):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer
    )
    from reportlab.lib.styles import getSampleStyleSheet

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
        "Date: " + escape(result["time"]),
        "Predicted Class: " + escape(result["label"]),
        (
            "Highest Model Score: "
            f'{result["confidence"]:.2f}%'
        )
    ]

    for label, score in result["scores"].items():
        lines.append(
            f"{escape(label.title())}: {score:.2f}%"
        )

    lines.append(
        "Disclaimer: This report was generated by "
        "an educational deep learning model. "
        "It is not a medical diagnosis. "
        "Model scores are not calibrated clinical probabilities."
    )

    for line in lines:
        story.append(
            Paragraph(line, styles["Normal"])
        )
        story.append(Spacer(1, 12))

    document.build(story)

    return buffer.getvalue()


# ==================================================
# SESSION STATE
# ==================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

if "last_file" not in st.session_state:
    st.session_state.last_file = None


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:

    st.title("LungAI")

    st.caption(
        "AI-POWERED LUNG CT IMAGE ANALYSIS"
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

    st.subheader("Model Information")

    st.write("Architecture: Custom CNN")
    st.write("Framework: TensorFlow / Keras")
    st.write("Classification Classes: 3")
    st.write("Model Format: H5")

    st.divider()

    st.subheader("Quick Access")

    st.link_button(
        "GitHub Repository",
        PROJECT_LINKS["GitHub"],
        use_container_width=True
    )

    st.link_button(
        "Project Report",
        PROJECT_LINKS["Project Report"],
        use_container_width=True
    )

    st.link_button(
        "Watch Demonstration",
        PROJECT_LINKS["YouTube"],
        use_container_width=True
    )

    st.warning(
        "Educational research prototype. "
        "Not intended for clinical diagnosis."
    )


# ==================================================
# HERO SECTION
# ==================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">
        DEEP LEARNING | MEDICAL IMAGE ANALYSIS
    </div>

    <h1>LungAI - Lung Cancer Classification</h1>

    <p>
        An AI-powered lung CT scan classification
        system developed using Convolutional Neural
        Networks and TensorFlow.
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# PAGE 1: SCAN ANALYZER
# ==================================================

if section == "Scan Analyzer":

    st.title("CT Scan Analyzer")

    st.write(
        "Upload a lung CT scan image to explore "
        "the trained CNN model's classification output."
    )

    uploaded_file = st.file_uploader(
        "Upload CT Scan Image",
        type=["png", "jpg", "jpeg"],
        key="lung_upload"
    )

    if uploaded_file is None:
        st.info(
            "Upload a PNG, JPG, or JPEG image "
            "to begin classification."
        )

    else:

        image_bytes = uploaded_file.getvalue()

        file_id = hashlib.sha256(
            image_bytes
        ).hexdigest()

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
                with Image.open(
                    io.BytesIO(image_bytes)
                ) as image:
                    preview = image.convert("RGB")

                st.image(
                    preview,
                    caption=uploaded_file.name,
                    use_container_width=True
                )

                valid_image = True

            except (
                UnidentifiedImageError,
                OSError,
                ValueError
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
                    "Place the trained model beside app.py."
                )

            elif valid_image:

                if st.button(
                    "Analyze CT Scan",
                    type="primary",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Running CNN classification..."
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
                                max(score / 100.0, 0.0),
                                1.0
                            )
                        )

                    try:

                        pdf_bytes = generate_pdf(result)

                        st.download_button(
                            "Download Classification Report",
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
        "Research use only. Model predictions must "
        "not be used to make medical decisions."
    )


# ==================================================
# PAGE 2: ANALYSIS HISTORY
# ==================================================

elif section == "Analysis History":

    st.title("Analysis History")

    st.write(
        "Review classification results generated "
        "during your current browser session."
    )

    if not st.session_state.history:

        st.info(
            "No CT scans have been analyzed yet."
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
                    item["confidence"], 2
                )
            }

            row.update(item["scores"])

            records.append(row)

        dataframe = pd.DataFrame(records)

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Export Analysis History",
            data=dataframe.to_csv(index=False),
            file_name="LungAI_History.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("Clear Analysis History"):

            st.session_state.history = []
            st.session_state.result = None
            st.rerun()


# ==================================================
# PAGE 3: ABOUT LUNGAI
# ==================================================

elif section == "About LungAI":

    st.title("About LungAI")

    st.write("""
    LungAI is a Deep Learning semester project
    designed to classify lung CT scan images
    using a custom Convolutional Neural Network.

    The system analyzes uploaded images and
    generates classification scores across
    three categories: Malignant, Normal, and Benign.
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Classification Classes",
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
        "Reported accuracy is taken from the training "
        "notebook and requires independent validation."
    )

    st.divider()

    st.subheader("Problem Statement")

    st.write("""
    Lung CT scans contain complex visual patterns.
    This project investigates whether a CNN can
    learn image features that distinguish between
    malignant, benign, and normal dataset categories.

    The goal is to demonstrate deep learning
    techniques for medical image classification
    in an educational research environment.
    """)

    st.subheader("Dataset Description")

    st.write(
        "Dataset: IQ-OTH/NCCD Lung Cancer Dataset"
    )

    st.write("Total Images: 1,097")
    st.write("Malignant Images: 561")
    st.write("Normal Images: 416")
    st.write("Benign Images: 120")

    st.subheader("Dataset Distribution")

    import pandas as pd

    dataset_df = pd.DataFrame({
        "Class": [
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

    st.bar_chart(
        dataset_df.set_index("Class")
    )

    st.subheader("Model Architecture")

    st.write("""
    Model Type: Custom Convolutional Neural Network

    Framework: TensorFlow / Keras

    Optimizer: Stochastic Gradient Descent (SGD)

    Learning Rate: 0.001

    Loss Function: Categorical Cross-Entropy

    Training Epochs: 10

    Batch Size: 4

    Output Activation: Softmax

    Output Classes: 3
    """)

    st.subheader("Image Processing Pipeline")

    st.markdown("""
    1. Upload a CT scan image.
    2. Convert the image to RGB format.
    3. Resize to the model's expected input dimensions.
    4. Apply the configured image normalization.
    5. Pass the image through the trained CNN.
    6. Generate three classification scores.
    7. Display the highest-scoring category.
    8. Generate a downloadable PDF report.
    """)

    st.subheader("Project Results")

    st.write("""
    The training notebook reports a test accuracy
    of approximately 97.88%.

    However, the notebook's independently generated
    training and testing splits may overlap.
    Therefore, this result should be treated as
    preliminary until the model is evaluated using
    a strictly independent test set.
    """)

    st.subheader("Project Documentation")

    doc1, doc2 = st.columns(2)

    with doc1:

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

    with doc2:

        st.link_button(
            "View GitHub Repository",
            PROJECT_LINKS["GitHub"],
            use_container_width=True
        )

        st.link_button(
            "View Presentation",
            PROJECT_LINKS["Presentation"],
            use_container_width=True
        )

    st.warning(
        "This project is not clinically validated "
        "and must not be used for diagnosis."
    )


# ==================================================
# PAGE 4: PROJECT RESOURCES
# ==================================================

elif section == "Project Resources":

    st.title("Project Resources")

    st.write("""
    Access the complete LungAI Deep Learning
    semester project materials, including
    the dataset, training notebook, source code,
    report, presentation, and video demonstration.
    """)

    st.divider()

    resources = [
        {
            "title": "Dataset",
            "description": (
                "IQ-OTH/NCCD Lung Cancer CT Scan Dataset "
                "used for model training and evaluation."
            ),
            "url": PROJECT_LINKS["Dataset"]
        },
        {
            "title": "Google Colab Notebook",
            "description": (
                "Model development, image preprocessing, "
                "CNN training, and evaluation code."
            ),
            "url": PROJECT_LINKS["Notebook"]
        },
        {
            "title": "GitHub Repository",
            "description": (
                "Complete LungAI source code and "
                "project implementation."
            ),
            "url": PROJECT_LINKS["GitHub"]
        },
        {
            "title": "Live Deployment",
            "description": (
                "Access the deployed LungAI "
                "Streamlit web dashboard."
            ),
            "url": PROJECT_LINKS["Deployment"]
        },
        {
            "title": "Project Report",
            "description": (
                "Comprehensive project report covering "
                "the problem statement, dataset, "
                "CNN architecture, training, "
                "results, and conclusions."
            ),
            "url": PROJECT_LINKS["Project Report"]
        },
        {
            "title": "Project Presentation",
            "description": (
                "PowerPoint presentation explaining "
                "the Deep Learning semester project."
            ),
            "url": PROJECT_LINKS["Presentation"]
        },
        {
            "title": "YouTube Demonstration",
            "description": (
                "Recorded project demonstration "
                "and presentation."
            ),
            "url": PROJECT_LINKS["YouTube"]
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
                "Open Resource",
                resource["url"],
                use_container_width=True
            )

    st.divider()

    st.subheader("Project Video Demonstration")

    st.video(
        PROJECT_LINKS["YouTube"]
    )

    st.divider()

    st.subheader("GitHub Repository")

    st.code(
        PROJECT_LINKS["GitHub"],
        language=None
    )

    st.subheader("Project Report")

    st.link_button(
        "Read Complete Project Report",
        PROJECT_LINKS["Project Report"],
        use_container_width=True
    )

    st.info(
        "All resources are provided for academic "
        "review, reproducibility, and project evaluation."
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown("""
<div class="footer">
    <strong>LungAI</strong> | Deep Learning Semester Project
    <br>
    Developed using TensorFlow, Keras, Python and Streamlit
    <br>
    Educational and Research Purposes Only
</div>
""", unsafe_allow_html=True)
