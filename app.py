import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title='LungAI | CT Scan Analysis', page_icon='🫁', layout='wide', initial_sidebar_state='expanded')
LABELS = ['Malignant', 'Normal', 'Benign']
COLORS = {'Malignant': '#fb7185', 'Normal': '#34d399', 'Benign': '#fbbf24'}

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
[data-testid="stApp"] {background:radial-gradient(ellipse at 85% 0%,#122c40 0%,#091522 45%,#07111d 100%);color:#e8f3fa;}
[data-testid="stSidebar"] {background:#0b1c2c;border-right:1px solid #244052;}
h1,h2,h3 {font-family:'Outfit',sans-serif !important;letter-spacing:-.025em;}
h1 {font-weight:800 !important;}
.block-container {padding-top:2rem;max-width:1350px;}
.hero {background:linear-gradient(110deg,#12354b,#0d2436);border:1px solid #28546a;border-radius:22px;padding:30px 34px;margin-bottom:24px;}
.eyebrow {color:#57dec9;text-transform:uppercase;font-size:12px;font-weight:800;letter-spacing:2px;}
.hero h1 {font-size:40px;margin:9px 0;color:#f1faff;}
.hero p {color:#afc5d4;font-size:15px;margin:0;}
.panel {background:#102335;border:1px solid #284356;border-radius:18px;padding:20px 22px;margin-bottom:16px;}
.panel-title {color:#d7e9f4;font-weight:700;font-size:16px;margin-bottom:8px;}
.muted {color:#91aebe;font-size:13px;}
.result {border-radius:17px;padding:23px;background:#102b3b;border:1px solid #31536a;}
.result-label {font-size:34px;font-family:Outfit,sans-serif;font-weight:800;}
[data-testid="stFileUploader"] {border:1px dashed #37677a;border-radius:16px;padding:12px;background:#102638;}
[data-testid="stMetric"] {background:#102638;border:1px solid #28485a;border-radius:15px;padding:16px;}
.stButton>button, .stDownloadButton>button {background:#19bba8;color:#041b24;border:0;border-radius:10px;font-weight:800;padding:11px 22px;}
.stButton>button:hover, .stDownloadButton>button:hover {background:#56ddcc;color:#041b24;}
[data-testid="stAlert"] {border-radius:12px;}
</style>''', unsafe_allow_html=True)

@st.cache_resource(show_spinner='Loading AI model...')
def load_model():
    return tf.keras.models.load_model('Lung_Model.h5', compile=False)

@st.cache_data(show_spinner=False)
def preprocess_image(image_bytes):
    with Image.open(io.BytesIO(image_bytes)) as source:
        img = source.convert('RGB').resize((244, 244))
    arr = np.asarray(img, dtype=np.float32)[None, ...]
    return tf.keras.applications.mobilenet_v2.preprocess_input(arr)

if 'history' not in st.session_state:
    st.session_state.history = []
if 'result' not in st.session_state:
    st.session_state.result = None

with st.sidebar:
    st.markdown('## 🫁 LungAI')
    st.caption('INTELLIGENT CT IMAGE ANALYSIS')
    st.divider()
  page = st.sidebar.selectbox(
    "Select Page",
    ["🔬 Scan Analyzer", "📊 Analysis History", "ℹ️ About Model"],
    index=0
)
    st.divider()
    st.markdown('**MODEL CONFIGURATION**')
    st.caption('Architecture: Keras / TensorFlow')
    st.caption('Input: RGB · 244 × 244')
    st.caption('Classes: Malignant · Normal · Benign')
    st.caption('Preprocessing: MobileNetV2')
    st.divider()
    st.warning('Research prototype only. Not for clinical diagnosis.')

try:
    model = load_model()
    model_ready = True
except Exception as exc:
    model_ready = False
    model_error = str(exc)

if page == '🔬 Scan Analyzer':
    st.markdown('''<div class="hero"><div class="eyebrow">AI-POWERED MEDICAL IMAGE ANALYSIS</div><h1>Lung CT Scan Analyzer</h1><p>Upload a lung CT image to explore predictions from your trained deep-learning model.</p></div>''', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric('Classification categories', '03')
    b.metric('Input resolution', '244 × 244')
    c.metric('Scans this session', len(st.session_state.history))
    st.write('')
    left, right = st.columns([1.1, 1], gap='large')
    with left:
        st.subheader('01  ·  Upload CT Image')
        uploaded = st.file_uploader('Select a CT image (PNG, JPG, JPEG)', type=['png', 'jpg', 'jpeg'])
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            try:
                preview = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                st.image(preview, caption='Uploaded scan · ' + uploaded.name, use_container_width=True)
            except Exception:
                st.error('The selected file could not be decoded as an image.')
                uploaded = None
        if uploaded is not None:
            if st.button('✦ Analyze CT Scan', type='primary', use_container_width=True, disabled=not model_ready):
                with st.spinner('Analyzing image...'):
                    try:
                        batch = preprocess_image(image_bytes)
                        raw = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
                        if raw.size != len(LABELS) or not np.all(np.isfinite(raw)):
                            raise ValueError('Expected three finite model outputs.')
                        # Supports either probability outputs or raw logits.
                        if np.any(raw < 0) or np.any(raw > 1) or not np.isclose(raw.sum(), 1, atol=0.02):
                            scores = tf.nn.softmax(raw).numpy()
                        else:
                            scores = raw / raw.sum()
                        index = int(np.argmax(scores))
                        result = {'filename': uploaded.name, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'label': LABELS[index], 'confidence': float(scores[index] * 100), 'scores': [float(x * 100) for x in scores]}
                        st.session_state.result = result
                        st.session_state.history.insert(0, result)
                    except Exception as exc:
                        st.error(f'Prediction failed: {exc}')
        if not model_ready:
            st.error('Could not load Lung_Model.h5. Place the trained model beside app.py. Details: ' + model_error)
    with right:
        st.subheader('02  ·  AI Classification')
        result = st.session_state.result
        if result is None:
            st.markdown('<div class="panel"><div class="panel-title">Awaiting analysis</div><p class="muted">Upload an image and click Analyze CT Scan to display the predicted class and model scores.</p></div>', unsafe_allow_html=True)
        else:
            label = result['label']
            st.markdown(f'<div class="result"><div class="eyebrow">PREDICTED CLASS</div><div class="result-label" style="color:{COLORS[label]}">{label}</div><div class="muted">Highest model score: {result["confidence"]:.2f}%</div></div>', unsafe_allow_html=True)
            st.write('')
            st.markdown('**Class scores**')
            for name, score in zip(LABELS, result['scores']):
                st.markdown(f'**{name}** — {score:.2f}%')
                st.progress(min(1.0, max(0.0, score / 100)))
            st.caption('Scores describe the model output, not clinical certainty or calibrated disease risk.')
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            pdf_buffer = io.BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            lines = [Paragraph('LungAI - Research Prediction Summary', styles['Title']), Spacer(1, 20)]
            for value in [f'Image: {result["filename"]}', f'Analyzed: {result["timestamp"]}', f'Predicted class: {label}', f'Highest model score: {result["confidence"]:.2f}%']:
                lines.append(Paragraph(value, styles['Normal']))
                lines.append(Spacer(1, 10))
            for name, score in zip(LABELS, result['scores']):
                lines.append(Paragraph(f'{name}: {score:.2f}%', styles['Normal']))
            lines.extend([Spacer(1, 20), Paragraph('Educational research output only. Not a diagnosis. Consult a qualified clinician for medical interpretation.', styles['Normal'])])
            doc.build(lines)
            st.download_button('↓ Download PDF Summary', data=pdf_buffer.getvalue(), file_name='LungAI_prediction_report.pdf', mime='application/pdf', use_container_width=True)
    st.info('A CT scan is not sufficient on its own to confirm or exclude lung cancer. Results require qualified clinical interpretation.')
elif page == ' Analysis History':
    st.title('Analysis History')
    st.caption('Results are stored only in this Streamlit session; they disappear when the session is reset.')
    if st.session_state.history:
        rows = [{k: v for k, v in entry.items() if k != 'scores'} for entry in st.session_state.history]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.download_button('Download CSV', pd.DataFrame(rows).to_csv(index=False), file_name='lungai_history.csv', mime='text/csv')
        if st.button('Clear Session History'):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()
    else:
        st.info('No scans have been analyzed in this session.')
else:
    st.title('About LungAI')
    st.markdown('**Purpose:** Educational demonstration of three-class CT image classification using a pretrained Keras model.')
    st.markdown('**Labels in model output order:** Malignant, Normal, Benign.')
    st.markdown('**Pipeline:** Decode image → RGB conversion → resize to 244 × 244 → MobileNetV2 preprocessing → Keras inference → display class scores.')
    st.warning('Before deployment, verify the training image size, preprocessing, label ordering, dataset provenance, external validation, and score calibration. Do not upload identifiable patient scans without appropriate authorization and privacy safeguards.')
