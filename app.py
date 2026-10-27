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
        else:
            st.info("Your uploaded CT image preview will appear here.")
        if not MODEL_PATH.is_file():
            st.warning("Lung_Model.h5 is missing. Add it next to app.py to enable predictions.")
        if st.button("🔬 Analyze CT Scan", type="primary", use_container_width=True, disabled=(data is None or not MODEL_PATH.is_file())):
            try:
                with st.spinner("Analyzing image..."):
                    probs = infer(data)
                idx = int(np.argmax(probs))
                record = {"file": uploaded.name, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          "label": LABELS[idx], "confidence": float(probs[idx] * 100),
                          "scores": {label: float(p * 100) for label, p in zip(LABELS, probs)}}
                st.session_state.lung_result = record
                st.session_state.lung_history.insert(0, record)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.caption("Check model compatibility, class order and preprocessing.")
    with right:
        st.subheader("Prediction Results")
        record = st.session_state.lung_result
        if record is None:
            st.info("Upload an image and click Analyze CT Scan to see the classification results.")
        else:
            st.metric("Predicted class", record["label"])
            st.metric("Highest model score", f"{record['confidence']:.2f}%")
            for label, value in record["scores"].items():
                st.write(f"**{label}** — {value:.2f}%")
                st.progress(min(max(value / 100, 0), 1))
            try:
                st.download_button("⬇ Download PDF Report", report_pdf(record), "LungAI_report.pdf", "application/pdf", use_container_width=True)
            except ImportError:
                st.warning("Install reportlab for PDF export.")
    st.divider()
    st.caption("Model scores are not clinically validated disease probabilities. Do not use this app for diagnosis.")

elif section == "📊 Analysis History":
    st.subheader("Analysis History")
    if st.session_state.lung_history:
        rows = [{"Time": r["time"], "File": r["file"], "Prediction": r["label"], "Highest score (%)": round(r["confidence"], 2)} for r in st.session_state.lung_history]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        import csv
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("⬇ Download History CSV", out.getvalue(), "LungAI_history.csv", "text/csv")
        if st.button("Clear History"):
            st.session_state.lung_history = []
            st.session_state.lung_result = None
            st.rerun()
    else:
        st.info("No predictions yet. Choose Scan Analyzer from the sidebar to upload an image.")

else:
    st.subheader("About LungAI")
    st.write("LungAI is an educational research interface for a Keras CT-image classification model.")
    st.write("**Classes:** Malignant, Normal, Benign")
    st.write("**Pipeline:** Upload → RGB → 244 × 244 → MobileNetV2 preprocessing → Model → Classification scores")
    st.warning("Verify the class-index order and preprocessing against your training notebook. Never treat model output as a medical diagnosis.")
