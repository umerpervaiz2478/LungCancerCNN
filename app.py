import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained Keras model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('Lung_Model.h5')
    return model

model = load_model()

# Define the class labels
class_labels = ['Malignant', 'Normal', 'benign']

# Set up the Streamlit page
st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")
st.title("Lung Cancer Prediction from CT Scans")
st.write("Upload a CT scan image to predict if it's Malignant, Normal, or Benign.")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Preprocess the image
    st.write("Classifying...")
    # Resize to 244x244 as required by the model, ensure RGB
    image = image.resize((244, 244))
    image_array = np.array(image)

    # Check if image has 4 channels (RGBA) and convert to RGB if necessary
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3] # Take only RGB channels

    # Expand dimensions to create a batch of 1 image
    image_array = np.expand_dims(image_array, axis=0)

    # Normalize the image using MobileNetV2 preprocess_input
    processed_image = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

    # Make a prediction
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100

    predicted_class_name = class_labels[predicted_class_index]

    # Display the results
    st.success(f"Prediction: {predicted_class_name}")
    st.info(f"Confidence: {confidence:.2f}%")

    # Optional: Display all class probabilities
    st.subheader("All Class Probabilities:")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{class_labels[i]}: {prob*100:.2f}%")
