## README.md for GitHub

# Lung Cancer Prediction from CT Scans

This repository contains the code and resources for a Deep Learning model designed to predict lung cancer from CT scan images. The project utilizes a Convolutional Neural Network (CNN) trained on the IQ-OTHNCCD Lung Cancer Dataset to classify CT scans into three categories: 'Malignant', 'Normal', or 'Benign'. A Streamlit application is also provided for an easy-to-use interface to test the model.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Streamlit Application](#streamlit-application)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Results and Metrics](#results-and-metrics)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to develop an AI-powered tool to assist in the early detection of lung cancer. By analyzing CT scan images, the model aims to provide quick and accurate classifications, potentially aiding medical professionals.

The project consists of:
- **Data Loading and Preprocessing**: Handling CT scan image data and preparing it for model training.
- **CNN Model Training**: Building and training a custom CNN model using TensorFlow/Keras.
- **Model Evaluation**: Assessing the model's performance using metrics like accuracy, precision, recall, F1-score, and confusion matrices.
- **Streamlit Web Application**: A user-friendly interface to upload CT scan images and get real-time predictions.

## Dataset

The model was trained on the **IQ-OTHNCCD Lung Cancer Dataset**. This dataset contains CT scan images categorized into 'Malignant', 'Normal', and 'Benign' cases.

## Model Architecture

The core of this project is a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras. The model comprises several convolutional layers, batch normalization, max-pooling layers, and fully connected dense layers, culminating in a softmax activation for classification.

## Streamlit Application

A Streamlit application (`app.py`) provides an interactive way to use the trained model. Users can upload a CT scan image, and the application will display the predicted class (Malignant, Normal, or Benign) along with a confidence score and individual class probabilities.

**Live Demo:** You can access the live Streamlit application here: [https://lungcancer-cnn-by-umer.streamlit.app/](https://lungcancer-cnn-by-umer.streamlit.app/)

## Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd lung-cancer-prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (You will need to create a `requirements.txt` file from your notebook's imports, e.g., `pip freeze > requirements.txt`)
    Alternatively, install individually:
    ```bash
    pip install tensorflow keras numpy pandas scikit-learn matplotlib seaborn Pillow streamlit
    ```

4.  **Download the trained model:**
    The trained model `Lung_Model.h5` should be available in the repository or you might need to download it separately if it's too large for GitHub. Make sure it's in the same directory as `app.py`.

## Usage

To run the Streamlit application:

1.  Ensure you are in the project's root directory and your virtual environment is activated.
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3.  Your web browser will automatically open to the application (usually `http://localhost:8501`). You can then upload CT scan images for prediction.

## Deployment

The Streamlit application can be deployed to various cloud platforms such as Streamlit Cloud, Heroku, AWS EC2, or Google Cloud Platform. For the live demo, it's deployed on Streamlit Cloud.

## Results and Metrics

The model achieved excellent performance on the test set:

```
              precision    recall  f1-score   support

   Malignant       1.00      1.00      1.00       174
      Normal       0.99      0.98      0.99       116
      benign       0.95      0.97      0.96        40

    accuracy                           0.99       330
   macro avg       0.98      0.99      0.98       330
weighted avg       0.99      0.99      0.99       330

Accuracy of the Model: 99.1%
```

This indicates strong capabilities in classifying lung CT scans into the defined categories.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
