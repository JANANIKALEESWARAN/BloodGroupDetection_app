import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import pandas as pd

# Define your blood group labels (must match model training order)
CLASS_NAMES = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Load model only once using Streamlit cache
@st.cache_resource
def load_resnet_model():
    return load_model("model.keras")  # Updated to .keras format

model = load_resnet_model()

# Preprocessing function
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # match training preprocessing
    return img_array

# Confidence color helper
def get_confidence_color(conf):
    if conf > 80:
        return "green"
    elif conf > 50:
        return "orange"
    else:
        return "red"

# Custom CSS for style
st.markdown("""
    <style>
        .stApp {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            text-align: center;
            color: #ff4b4b;
        }
        .result-card {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            text-align: center;
            margin-top: 20px;
        }
        .pred-label {
            font-size: 22px;
            font-weight: bold;
        }
        .confidence {
            font-size: 18px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>🩸 Blood Group Classification</h1>", unsafe_allow_html=True)
st.write("Upload an image of a fingerprint and get the predicted blood group.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)[0]

        # Sort predictions (highest first)
        top_indices = predictions.argsort()[::-1][:3]
        
        # Main prediction
        main_index = top_indices[0]
        main_conf = predictions[main_index] * 100
        main_label = CLASS_NAMES[main_index]
        
        # Result Card
        st.markdown(
            f"""
            <div class='result-card'>
                <div class='pred-label'>Predicted Blood Group: {main_label}</div>
                <div class='confidence' style='color:{get_confidence_color(main_conf)}'>
                    Confidence: {main_conf:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Top-3 Predictions Table
        st.subheader("📊 Top 3 Predictions")
        top_df = pd.DataFrame({
            "Blood Group": [CLASS_NAMES[i] for i in top_indices],
            "Confidence (%)": [round(predictions[i]*100, 2) for i in top_indices]
        })
        st.dataframe(top_df, use_container_width=True)
