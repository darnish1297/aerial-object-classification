import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------
# 1️⃣ Load Model
# ---------------------------
MODEL_PATH = r"C:\Users\Darnish S\aerial-object-classification\best_resnet50_finetuned.keras"

model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (update based on your dataset)
class_labels = ['bird', 'drone']

# ---------------------------
# 2️⃣ Streamlit UI
# ---------------------------
st.set_page_config(page_title="Aerial Object Classifier", page_icon="🛰️", layout="centered")

st.title("🛰️ Aerial Object Classification")
st.write("Upload an image to classify whether it’s a **Bird** or a **Drone** using the fine-tuned ResNet50 model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Predict
    prediction = model.predict(img_array)
    pred_idx = int(prediction[0] > 0.5)
    pred_label = class_labels[pred_idx]
    confidence = prediction[0][0] if pred_idx == 1 else 1 - prediction[0][0]

    # Output
    st.subheader("🔍 Prediction Result")
    st.success(f"**Predicted Class:** {pred_label.upper()}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")

    # Optional: Show probability bar
    st.progress(float(confidence))

# ---------------------------
# 3️⃣ Footer
# ---------------------------
st.markdown("---")
st.caption("Developed by Darnish | Aerial Object Classification using Transfer Learning (ResNet50)")
