import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import base64
import os
import sys

# Get the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "trained.h5")

# Add project root to Python path
sys.path.append(PROJECT_ROOT)

from src.get_data import read_params

# Page configuration
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: black;'>Brain Tumor Classification</h1>", unsafe_allow_html=True)

# Load the trained model with error handling
try:
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.info("Please ensure the model is trained and saved correctly.")
        st.stop()
    model = load_model(MODEL_PATH)
    input_shape = model.input_shape[1:3]
    st.success(f"Model loaded successfully! Expected input shape: {input_shape}")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Define class labels
classes = ['no_tumor', 'pituitary_tumor', 'meningioma_tumor', 'glioma_tumor']

# Function to preprocess the image
def preprocess_image(image, target_size=(255, 255)):
    """Preprocess image for model prediction"""
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        image = image.resize(target_size)
        image_array = np.array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Function to generate Grad-CAM visualization
def generate_gradcam(image, model):
    """Generate Grad-CAM visualization for the image"""
    try:
        img_array = np.array(image)
        last_conv_layer = next(layer for layer in reversed(model.layers) 
                             if isinstance(layer, tf.keras.layers.Conv2D))
        
        grad_model = tf.keras.Model(
            model.inputs,
            [last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(np.array([img_array]))
            class_idx = tf.argmax(predictions[0])
            class_output = predictions[:, class_idx]
            
        grads = tape.gradient(class_output, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
        heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
        return Image.fromarray(heatmap)
    except Exception as e:
        st.warning(f"Could not generate Grad-CAM: {e}")
        return None

# Function to generate and download report
def download_report(pred_class, confidence):
    """Generate downloadable report with prediction results"""
    report_text = f"""
    Brain Tumor Classification Report
    -------------------------------
    Prediction: {pred_class}
    Confidence: {confidence:.2f}%
    Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    b64 = base64.b64encode(report_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="brain_tumor_report.txt">Download Report</a>'
    return href

# Sidebar for uploading MRI scan
st.sidebar.title("Upload MRI Scan")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI Scan', use_column_width=True)

        if st.button("Classify Image"):
            with st.spinner('Processing image...'):
                input_tensor = preprocess_image(image, target_size=input_shape)
                
                if input_tensor is not None:
                    st.info(f"Input shape: {input_tensor.shape}")
                    output = model.predict(input_tensor)
                    pred_idx = np.argmax(output, axis=1)[0]
                    confidence = float(output[0][pred_idx] * 100)
                    pred_class = classes[pred_idx]

                    # Display results
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.success(f"Prediction: {pred_class}")
                        st.info(f"Confidence: {confidence:.2f}%")
                    
                    with col2:
                        gradcam_image = generate_gradcam(image, model)
                        if gradcam_image:
                            st.image(gradcam_image, caption='Grad-CAM Visualization')

                    # Provide download link for report
                    st.markdown(download_report(pred_class, confidence), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")