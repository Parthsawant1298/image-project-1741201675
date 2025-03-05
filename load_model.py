import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Load the model
model_path = 'best_model.keras'
try:
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
    print(f"Model output shape: {model.output_shape}")
    num_classes = model.output_shape[1]
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define class labels based on the class indices
# You may need to update these class labels based on your specific dataset
class_labels = {0: 'cat', 1: 'dog', 2: 'horse'}

def predict_image(img_path):
    # Load and preprocess the image
    img = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format
    img = img.resize((64, 64))  # Resize to match training images
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Ensure scaling matches training preprocessing
    
    # Predict the class
    result = model.predict(img_array)
    predicted_class_index = np.argmax(result[0])
    
    # Get the class name
    if predicted_class_index in class_labels:
        prediction = class_labels[predicted_class_index]
    else:
        prediction = f"Unknown class {predicted_class_index}"
    
    return prediction, result[0]

st.title("Image Classification")
st.write("Upload an image to classify.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
    st.write("")
    st.write("Classifying...")
    
    try:
        prediction, probabilities = predict_image(uploaded_file)
        
        st.write(f"Prediction: {prediction}")
        
        # Display probabilities
        st.write("Prediction probabilities:")
        for i, prob in enumerate(probabilities):
            class_name = class_labels.get(i, f"Class {i}")
            st.write(f"{class_name}: {prob:.4f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")