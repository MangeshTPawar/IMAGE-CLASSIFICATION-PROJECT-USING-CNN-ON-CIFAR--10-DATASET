import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load models
cnn_model = load_model('cifar10_cnn_model.h5')
mobilenet_model = load_model('mobilenetv2.h5')

# CIFAR-10 Class Names
cifar10_classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def classify_image_cnn(image):
    # Preprocess for CNN
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    predictions = cnn_model.predict(image_array)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]
    return cifar10_classes[class_idx], confidence

def classify_image_mobilenet(image):
    # Preprocess for MobileNetV2
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = preprocess_input(np.expand_dims(image_array, axis=0))
    predictions = mobilenet_model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0][0]
    return decoded_predictions[1], decoded_predictions[2]

# Streamlit App
st.title("Image Classification App")
st.write("Upload an image to classify it using either the custom CNN model or MobileNetV2.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Model Selection
    model_choice = st.radio("Select a model for classification:", ('Custom CNN', 'MobileNetV2'))

    if st.button("Classify"):
        if model_choice == 'Custom CNN':
            class_name, confidence = classify_image_cnn(image)
            st.write(f"**Prediction:** {class_name}")
            st.write(f"**Confidence:** {confidence:.2f}")
        elif model_choice == 'MobileNetV2':
            class_name, confidence = classify_image_mobilenet(image)
            st.write(f"**Prediction:** {class_name}")
            st.write(f"**Confidence:** {confidence:.2f}")