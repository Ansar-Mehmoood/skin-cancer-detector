import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Model load karna
model = tf.keras.models.load_model('model.h5')  # Make sure model.h5 is in same folder

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))  # model input size
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI
st.title("Skin Cancer Detection App")
st.write("Upload a skin image to detect if it's Benign or Malignant.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        class_names = ["Benign", "Malignant"]  # Adjust according to your model
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        st.markdown(f"### 🔍 Prediction: `{predicted_class}`")
        st.markdown(f"### ✅ Confidence: `{confidence:.2f}`")
