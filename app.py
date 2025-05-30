import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# 🔧 Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("waste_classifier_model.h5")
    return model

model = load_model()

# 🏷️ Class labels (update if changed)
class_names = ['metal', 'organic', 'paper', 'plastic']

# 🌟 Streamlit UI
st.title("♻️ Waste Material Classifier")
st.write("Upload an image of waste material (e.g. paper, plastic, organic, or metal) to identify its category.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # ✂️ Preprocess
    img = img.resize((150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # 🔮 Predict
    preds = model.predict(x)
    pred_class = class_names[np.argmax(preds)]
    confidence = 100 * np.max(preds)

    # 🎯 Output
    st.subheader("Prediction:")
    st.success(f"This image is predicted to be **{pred_class.upper()}** waste.")
    st.info(f"Confidence: {confidence:.2f}%")
