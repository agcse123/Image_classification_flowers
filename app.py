import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config (modern Streamlit style)
st.set_page_config(page_title="Flower Classifier", layout="centered")

# Cache model properly (new API)
'''@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("flower_model_trained.hdf5")
    return model'''

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "flower_model_trained.hdf5",
        compile=False
    )
    return model



def preprocess_image(image):
    image = image.convert("RGB")
    image = np.array(image)
    image = tf.image.resize(image, (180, 180))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def predict_class(image, model):
    prediction = model.predict(image)
    return prediction


# Load model
model = load_model()

# UI
st.title("🌸 Flower Classifier")
st.write("Upload a flower image and the model will identify it.")

file = st.file_uploader(
    "Upload an image of a flower",
    type=["jpg", "jpeg", "png"]
)

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running inference..."):
        processed_image = preprocess_image(image)
        prediction = predict_class(processed_image, model)

    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    result = class_names[np.argmax(prediction)]

    st.success(f"🌼 Prediction: **{result.capitalize()}**")
else:
    st.info("Please upload a flower image to start.")
