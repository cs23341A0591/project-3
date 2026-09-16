"""
Streamlit Image Classification App
-----------------------------------
Loads the CNN trained in cifar10_cnn_classifier.ipynb and classifies
uploaded images into one of the 10 CIFAR-10 categories:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

Run with:
    streamlit run streamlit_app.py

Expects these files in the same folder (produced by the notebook):
    - cifar10_cnn.keras
    - class_names.json   (optional — falls back to a built-in list)
"""

import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "cifar10_cnn.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (32, 32)

DEFAULT_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            return json.load(f)
    return DEFAULT_CLASS_NAMES


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize to 32x32, ensure RGB, normalize to [0, 1], add batch dim."""
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.asarray(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    st.title("🖼️ CIFAR-10 Image Classifier")
    st.write(
        "Upload an image and the CNN will classify it into one of 10 categories: "
        "**airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck**."
    )

    model = load_model()
    class_names = load_class_names()

    if model is None:
        st.error(
            f"Couldn't find `{MODEL_PATH}` in this folder. "
            "Run the notebook first — it saves the trained model here — "
            "then place it next to this script."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is None:
        st.info("Waiting for an image to be uploaded.")
        return

    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Classifying..."):
        batch = preprocess_image(image)
        probs = model.predict(batch, verbose=0)[0]

    top_idx = int(np.argmax(probs))
    top_label = class_names[top_idx]
    top_conf = float(probs[top_idx]) * 100

    with col2:
        st.subheader("Prediction")
        st.metric(label="Predicted class", value=top_label.capitalize())
        st.write(f"Confidence: **{top_conf:.2f}%**")

    st.subheader("Class probabilities")
    prob_dict = {
        name: float(p) for name, p in sorted(
            zip(class_names, probs), key=lambda x: x[1], reverse=True
        )
    }
    st.bar_chart(prob_dict)

    with st.expander("Show raw probabilities"):
        for name, p in prob_dict.items():
            st.write(f"{name.capitalize()}: {p * 100:.2f}%")

    st.caption(
        "Note: this model was trained on 32×32 CIFAR-10 images, so it works best "
        "on simple, single-object photos similar to those 10 categories."
    )


if __name__ == "__main__":
    main()
