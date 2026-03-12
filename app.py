import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("🚗 Vehicle Damage Detector")

@st.cache_resource
def load_model():
    return YOLO("trained.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload a car image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = model(np.array(image))
    annotated = results[0].plot()

    st.image(annotated, caption="Detection Result", use_container_width=True)