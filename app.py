import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load model
model = YOLO("trained.pt")

st.title("🚗 Damage Detection")

st.write("Capture an image of the vehicle to detect damage")

# camera input
picture = st.camera_input("Take a picture")

if picture:

    image = Image.open(picture)
    img = np.array(image)

    results = model(img)

    annotated = results[0].plot()

    st.image(annotated, caption="Detection Result", use_column_width=True)