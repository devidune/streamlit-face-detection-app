import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Human Face Identification App",
    page_icon="🙂",
    layout="centered"
)

st.title("🧠 Human Face Identification")
st.write("Upload an image and the app will detect and tag human faces.")

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Image uploader
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("📷 Original Image")
    st.image(image, use_container_width=True)

    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Face detection parameters (controllable if needed)
    scale_factor = st.slider("Scale Factor", 1.05, 1.5, 1.2, 0.05)
    min_neighbors = st.slider("Min Neighbors", 3, 10, 5)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image_np,
            "Human face identified",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    st.subheader("✅ Detection Result")
    st.image(image_np, use_container_width=True)

    st.success(f"Number of faces detected: {len(faces)}")
else:
    st.info("Please upload an image to begin.")
