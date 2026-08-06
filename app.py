import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io

# Local backend API URL (or your deployed cloud API URL)
API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Damaged Apples Detector", layout="centered")
st.title("🍎 Damaged Apples Detection System")
st.write("Upload an image or use your camera to identify plant species.")

# Sidebar Configuration
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Filter out detections with a confidence score below this threshold."
)

PLANT_CARE_INFO = {
    "monstera": "Water every 1-2 weeks. Thrives in bright to medium indirect light.",
    "snake_plant": "Water every 2-3 weeks. Tolerates low light and dry air.",
    "pothos": "Water when top 2 inches of soil are dry. Prefers indirect light.",
    "succulent": "Water sparingly every 2-4 weeks. Requires direct sunlight."
}

option = st.radio("Choose Input Method:", ("Upload Image", "Use Camera"))

image_bytes = None
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
elif option == "Use Camera":
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        image_bytes = camera_file.read()

if image_bytes is not None:
    image = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(image.convert("RGB"))

    # Send request to FastAPI backend
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    try:
        response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            raw_predictions = response.json().get("predictions", [])
            
            # Filter predictions based on slider confidence threshold
            predictions = [
                pred for pred in raw_predictions 
                if pred["confidence"] >= conf_threshold
            ]
            
            # Draw bounding boxes for filtered predictions
            for pred in predictions:
                box = pred["bbox"]
                label = f"{pred['class_name']} ({pred['confidence']*100:.1f}%)"
                
                pt1 = (int(box[0]), int(box[1]))
                pt2 = (int(box[2]), int(box[3]))
                
                # Draw bounding box (thickness set to 3)
                cv2.rectangle(img_array, pt1, pt2, (0, 255, 0), 3)
                
                # Draw label text above the bounding box
                cv2.putText(
                    img_array, 
                    label, 
                    (pt1[0], max(pt1[1] - 12, 35)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, 
                    (0, 255, 0), 
                    3
                )

            st.subheader("Detection Results")
            st.image(img_array, caption="Detected Plants", use_container_width=True)

            if len(predictions) > 0:
                st.subheader("Details & Care Instructions")
                for pred in predictions:
                    name = pred["class_name"]
                    conf = pred["confidence"] * 100
                    st.write(f"**Species:** `{name}` | **Confidence:** `{conf:.1f}%`")
                    care = PLANT_CARE_INFO.get(name.lower(), "No specific care instructions available.")
                    st.info(f"💡 **Care Tip:** {care}")
            else:
                st.warning(f"No plant species detected with a confidence ratio ≥ {conf_threshold:.2f}. Try lowering the confidence threshold slider.")
        else:
            st.error("Failed to connect to the model API.")
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")