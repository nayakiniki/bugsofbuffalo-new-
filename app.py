import streamlit as st
import cv2
import torch
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# -------------------------
# Load YOLOv8 model
# -------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # default pretrained
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, device

model, device = load_model()

# -------------------------
# Load breed info JSON
# -------------------------
with open("breeds_info.json", "r") as f:
    breed_info = json.load(f)

# -------------------------
# Function to call Mixtral
# -------------------------
def ask_mixtral(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="mistralai/mixtral-8x7b-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for dog breeds."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Dog Breed Detector + AI", layout="wide")
st.title("🐶 Dog Breed Detection + Mixtral AI Q&A")
st.write("Upload an **image** or **video** to detect dog breeds. Then ask Mixtral AI questions about them.")

upload_type = st.radio("Choose input type:", ["Image", "Video"])

detected_labels = []  # store detections to pass into Mixtral

# -------------------------
# Image Upload
# -------------------------
if upload_type == "Image":
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        image = Image.open(img_file).convert("RGB")

        # Run YOLO
        results = model(image)

        # Annotated result
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Display
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.image(annotated, caption="Detected Objects", use_column_width=True)

        # Show breed info
        st.subheader("📖 Detection Results")
        for r in results[0].boxes:
            cls_id = int(r.cls)
            conf = float(r.conf)
            label = model.names[cls_id]

            if label not in detected_labels:
                detected_labels.append(label)

            st.markdown(f"**Breed:** {label} | **Confidence:** {conf:.2f}")

            if label in breed_info:
                info = breed_info[label]
                st.write(f"- 📝 {info['description']}")
                st.write(f"- 🌍 Origin: {info['origin']}")
                st.write(f"- 📏 Size: {info['size']}")

# -------------------------
# Video Upload
# -------------------------
elif upload_type == "Video":
    vid_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if vid_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # collect labels
            for r in results[0].boxes:
                cls_id = int(r.cls)
                label = model.names[cls_id]
                if label not in detected_labels:
                    detected_labels.append(label)

            stframe.image(annotated, channels="RGB", use_column_width=True)

        cap.release()

# -------------------------
# Mixtral Q&A Section
# -------------------------
if detected_labels:
    st.subheader("💡 Ask Mixtral AI about the detected breeds")
    st.write(f"Detected breeds: {', '.join(detected_labels)}")

    user_q = st.text_input("Enter a question about the detected breed(s):")
    if user_q:
        with st.spinner("Mixtral is thinking..."):
            context = f"The detected breeds are: {', '.join(detected_labels)}.\n\n"
            ai_answer = ask_mixtral(context + user_q)
            st.markdown("### 🤖 Mixtral AI Answer")
            st.write(ai_answer)



