import streamlit as st
from PIL import Image
import json
import requests
import io
import os

# Set page config
st.set_page_config(page_title="Smart Breed Detection", layout="wide")

# Initialize session state
if 'detected_breed' not in st.session_state:
    st.session_state.detected_breed = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Try to import ultralytics with error handling
try:
    from ultralytics import YOLO
except ImportError:
    st.error("Ultralytics package not found. Please make sure requirements.txt includes 'ultralytics'")
    st.stop()

# Load your custom trained model with error handling
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Please make sure it's in your repository.")
    st.stop()

try:
    model = YOLO(MODEL_PATH)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Load breed information with error handling
BREEDS_INFO_PATH = "breeds_info.json"
if not os.path.exists(BREEDS_INFO_PATH):
    st.error(f"Breed information file '{BREEDS_INFO_PATH}' not found.")
    st.stop()

try:
    with open(BREEDS_INFO_PATH, "r", encoding="utf-8") as f:
        breed_info = json.load(f)
except Exception as e:
    st.error(f"Error loading breed information: {str(e)}")
    st.stop()

# Available languages
LANGUAGES = {
    "English": "en",
    "Hindi": "hi", 
    "Odia": "or",
    "Bengali": "bn",
    "Punjabi": "pa"
}

# Function to call Mistral via OpenRouter
def query_mistral(prompt, api_key):
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/mixtral-8x7b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 500
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

st.title("🐂 Indian Cattle & Buffalo Breed Detection")
st.write("Upload an image to detect specific Indian breeds and get expert advice")

# API key input
api_key = st.sidebar.text_input("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", type="password")
if not api_key:
    st.sidebar.warning("Please enter your OpenRouter API key to use the chatbot")

# Language selection
selected_lang = st.sidebar.selectbox("Choose Language", list(LANGUAGES.keys()))
lang_code = LANGUAGES[selected_lang]

# File upload
uploaded_file = st.file_uploader("Choose an image of cattle/buffalo...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    img = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Perform prediction
    with st.spinner("Detecting breed..."):
        try:
            results = model.predict(img, conf=0.5)  # Confidence threshold
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            results = None
    
    with col2:
        if results and len(results[0].boxes) > 0:
            # Get the first detected breed
            breed_idx = int(results[0].boxes.cls[0])
            confidence = float(results[0].boxes.conf[0])
            breed_name = model.names[breed_idx]
            
            st.session_state.detected_breed = breed_name
            
            # Try to display the result image
            try:
                st.image(results[0].plot(), caption=f"Detection Result (Confidence: {confidence:.2f})", use_column_width=True)
            except:
                st.warning("Could not display detection visualization")
            
            st.success(f"**Detected Breed: {breed_name}**")
            
            # Display breed information in selected language
            if breed_name in breed_info:
                if lang_code in breed_info[breed_name]:
                    st.info(breed_info[breed_name][lang_code])
                else:
                    st.info(breed_info[breed_name]["en"])
            else:
                st.warning("Breed information not available in database.")
        else:
            st.error("No cattle/buffalo detected in the image. Please try another image.")

# Chatbot section
st.divider()
st.subheader("💬 Breed Expert Chatbot")

if st.session_state.detected_breed:
    st.write(f"Currently discussing: **{st.session_state.detected_breed}** breed")

user_question = st.text_input("Ask about treatment, diet, economic value, or any other question:")

if user_question and st.session_state.detected_breed and api_key:
    # Create a knowledgeable prompt with the detected breed
    breed_data = breed_info.get(st.session_state.detected_breed, {})
    english_info = breed_data.get("en", "No information available")
    
    prompt = f"""
    You are an expert veterinary advisor for Indian farmers. Answer the following question about {st.session_state.detected_breed} cattle breed in {selected_lang} language.
    
    Base your answer on this information: {english_info}
    
    Question: {user_question}
    
    Provide a helpful, accurate response in {selected_lang}. If the question is not related to cattle breeding, politely decline to answer.
    """
    
    with st.spinner("Consulting with expert..."):
        response = query_mistral(prompt, api_key)
    
    # Store in chat history
    st.session_state.chat_history.append({"question": user_question, "answer": response})
    
    st.write("**Expert Advice:**")
    st.write(response)

# Display chat history
if st.session_state.chat_history:
    st.divider()
    st.subheader("Chat History")
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Q: {chat['question']}"):
            st.write(f"**A:** {chat['answer']}")

