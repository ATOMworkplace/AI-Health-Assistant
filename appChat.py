import streamlit as st
import json
import random
import os
import re
import numpy as np
import asyncio
import pickle
import traceback # Added for detailed debugging

# --- Imports for Machine Learning ---
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import SGD
except ImportError as e:
    st.error(f"CRITICAL: TensorFlow Import Error. Details: {e}")
    st.stop()

# --- Custom Module Imports ---
try:
    from mood_selector import mood_selector
    from sentiment_model import SentimentModel
except ImportError as e:
    st.error(f"CRITICAL: Module Import Error. Ensure 'mood_selector.py' and 'sentiment_model.py' exist. Details: {e}")

# --- Initialization ---
# Initialize sentiment model globally to avoid reloading
if 'sent_model' not in st.session_state:
    try:
        st.session_state.sent_model = SentimentModel()
    except Exception as e:
        st.error(f"DEBUG: Failed to initialize SentimentModel: {e}")
        
sent_model = st.session_state.get('sent_model', None)

import nltk
# Ensure NLTK data path is correct
nltk.data.path.append("./nltk_data") 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# --- Helper Functions ---

@st.cache_data(show_spinner=False)
def simple_tokenize(text):
    """Basic tokenizer using regex."""
    if not text or not isinstance(text, str):
        return []
    return re.findall(r'\w+', text.lower())

@st.cache_data(show_spinner=False)
def simple_lemmatize(word):
    """Lemmatize words using NLTK."""
    return lemmatizer.lemmatize(word)

def load_chat_history():
    """
    Loads chat history from file.
    """
    file_path = 'chat_history.json'
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # FIX: Automatically reset corrupted file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
            
            st.warning("⚠️ DEBUG: Chat history JSON was corrupted and has been reset.")
            return []
        except Exception as e:
            st.error(f"DEBUG: Error reading chat history: {e}")
            return []
            
    return []

def save_chat_history(messages):
    """
    Saves chat history to JSON.
    """
    file_path = 'chat_history.json'
    
    # Attempt to track sentiment
    if messages and messages[-1]['role'] == 'user' and sent_model:
        try:
            last_msg_content = str(messages[-1].get("content", ""))
            _ = sent_model.get_sentiment(last_msg_content)
        except Exception:
            pass 

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(messages, file, ensure_ascii=False, indent=4)
    except IOError as e:
        st.warning(f"DEBUG: Unable to save chat history: {e}")

@st.cache_data(show_spinner=False)
def load_intents(file_path):
    """Safely load the intents JSON file with UTF-8 encoding."""
    if not os.path.exists(file_path):
        st.error(f"DEBUG: Intents file NOT FOUND at: {os.path.abspath(file_path)}")
        return {"intents": []}

    try:
        # The fix is adding: encoding="utf-8"
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except UnicodeDecodeError as e:
        st.error(f"DEBUG: Encoding Error in '{file_path}'. Try saving it as UTF-8. Details: {e}")
        st.code(traceback.format_exc()) # Show full traceback
        return {"intents": []}
    except json.JSONDecodeError as e:
        st.error(f"DEBUG: Invalid JSON format in '{file_path}'. Details: {e}")
        st.code(traceback.format_exc())
        return {"intents": []}
    except Exception as e:
        st.error(f"DEBUG: Unexpected error loading intents: {e}")
        st.code(traceback.format_exc())
        return {"intents": []}

@st.cache_data(show_spinner=False)
def clean_up_sentence(sentence):
    """Tokenize and lemmatize a sentence."""
    sentence_words = simple_tokenize(sentence)
    sentence_words = [simple_lemmatize(word) for word in sentence_words]
    return sentence_words

@st.cache_data(show_spinner=False)
def bag_of_words(sentence, words):
    """Convert sentence to bag-of-words array."""
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

@st.cache_data(show_spinner=False)
def predict_class(sentence, _model, words, classes):
    """Predict the intent of the sentence using the Keras model."""
    try:
        bow = bag_of_words(sentence, words)
        res = _model.predict(np.array([bow]), verbose=0)[0]
        
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        
        # Sort by probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
        return return_list
    except Exception as e:
        st.error(f"DEBUG: Prediction Error: {e}")
        return []

@st.cache_data(show_spinner=False)
def get_response(intents_list, intents_json):
    """Retrieve a random response for the predicted intent."""
    if intents_list:
        tag = intents_list[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    
    return "I'm not sure I understand. Can you please rephrase?"

async def handle_message_async(prompt, model, words, classes, data):
    """Async wrapper for prediction and response generation."""
    intents = predict_class(prompt, model, words, classes)
    return get_response(intents, data)

# ----------------------------------------------------------------
# 🔥 UPDATED: LOAD MODEL FROM FILE, DO NOT RETRAIN EVERY RUN
# ----------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def initialize_model_and_data():
    """
    Loads the trained model IF it exists.
    Trains automatically ONLY if model files do not exist.
    """
    # Check ALL required files. If ANY is missing, we must retrain.
    model_exists = os.path.exists("model.h5")
    words_exists = os.path.exists("words.pkl")
    classes_exists = os.path.exists("classes.pkl")

    if not (model_exists and words_exists and classes_exists):
        with st.spinner("First time setup: Training the model..."):
            try:
                import train_model
                train_model.main()
                st.success("Training complete!")
            except Exception as e:
                st.error(f"DEBUG: Failed to train model during setup: {e}")
                st.code(traceback.format_exc())
                st.stop()

    # Load resources with specific debug checks
    try:
        if not os.path.exists("model.h5"): raise FileNotFoundError("model.h5 missing")
        model = load_model("model.h5")

        if not os.path.exists("words.pkl"): raise FileNotFoundError("words.pkl missing")
        words = pickle.load(open("words.pkl", "rb"))

        if not os.path.exists("classes.pkl"): raise FileNotFoundError("classes.pkl missing")
        classes = pickle.load(open("classes.pkl", "rb"))
        
        if not os.path.exists("intents.json"): raise FileNotFoundError("intents.json missing")
        data = load_intents("intents.json")
        
        return model, words, classes, data

    except Exception as e:
        st.error(f"DEBUG: Error loading model resources: {e}")
        st.code(traceback.format_exc())
        st.stop()

# ----------------------------------------------------------------

def apply_complete_theme(theme_mode):
    """Apply comprehensive theme styling for Light and Dark modes."""
    if theme_mode == "Light":
        return """
        <style>
        body { background-color: #f2f8fd; }
        .title-box { background-color: #e0f0ff; padding: 1.5rem; border-radius: 20px; text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .chat-message { padding: 1rem; margin-bottom: 0.8rem; border-radius: 12px; color: black; }
        .user { background-color: #D0E8FF; text-align: right; }
        .assistant { background-color: #E6E6FA; }
        .stButton>button { border-radius: 10px; font-weight: 600; }
        .sidebar .stButton>button { background-color: #d6eaff; color: black; }
        </style>
        """
    else:  # Dark theme
        return """
        <style>
        .stApp { background-color: #1a1d24; color: #ffffff; }
        [data-testid="stSidebar"] { background-color: #262b35; }
        h1, h2, h3, h4, p, span, div, label { color: #e0e0e0 !important; }
        .title-box { background-color: #2d3748; padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem; border: 1px solid #3d4451; }
        .stChatMessage { background-color: #2d3748; border-radius: 10px; border: 1px solid #3d4451; }
        [data-testid="stChatInput"] textarea { background-color: #363b47 !important; color: #ffffff !important; border: 1px solid #4a5060 !important; }
        .stButton > button { background-color: #363b47; color: #ffffff; border: 1px solid #4a5060; }
        .stButton > button:hover { background-color: #404552; }
        ::-webkit-scrollbar { width: 10px; }
        ::-webkit-scrollbar-track { background: #1a1d24; }
        ::-webkit-scrollbar-thumb { background: #3d4451; border-radius: 5px; }
        </style>
        """

# --- Main Application ---

async def main():
    st.set_page_config(page_title="AI Health Assistance", page_icon="🤖", layout="wide")
    
    # --- CRITICAL: Initialize Session State Messages FIRST ---
    if 'messages' not in st.session_state:
        st.session_state.messages = load_chat_history()

    # Initialize Session State for Theme
    if 'theme' not in st.session_state:
        st.session_state.theme = "Light"
    
    # Initialize Model
    model, words, classes, data = initialize_model_and_data()

    # Apply Theme
    st.markdown(apply_complete_theme(st.session_state.theme), unsafe_allow_html=True)

    # --- Header Section ---
    st.markdown("""
    <div class="title-box">
        <h2>🤝 AI Health Assistant</h2>
        <p>Here to listen, support, and guide—at your pace.</p>
        <p style="margin-top: 10px;">
            🌈 This chatbot is designed to support <strong>neurodiverse individuals</strong>. <br>
            You're in control — feel free to ask at your own pace. 😊
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.header("🧠 Mood Tracker")
        
        # Mood Selector Logic
        try:
            current_mood = mood_selector()
            if current_mood:
                st.info(f"Current Mood: {current_mood}")
        except Exception as e:
            st.error(f"DEBUG: Mood Selector Crashed: {e}")

        st.markdown("---")
        st.header("💡 Options")
        
        # Theme Toggle
        theme_option = st.radio(
            "Theme:", ["Light", "Dark"],
            index=0 if st.session_state.theme == "Light" else 1,
            key="theme_radio"
        )
        if theme_option != st.session_state.theme:
            st.session_state.theme = theme_option
            st.rerun()
        
        # Download History
        chat_json = json.dumps(st.session_state.messages, indent=4, ensure_ascii=False)
        st.download_button(
            label="📥 Download Chat History",
            data=chat_json,
            file_name="chat_history.json",
            mime="application/json",
        )
        
        # New Chat Button
        if st.button("Start a New Chat", type="primary"):
            st.session_state.messages = []
            save_chat_history(st.session_state.messages)
            st.rerun()
            
        st.markdown("---")
        st.markdown("Made with ❤️ for Health")

    # --- Chat Logic ---

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Input
    if prompt := st.chat_input("What's on your mind?"):
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.text("Thinking...")
            
            # Async prediction
            response = await handle_message_async(prompt, model, words, classes, data)
            
            placeholder.empty()
            placeholder.markdown(response)
            
            # Add assistant message to state
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Sync to File
        save_chat_history(st.session_state.messages)

    # --- Footer / Disclaimer ---
    st.markdown("""
        <hr style="margin-top: 3em; margin-bottom: 1em;">
        <div style="font-size: 0.85rem; color: gray; text-align: center;">
            ⚠️ <strong>Disclaimer:</strong> This chatbot is an AI-based assistant meant for general wellness and support. 
            It is <strong>not a substitute for professional medical advice</strong>. <br>
            If you are in crisis, please contact emergency services immediately.
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    asyncio.run(main())