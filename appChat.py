import streamlit as st
import json
import random
import os
import re
import numpy as np
import asyncio
import pickle

# --- Imports for Machine Learning ---
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import SGD
except ImportError:
    st.error("TensorFlow is required. Please add it to requirements.txt")

from mood_selector import mood_selector
from sentiment_model import SentimentModel

# --- Initialization ---
sent_model = SentimentModel()

import nltk
nltk.data.path.append("./nltk_data") 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# --- Helper Functions ---

@st.cache_data(show_spinner=False)
def simple_tokenize(text):
    if not text or not isinstance(text, str):
        return []
    return re.findall(r'\w+', text.lower())

@st.cache_data(show_spinner=False)
def simple_lemmatize(word):
    return lemmatizer.lemmatize(word)

@st.cache_data(show_spinner=False)
def load_chat_history():
    file_path = 'chat_history.json'
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except json.JSONDecodeError:
            st.warning("Chat history file is corrupted. Starting with an empty history.")
    return []

def save_chat_history(messages):
    """Saves chat history. Contains the FIX for the AttributeError crash."""
    file_path = 'chat_history.json'
    
    # FIX: Get sentiment of the string content, not the list object
    if messages:
        try:
            last_msg_content = str(messages[-1].get("content", ""))
            sentiment = sent_model.get_sentiment(last_msg_content)
        except Exception:
            pass # Fail silently on sentiment to keep app running

    try:
        with open(file_path, 'w') as file:
            json.dump(messages, file)
    except IOError:
        st.warning("Unable to save chat history.")

@st.cache_data(show_spinner=False)
def load_intents(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading intents file: {e}")
        return {"intents": []}

@st.cache_data(show_spinner=False)
def clean_up_sentence(sentence):
    sentence_words = simple_tokenize(sentence)
    sentence_words = [simple_lemmatize(word) for word in sentence_words]
    return sentence_words

@st.cache_data(show_spinner=False)
def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

@st.cache_data(show_spinner=False)
def predict_class(sentence, _model, words, classes):
    bow = bag_of_words(sentence, words)
    res = _model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

@st.cache_data(show_spinner=False)
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "I'm not sure I understand. Can you please rephrase?"

async def handle_message_async(prompt, model, words, classes, data):
    intents = predict_class(prompt, model, words, classes)
    response = get_response(intents, data)
    return response

@st.cache_resource(show_spinner=False)
def initialize_model_and_data():
    """
    Contains the FIX for Deployment Crash.
    If model.h5 is missing, it trains it automatically.
    """
    if not os.path.exists("model.h5") or not os.path.exists("words.pkl"):
        with st.spinner("First time setup: Training the model..."):
            try:
                import train_model
                train_model.main()
                st.success("Training complete!")
            except Exception as e:
                st.error(f"Failed to train model: {e}")
                st.stop()

    model = load_model("model.h5")
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    data = load_intents("intents.json")

    return model, words, classes, data

def apply_complete_theme(theme_mode):
    """Apply comprehensive theme styling - EXACTLY AS REQUESTED"""
    if theme_mode == "Light":
        return """
        <style>
        body {
            background-color: #f2f8fd;
        }
        .title-box {
            background-color: #e0f0ff;
            padding: 1.5rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
        }
        .chat-message {
            padding: 1rem;
            margin-bottom: 0.8rem;
            border-radius: 12px;
            color: black;
        }
        .user {
            background-color: #D0E8FF;
            text-align: right;
        }
        .assistant {
            background-color: #E6E6FA;
        }
        .sidebar .stButton>button {
            background-color: #d6eaff;
            color: black;
            font-weight: 600;
            border-radius: 10px;
        }
        button[kind="secondary"] {
            border-radius: 15px !important;
            font-size: 20px !important;
            padding: 15px !important;
        }
        button {
            height: 90px;
            width: 110px;
            font-size: 28px;
            text-align: center;
        }
        </style>
        """
    else:  # Dark theme
        return """
        <style>
        /* Main app background */
        .stApp {
            background-color: #1a1d24;
            color: #ffffff;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #262b35;
        }
        
        [data-testid="stSidebar"] > div:first-child {
            background-color: #262b35;
        }
        
        /* Sidebar text */
        [data-testid="stSidebar"] .element-container {
            color: #ffffff;
        }
        
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4 {
            color: #ffffff;
        }
        
        [data-testid="stSidebar"] p {
            color: #e0e0e0;
        }
        
        /* Sidebar buttons */
        [data-testid="stSidebar"] .stButton > button {
            background-color: #363b47;
            color: #ffffff;
            border: 1px solid #4a5060;
            border-radius: 10px;
        }
        
        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #404552;
            border-color: #5a6070;
        }
        
        /* Sidebar radio buttons */
        [data-testid="stSidebar"] .stRadio > label {
            color: #ffffff;
        }
        
        [data-testid="stSidebar"] [data-baseweb="radio"] {
            background-color: #262b35;
        }
        
        /* Top header area */
        header[data-testid="stHeader"] {
            background-color: #262b35;
        }
        
        /* Title box area */
        .title-box {
            background-color: #2d3748;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            border: 1px solid #3d4451;
        }
        
        /* Main title */
        h1 {
            color: #ffffff;
        }
        
        /* Chat messages */
        .stChatMessage {
            background-color: #2d3748;
            border-radius: 10px;
            border: 1px solid #3d4451;
        }
        
        [data-testid="stChatMessageContent"] {
            color: #ffffff;
        }
        
        /* Chat input - MAXIMUM OVERRIDE for dark mode */
        /* Target every possible container and remove borders */
        [data-testid="stChatInput"],
        [data-testid="stChatInput"] *:not(textarea):not(button),
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] > div > div,
        [data-testid="stChatInput"] > div > div > div,
        [data-testid="stChatInput"] div,
        .stChatInput,
        .stChatInput *:not(textarea):not(button),
        .stChatInput > div,
        .stChatInput > div > div,
        .stChatInput div,
        [data-testid="stChatInputContainer"],
        .stChatInputContainer {
            background-color: #1a1d24 !important;
            background: #1a1d24 !important;
            border: none !important;
            box-shadow: none !important;
        }
        
        /* The actual textarea input */
        [data-testid="stChatInput"] textarea,
        .stChatInput textarea,
        textarea[aria-label="Chat input"],
        textarea[placeholder="What's on your mind?"] {
            background-color: #363b47 !important;
            background: #363b47 !important;
            color: #ffffff !important;
            border: 1px solid #4a5060 !important;
            border-radius: 10px !important;
            caret-color: #ffffff !important;
        }
        
        [data-testid="stChatInput"] textarea:focus,
        .stChatInput textarea:focus {
            border-color: #5a6070 !important;
            box-shadow: 0 0 0 1px #5a6070 !important;
            background-color: #363b47 !important;
            background: #363b47 !important;
            outline: none !important;
        }
        
        [data-testid="stChatInput"] textarea::placeholder,
        .stChatInput textarea::placeholder {
            color: #a0a0a0 !important;
        }
        
        /* Bottom container - complete override */
        .stChatFloatingInputContainer,
        [data-testid="stBottom"],
        [data-testid="stBottom"] > div,
        [data-testid="stBottom"] > div > div,
        [data-testid="stBottom"] div,
        section[data-testid="stBottom"],
        section[data-testid="stBottom"] > div,
        section[data-testid="stBottom"] div {
            background-color: #1a1d24 !important;
            background: #1a1d24 !important;
        }
        
        /* Send button styling */
        [data-testid="stChatInput"] button,
        .stChatInput button {
            background-color: #363b47 !important;
            color: #ffffff !important;
            border: 1px solid #4a5060 !important;
        }
        
        [data-testid="stChatInput"] button:hover,
        .stChatInput button:hover {
            background-color: #404552 !important;
        }
        
        /* Dividers */
        hr {
            border-color: #3d4451;
        }
        
        /* General text */
        p, span, div {
            color: #e0e0e0;
        }
        
        /* Markdown text */
        .stMarkdown {
            color: #e0e0e0;
        }
        
        /* Disclaimer styling */
        .stMarkdown strong {
            color: #b0b0b0;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1a1d24;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #3d4451;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #4d5461;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {visibility: hidden;}
        </style>
        """

# --- Main Application ---

async def main():
    st.set_page_config(page_title="AI Health Assistance", page_icon="🤖", layout="wide")
    
    if 'theme' not in st.session_state:
        st.session_state.theme = "Light"
    
    model, words, classes, data = initialize_model_and_data()

    # Apply theme CSS
    st.markdown(apply_complete_theme(st.session_state.theme), unsafe_allow_html=True)

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

    # Sidebar
    with st.sidebar:
        st.header("💡 Options")
        
        theme_option = st.radio(
            "Theme:", ["Light", "Dark"],
            index=0 if st.session_state.theme == "Light" else 1,
            key="theme_radio"
        )
        if theme_option != st.session_state.theme:
            st.session_state.theme = theme_option
            st.rerun()
        
        if st.button("Start a New Chat"):
            st.session_state.messages = []
            save_chat_history(st.session_state.messages)
            st.rerun()

    # Chat Logic
    if 'messages' not in st.session_state:
        st.session_state.messages = load_chat_history()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What's on your mind?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.text("Thinking...")
            
            response = await handle_message_async(prompt, model, words, classes, data)
            
            placeholder.empty()
            placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        save_chat_history(st.session_state.messages)

    # Disclaimer
    st.markdown("""
        <hr style="margin-top: 3em; margin-bottom: 1em;">
        <div style="font-size: 0.85rem; color: gray;">
            ⚠️ <strong>Disclaimer:</strong> This chatbot is an AI-based assistant meant for general wellness and support. 
            It is <strong>not a substitute for professional medical advice</strong>.
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    asyncio.run(main())