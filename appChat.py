import streamlit as st 
import json
import random
import os
import re
import numpy as np
import asyncio
import pickle

try:
    from tensorflow.keras.models import load_model
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
    """Saves chat history WITHOUT causing crashes."""
    file_path = 'chat_history.json'
    
    try:
        last_msg_content = str(messages[-1].get("content", "")) if messages else ""
        _ = sent_model.get_sentiment(last_msg_content)
    except Exception:
        pass

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
    return [simple_lemmatize(word) for word in simple_tokenize(sentence)]

@st.cache_data(show_spinner=False)
def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

@st.cache_data(show_spinner=False)
def predict_class(sentence, _model, words, classes):
    bow = bag_of_words(sentence, words)
    res = _model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

@st.cache_data(show_spinner=False)
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for i in intents_json["intents"]:
            if i["tag"] == tag:
                return random.choice(i["responses"])
    return "I'm not sure I understand. Can you please rephrase?"

async def handle_message_async(prompt, model, words, classes, data):
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

    model_exists = os.path.exists("model.h5")
    words_exists = os.path.exists("words.pkl")
    classes_exists = os.path.exists("classes.pkl")

    if not (model_exists and words_exists and classes_exists):
        with st.spinner("Training model for the first time..."):
            try:
                import train_model
                train_model.main()     # This creates model.h5, words.pkl, classes.pkl
                st.success("Training complete!")
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.stop()

    # Always load saved model files
    model = load_model("model.h5")
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    data = load_intents("intents.json")

    return model, words, classes, data

# ----------------------------------------------------------------


def apply_complete_theme(theme_mode):
    # (KEEPING ALL YOUR STYLING THE SAME)
    ...
    # (Your entire CSS code remains unchanged)
    ...


# --- Main Application ---

async def main():
    st.set_page_config(page_title="AI Health Assistance", page_icon="🤖", layout="wide")
    
    if 'theme' not in st.session_state:
        st.session_state.theme = "Light"
    
    model, words, classes, data = initialize_model_and_data()

    st.markdown(apply_complete_theme(st.session_state.theme), unsafe_allow_html=True)

    st.markdown("""
    <div class="title-box">
        <h2>🤝 AI Health Assistant</h2>
        <p>Here to listen, support, and guide—at your pace.</p>
        <p style="margin-top: 10px;">
            🌈 Designed for <strong>neurodiverse individuals</strong>. Take your time. 😊
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("💡 Options")
        
        theme_option = st.radio("Theme:", ["Light", "Dark"], index=0 if st.session_state.theme=="Light" else 1)
        if theme_option != st.session_state.theme:
            st.session_state.theme = theme_option
            st.rerun()
        
        if st.button("Start a New Chat"):
            st.session_state.messages = []
            save_chat_history([])
            st.rerun()

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

    st.markdown("""
        <hr style="margin-top: 3em; margin-bottom: 1em;">
        <div style="font-size: 0.85rem; color: gray;">
            ⚠️ <strong>Disclaimer:</strong> This chatbot is not a substitute for professional medical advice.
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    asyncio.run(main())

