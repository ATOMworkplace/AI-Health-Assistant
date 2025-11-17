import streamlit as st
import json
import os
from datetime import datetime

MOOD_FILE = "mood_tracker.json"

def load_mood_data():
    if not os.path.exists(MOOD_FILE):
        return []
    try:
        # FIX: Added encoding='utf-8' to handle emojis
        with open(MOOD_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # If file is corrupted or has encoding issues, return empty list
        return []

def save_mood_data(entry):
    data = load_mood_data()
    data.append(entry)
    try:
        # FIX: Added encoding='utf-8' and ensure_ascii=False
        with open(MOOD_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving mood: {e}")

def mood_selector():
    st.subheader("How are you feeling?")
    
    moods = ["😊 Happy", "😐 Neutral", "😢 Sad", "😡 Angry", "😰 Anxious", "😴 Tired"]
    selected_mood = st.selectbox("Select your mood", moods, label_visibility="collapsed")
    
    note = st.text_input("Add a note (optional)")
    
    if st.button("Log Mood"):
        entry = {
            "mood": selected_mood,
            "note": note,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_mood_data(entry)
        st.success("Mood logged!")
        return selected_mood
        
    return None