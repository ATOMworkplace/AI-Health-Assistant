# mood_selector.py
import streamlit as st
import json
from datetime import datetime
import os

MOOD_FILE = "mood_tracker.json"

# Ensure file exists
if not os.path.exists(MOOD_FILE):
    with open(MOOD_FILE, "w") as f:
        json.dump([], f)


def save_mood(mood):
    """Append mood + timestamp to mood_tracker.json"""
    entry = {
        "mood": mood,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(MOOD_FILE, "r+") as f:
        data = json.load(f)
        data.append(entry)
        f.seek(0)
        json.dump(data, f, indent=4)


def mood_selector():
    st.markdown(
        """
        <h3 style='text-align:center;margin-bottom:5px;'>How are you feeling today?</h3>
        <p style='text-align:center;color:#777;margin-top:-10px;'>Select a mood before starting the chat</p>
        """,
        unsafe_allow_html=True,
    )

    # Emoji buttons
    cols = st.columns(5)
    moods = {
        "😊": "Happy",
        "😐": "Neutral",
        "😔": "Sad",
        "😡": "Angry",
        "😴": "Tired",
    }

    selected = None

    for col, (emoji, label) in zip(cols, moods.items()):
        if col.button(f"{emoji}\n{label}", key=emoji, help=f"Feeling {label}"):
            selected = label

    if selected:
        save_mood(selected)
        st.success(f"Mood recorded: **{selected}**")
        st.session_state["mood_selected"] = True
