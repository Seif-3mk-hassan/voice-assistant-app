# app.py
import streamlit as st
from wishper import TTS_class
from the_llm import build_assistant_system

st.set_page_config(page_title="Voice Assistant", page_icon="🎙️", layout="centered")

@st.cache_resource
def load_tts():
    return TTS_class()

@st.cache_resource
def load_graph():
    return build_assistant_system()

tts = load_tts()
graph = load_graph()

if "memory" not in st.session_state:
    st.session_state.memory = []

st.title("🎙️ Voice Assistant")
st.caption("Click the button, speak, then wait for a response.")

# Render conversation history
for turn in st.session_state.memory:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])

col1, col2 = st.columns([1, 4])
with col1:
    speak_clicked = st.button("🎤 Speak")

if speak_clicked:
    with st.spinner("Listening..."):
        text = tts.listen_and_transcribe()

    if not text.strip():
        st.warning("No speech detected — try again.")
    else:
        with st.chat_message("user"):
            st.write(text)

        with st.spinner("Thinking..."):
            content = {
                "name": "Voice Assistant",
                "question": text,
                "answer": "",
                "memory": st.session_state.memory,
            }
            result = graph.invoke(content)

        with st.chat_message("assistant"):
            st.write(result["answer"])

        st.session_state.memory.append({
            "question": result["question"],
            "answer": result["answer"],
        })
        st.rerun()

if st.button("Clear conversation"):
    st.session_state.memory = []
    st.rerun()