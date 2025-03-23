import streamlit as st
import whisper
import tempfile
import os
import chromadb
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Set Page Config (First Command)
st.set_page_config(page_title="Subtitle Generator & Search")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="subtitles")

# Load Whisper Model
@st.cache_resource()
def load_model():
    return whisper.load_model("tiny")

whisper_model = load_model()

# Load Sentence Transformer Model (For Semantic Search)
@st.cache_resource()
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# Load and Store Subtitle Data from Pickle
@st.cache_resource()
def load_subtitle_data():
    with open("subtitle_embeddings.pkl", "rb") as f:
        subtitle_data = pickle.load(f)
    #st.success("Pickle data loaded into ChromaDB successfully!")

# Format Time for SRT
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# Transcribe Audio and Store in ChromaDB
@st.cache_resource()
def transcribe_audio(audio_file):
    result = whisper_model.transcribe(audio_file, beam_size=1)
    segments = result['segments']
    subtitles = []
    plain_subtitles = []

    for i, segment in enumerate(segments):
        start_time = format_time(segment['start'])
        end_time = format_time(segment['end'])
        text = segment['text']
        subtitles.append(f"{i+1}\n{start_time} --> {end_time}\n{text}\n")
        plain_subtitles.append(text)

        # Store into ChromaDB with timestamp
        collection.upsert(
            documents=[text],
            metadatas=[{"start": start_time, "end": end_time}],
            ids=[f"transcribe_{i}"]
        )

    return "\n".join(subtitles), " ".join(plain_subtitles)

# Search using ChromaDB
def search_subtitles(query):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    
    if 'documents' in results and results['documents']:
        output = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            output.append(f"📌 {doc} (Start: {metadata['start']}, End: {metadata['end']})")
        return output
    return []

# UI Header
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🎬 AI-Powered Subtitle Generator & Search</h1>
    <p style='text-align: center; font-size: 18px; color: #555;'>Generate subtitles from audio and search within generated subtitles.</p>
    """,
    unsafe_allow_html=True
)

# Load Subtitle Data from Pickle
load_subtitle_data()

# File Upload
st.markdown("### 🔊 Upload Your Audio File")
uploaded_file = st.file_uploader("Supported Formats: MP3, WAV", type=["mp3", "wav"])

# Create Tabs with Color Styling
st.markdown(
    """
    <style>
    div.stTabs [data-baseweb="tab"] {
        background-color: #E3F2FD;
        color: #1976D2;
        font-weight: bold;
    }
    div.stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1976D2;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create Tabs with Colors
tab1, tab2 = st.tabs([
    "🎧 Generate Subtitles",
    "🔎 Search Subtitles"
])

with tab1:
    if uploaded_file:
        st.audio(uploaded_file, format='audio/mp3')
        with st.spinner('Transcribing audio... Please wait!'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                if uploaded_file.name.endswith(".mp3"):
                    audio = AudioSegment.from_mp3(uploaded_file)
                elif uploaded_file.name.endswith(".wav"):
                    audio = AudioSegment.from_wav(uploaded_file)
                audio = audio.set_frame_rate(16000)
                audio.export(temp_audio.name, format="wav")
                temp_audio_path = temp_audio.name

            transcribed_text, plain_text = transcribe_audio(temp_audio_path)
            st.success("Transcription Complete!")

            # Display Subtitles
            st.markdown("### 📜 Generated Subtitles")
            st.text_area("Subtitles (SRT Format)", transcribed_text, height=300)

            # Download Buttons
            st.download_button("📥 Download Subtitles (SRT)", transcribed_text, file_name="subtitles.srt", mime="text/plain")
            st.download_button("📥 Download Subtitles (Plain Text)", plain_text, file_name="subtitles.txt", mime="text/plain")

            os.remove(temp_audio_path)

with tab2:
    st.subheader("🔎 Search for Words or Sentences in Subtitles")
    search_query = st.text_input("Enter a word or sentence")
    if search_query:
        with st.spinner('Searching in ChromaDB...'):
            search_results = search_subtitles(search_query)
            if search_results:
                st.success("Results Found!")
                for result in search_results:
                    st.write(result)
            else:
                st.warning("No results found. Try a different query.")
