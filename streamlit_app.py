import streamlit as st
from pyannote.audio import Pipeline
import requests
import os

# Hugging Face API setup
TRANSCRIPTION_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large"
API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Initialize pyannote.audio diarization pipeline
@st.cache_resource
def load_diarization_pipeline():
    return Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=API_TOKEN)

diarization_pipeline = load_diarization_pipeline()

# Function for transcription using Whisper
def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        audio_data = f.read()

    response = requests.post(TRANSCRIPTION_API_URL, headers=HEADERS, data=audio_data)
    if response.status_code == 200:
        return response.json().get("text", "Transcription failed.")
    else:
        return f"API Error: {response.status_code} - {response.text}"

# Streamlit UI
st.title("🔊 Speaker Diarization and Transcription Web App")
st.write("Upload an audio file, and this app will perform speaker diarization and transcription.")

# File uploader
uploaded_file = st.file_uploader("Upload your audio file (e.g., .wav, .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    temp_file_path = "temp_audio_file.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file, format="audio/wav")
    st.info("Processing audio... This may take a few moments.")

    # Perform diarization
    try:
        diarization = diarization_pipeline(temp_file_path)

        # Display diarization results
        st.success("Diarization Complete!")
        st.write("Speaker Diarization Results:")

        # Convert diarization results to a readable format
        diarization_results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarization_results.append(f"{turn.start:.1f}s - {turn.end:.1f}s: Speaker {speaker}")

        st.write("\n".join(diarization_results))
    except Exception as e:
        st.error(f"An error occurred during diarization: {e}")

    # Perform transcription
    try:
        st.info("Transcribing audio... This may take a moment.")
        transcription_result = transcribe_audio(temp_file_path)

        if transcription_result:
            st.success("Transcription Complete!")
            st.write(transcription_result)
        else:
            st.error("Failed to transcribe the audio.")
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")

    # Clean up temporary file
    os.remove(temp_file_path)
