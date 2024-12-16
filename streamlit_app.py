import streamlit as st
import requests
from speechbrain.pretrained import SpeakerDiarization

# Set up Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"

# Retrieve Hugging Face API token from Streamlit secrets
API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Function to send the audio file to the API (for transcription)
def transcribe_audio(file):
    try:
        # Read the file as binary
        data = file.read()
        response = requests.post(API_URL, headers=HEADERS, data=data)
        if response.status_code == 200:
            return response.json()  # Return transcription
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# Function for speaker diarization
def diarize_audio(file):
    try:
        # Initialize the speaker diarization pipeline
        diarization = SpeakerDiarization.from_hparams(source="speechbrain/diarization-librispeech", savedir="tmpdir")
        
        # Perform diarization
        diarization_result = diarization.diarize_file(file)
        
        # Process diarization result
        speakers_segments = []
        for segment in diarization_result:
            speakers_segments.append({
                "speaker": segment["speaker"],
                "start": segment["start"],
                "end": segment["end"]
            })
        
        return speakers_segments
    except Exception as e:
        return {"error": str(e)}

# Streamlit UI
st.title("🎙️ Audio Transcription and Speaker Diarization Web App")
st.write("Upload an audio file, and this app will transcribe it using OpenAI Whisper via Hugging Face API and perform speaker diarization.")

# File uploader
uploaded_file = st.file_uploader("Upload your audio file (e.g., .wav, .flac, .mp3)", type=["wav", "flac", "mp3"])

if uploaded_file is not None:
    # Display uploaded audio
    st.audio(uploaded_file, format="audio/mp3", start_time=0)
    
    # Display transcription
    st.info("Transcribing audio... Please wait.")
    result = transcribe_audio(uploaded_file)
    
    # Display transcription result
    if "text" in result:
        st.success("Transcription Complete:")
        st.write(result["text"])
    elif "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.warning("Unexpected response from the API.")
    
    # Perform speaker diarization
    st.info("Performing speaker diarization... Please wait.")
    diarization_result = diarize_audio(uploaded_file)
    
    if "error" in diarization_result:
        st.error(f"Error: {diarization_result['error']}")
    else:
        st.success("Speaker Diarization Result:")
        for segment in diarization_result:
            st.write(f"Speaker {segment['speaker']} | Start: {segment['start']}s | End: {segment['end']}s")

