import streamlit as st
import requests
import librosa
from pyaudioanalysis import audioSegmentation as aS

# Set up Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"

# Retrieve Hugging Face API token from Streamlit secrets
API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Function to send the audio file to the API
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

# Streamlit UI
st.title("🎙️ Customer Support Call Analysis")
st.write("Upload an audio file, and this app will transcribe it using OpenAI Whisper via Hugging Face API and perform speaker diarization.")

# File uploader
uploaded_file = st.file_uploader("Upload your audio file (e.g., .wav, .flac, .mp3)", type=["wav", "flac", "mp3"])

if uploaded_file is not None:
    # Display uploaded audio
    st.audio(uploaded_file, format="audio/wav")
    st.info("Transcribing audio... Please wait.")
    
    # Transcribe the uploaded audio file
    result = transcribe_audio(uploaded_file)
    
    # Display the transcription result
    if "text" in result:
        st.success("Transcription Complete:")
        st.write(result["text"])
    elif "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.warning("Unexpected response from the API.")
    
    # Save audio to temporary file for pyAudioAnalysis
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Speaker Diarization using pyAudioAnalysis
    [flagsInd, classesAll, acc, CM] = aS.mtFileClassification("temp_audio.wav", "data/svmSM", "svm", True)

    # Display Diarization Segments
    st.write("Diarization Segments:")
    for i, flag in enumerate(flagsInd):
        start_time = i * 0.1  # Segment length in seconds
        speaker = classesAll[flag]
        st.write(f"Speaker {speaker}: {start_time:.2f}s")

# Add any other analyses you wish to include...
