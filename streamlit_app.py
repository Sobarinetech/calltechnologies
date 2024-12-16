import streamlit as st
import requests
import os
import wave
import numpy as np
from pyAudioAnalysis import audioSegmentation as aS

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}

# Function to send the audio file to the Hugging Face API for transcription
def transcribe_audio(file):
    try:
        data = file.read()
        response = requests.post(API_URL, headers=HEADERS, data=data)
        if response.status_code == 200:
            return response.json()  # Return transcription
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# Function to perform speaker diarization using pyAudioAnalysis
def perform_diarization(file_path):
    [seg, class_names, accuracy] = aS.mtFileClassification(file_path, "pyAudioAnalysis/data/models/svm_rbf_smote", "svm")
    return seg, class_names

# Function to label speakers
def label_speakers(transcription, seg, class_names):
    lines = transcription.split('\n')
    labeled_transcription = []
    
    current_speaker = 0  # Start with the first speaker
    for i, line in enumerate(lines):
        # Map the segments to speakers
        speaker = class_names[int(seg[i][1])]
        labeled_transcription.append(f'{speaker}: {line.strip()}')

    return "\n".join(labeled_transcription)

# Streamlit UI
st.title("🎙️ Call Transcription and Analysis")
st.write("Upload an audio file, and this app will transcribe it and perform speaker diarization.")

# File uploader
uploaded_file = st.file_uploader("Upload your audio file", type=["wav", "flac", "mp3"])

if uploaded_file is not None:
    # Display uploaded audio
    st.audio(uploaded_file, format="audio/mp3", start_time=0)
    
    # Save the file to a temporary location for processing
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info("Transcribing audio... Please wait.")
    
    # Perform transcription using Hugging Face API
    result = transcribe_audio(uploaded_file)
    
    # If transcription is successful, perform diarization and display results
    if "text" in result:
        st.success("Transcription Complete:")
        transcription = result["text"]
        
        # Perform speaker diarization
        seg, class_names = perform_diarization(temp_file_path)
        
        # Label the transcription with speaker labels
        labeled_transcription = label_speakers(transcription, seg, class_names)
        st.text_area("Transcription with Speaker Labels", labeled_transcription, height=300)
        
        # Optional: show the raw transcription as well
        st.subheader("Raw Transcription:")
        st.write(transcription)
    elif "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.warning("Unexpected response from the API.")
    
    # Clean up temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
