import streamlit as st
import requests
import os
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
    # Perform speaker diarization
    [seg, class_names, accuracy] = aS.mtFileClassification(file_path, 
                                                         "pyAudioAnalysis/data/models/svm_rbf_smote", 
                                                         "svm")
    
    # Collecting speaker segments (start, end) and speaker labels
    speaker_segments = []
    for i, (segment, class_id) in enumerate(zip(seg, class_names)):
        speaker_segments.append((class_id, segment[0], segment[1]))
    
    return speaker_segments

# Function to label speakers in transcription
def label_speakers(transcription, speaker_segments):
    lines = transcription.split('\n')
    labeled_transcription = []

    current_segment_index = 0
    for line in lines:
        # Use the current segment to assign the correct speaker label
        if current_segment_index < len(speaker_segments):
            speaker, start, end = speaker_segments[current_segment_index]
            labeled_transcription.append(f"{speaker}: {line.strip()}")
            current_segment_index += 1
        else:
            labeled_transcription.append(f"Unknown Speaker: {line.strip()}")

    return "\n".join(labeled_transcription)

# Streamlit UI
st.title("🎙️ Call Transcription and Speaker Diarization")
st.write("Upload an audio file, and this app will transcribe it and perform speaker diarization.")

# File uploader
uploaded_file = st.file_uploader("Upload your audio file (wav, flac)", type=["wav", "flac"])

if uploaded_file is not None:
    # Display uploaded audio
    st.audio(uploaded_file, format="audio/wav", start_time=0)
    
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
        speaker_segments = perform_diarization(temp_file_path)
        
        # Label the transcription with speaker labels
        labeled_transcription = label_speakers(transcription, speaker_segments)
        st.text_area("Transcription with Speaker Labels", labeled_transcription, height=300)
        
        # Optional: show the raw transcription as well
        st.subheader("Raw Transcription:")
        st.write(transcription)
    elif "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        st.warning("Unexpected response from the API.")
    
    # Clean up temporary file
    import os
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
