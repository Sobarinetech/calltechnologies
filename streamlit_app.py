import streamlit as st
import librosa
import whisper
from pyaudioanalysis import audioSegmentation as aS

# Load Whisper model
model = whisper.load_model("base")

# Streamlit App UI
st.title("🎙️ Customer Support Call Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your audio file (e.g., .wav, .flac, .mp3)", type=["wav", "flac", "mp3"])

if uploaded_file is not None:
    # Load audio file
    audio, sample_rate = librosa.load(uploaded_file, sr=None)
    
    # Display audio player
    st.audio(uploaded_file, format="audio/wav")

    # Save audio to temporary file for Whisper and pyAudioAnalysis
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Transcription using Whisper
    transcription_result = model.transcribe("temp_audio.wav")
    text = transcription_result["text"]
    
    # Display Transcript
    st.write("Transcript:")
    st.write(text)

    # Speaker Diarization using pyAudioAnalysis
    [flagsInd, classesAll, acc, CM] = aS.mtFileClassification("temp_audio.wav", "data/svmSM", "svm", True)

    # Display Diarization Segments
    st.write("Diarization Segments:")
    for i, flag in enumerate(flagsInd):
        start_time = i * 0.1  # Segment length in seconds
        speaker = classesAll[flag]
        st.write(f"Speaker {speaker}: {start_time:.2f}s")

# Add any other analyses you wish to include...
